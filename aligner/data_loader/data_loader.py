from itertools import cycle
from pathlib import Path
from torch.utils.data import IterableDataset
from typing import Literal, List, Dict, Iterator
import torch
import torch.nn.functional as F
import json
import os
from ..utils.constants import *
from .utils import load_spectrogram, unfold_spectrogram, \
        midi_to_pandas, midi_to_tokens, midi_to_matrix, \
        find_random_subscore_covering_timestamps
import pandas as pd


class MaestroDataset(IterableDataset):

    def __init__(self, 
                 root_dir: str, # path to MAESTRO dataset
                 split: Literal['train', 'validation', 'test'] = 'train'):

        self.split = split
        self.metadata = json.load(open(os.path.join(root_dir, 'maestro-v3.0.0.json')))
        
        all_midi_uris = list(Path(root_dir).glob("**/*.mid")) + list(Path(root_dir).glob('**/*.midi'))
        
        # Load midi paths corresponding to split
        self.midi_uris: List[str] = []
        for idx in range(len(all_midi_uris)):
            if self.metadata['split'][str(idx)] == split:
                self.midi_uris.append(str(all_midi_uris[idx]))
        
        # Load wav paths corresponding to split
        self.wav_uris: List[str] = []
        for path in self.midi_uris:
            self.wav_uris.append(str(path).replace('.midi', '.wav'))


    def _preprocess(self) -> Iterator[Dict]:

        for wav_uri, midi_uri in zip(self.wav_uris, self.midi_uris):

            signal, timestamps = load_spectrogram(wav_uri)
            signal_clips, timestamps_clips = unfold_spectrogram(signal, timestamps)
            full_midi: pd.DataFrame = midi_to_pandas(midi_uri, EVENT_RESOLUTION)

            for signal_clip, timestamps_clip in zip(signal_clips, timestamps_clips):
                
                if self.split == 'train':
                    midi = find_random_subscore_covering_timestamps(full_midi, timestamps)
                else:
                    midi = full_midi

                encoding: Dict = midi_to_tokens(midi)
                Y: torch.Tensor = midi_to_matrix(midi, timestamps_clip)

                row = {
                    'wav_uri': wav_uri,
                    'midi_uri': midi_uri,
                    'audio_clip': signal_clip,
                    'Y': Y
                } | encoding

                yield row


    def __len__(self): 
        return len(self.midi_paths)
    

    def __iter__(self) -> Iterator[Dict]:
        """Returns a data sample generator.

        Yields:
            Dict[str, torch.Tensor]: Dictionary with the following fields:
                (0) "wav_uri": absolute path to audio file
                (1) "midi_uri": absolute path to midi file
                (2) "audio_clip": ~10 sec spectrogram. Size: (n_mels, n_frames).
                (3) "Y": alignment matrix. Size: (n_frames, n_events).
                (4) "input_ids": Integer array representation (tokenization) of the MIDI.
                    This is the encoding that the transformer model receives. During
                    tokenization, MIDI-scaled pitches [0, 121] are directly taken as unique 
                    tokens, and special tokens such as [BOS] and [event] are attached. See
                    `TOKEN_ID` in `constants.py` for their integer codes. Size: (n_tokens,).
                (5) "event_mask": Array filled with 0s and 1s, where 1s fill positions where
                    event markers lie in "input_ids". Size: (n_tokens,).
                (6) "attn_mask": mask for self-attention. Size: (n_tokens, n_tokens).
        """
        if self.split == 'train':
            return cycle(self._preprocess())
        else:
            return self._preprocess()


    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """Used for DataLoader batch processing"""

        wav_uris: List[str] = [item['wav_uri'] for item in batch]
        midi_uris: List[str] = [item['midi_uri'] for item in batch]

        audio_clip_b: torch.Tensor = torch.stack([item['audio_clip'] for item in batch])

        # Batch input_ids
        tensors = [item['input_ids'] for item in batch]
        max_n_tokens = max(len(t) for t in tensors)
        input_ids_b: torch.Tensor = torch.stack(
            [F.pad(t, (0, max_n_tokens-len(t)), value=TOKEN_ID['[PAD]']) for t in tensors]
        )
        
        # Batch event_mask
        tensors = [item['event_mask'] for item in batch]
        event_mask_b: torch.Tensor = torch.stack(
            [F.pad(t, (0, max_n_tokens-len(t)), value=0) for t in tensors]
        )

        # Batch attn_mask
        tensors = [item['attn_mask'] for item in batch]
        attn_mask_b: torch.Tensor = torch.stack(
            [F.pad(t, 
                (0, max_n_tokens-t.size(1), 0, max_n_tokens-t.size(0)), value=0
            ) for t in tensors]
        )

        # Batch alignment matrix Y
        # TODO: Can't turn this into a tensor!
        # because matrix Y scales by n_events, not n_tokens.
        Y_b: List[torch.Tensor] = [item['Y'] for item in batch]

        return {
            "wav_uris": wav_uris,           # Size: (B,)
            "midi_uris": midi_uris,         # Size: (B,)
            "audio_clip_b": audio_clip_b,   # Size: (B, N_MELS, N_FRAMES_PER_CLIP)
            "Y_b": Y_b,                     # Size: (B) * (N_FRAMES_PER_CLIP, n_events)
            "input_ids_b": input_ids_b,     # Size: (B, max_n_tokens)
            "event_mask_b": event_mask_b,   # Size: (B, max_n_tokens)
            "attn_mask_b": attn_mask_b,     # Size: (B, max_n_tokens, max_n_tokens)
        }

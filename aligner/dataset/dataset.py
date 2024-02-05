from itertools import cycle
from pathlib import Path
from torch.utils.data import IterableDataset
from collections.abc import Sequence
from typing import Literal, List, Iterator, NamedTuple, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor, LongTensor
import os
from ..utils.constants import *
from .toolbox import ParsedMIDI, load_spectrogram, unfold_spectrogram, sample_interval
import pandas as pd
from pandas import DataFrame



class MaestroDataset(IterableDataset):


    def __init__(self, 
                 root_dir: str, # path to MAESTRO dataset
                 split: Literal['train', 'validation', 'test'] = 'train'):

        if split not in {'train', 'validation', 'test'}:
            raise TypeError("Allowed values for split: 'train', 'validation', 'test'")

        self.split = split
        self.metadata = pd.read_csv(os.path.join(root_dir, "maestro-v3.0.0.csv"))
        
        metadata_split = self.metadata[self.metadata['split'] == split]
        self.ids: Sequence[int] = metadata_split.index
        self.midi_uris: Sequence[str] = metadata_split['midi_filename'].apply(
                lambda filename: os.path.join(root_dir, filename))
        self.audio_uris: Sequence[str] = metadata_split['audio_filename'].apply(
                lambda filename: os.path.join(root_dir, filename))
        

    class Item(NamedTuple):
        """Return type of the dataset iterator when split is 'train' or 
        'validation'.

        Args:
            id_ (Tuple[int, int]): Unique identifier of the item. First index
                corresponds to the index of the audio file as written in
                metadata.csv. The second index corresponds to the clip number
                of this audio clip within the whole audio file that contains the 
                clip.
            audio_frames (Tensor): ~10 sec spectrogram. 
                Size: (N_MELS, N_FRAMES_PER_CLIP).
            score_ids (LongTensor): Tokenization (int array representation) of the MIDI. 
                This is the encoding that the transformer receives. Size: (n_tokens,).
            score_event_mask (LongTensor): Marks the positions of event markers in the
                score_ids with 1s, the rest with 0s. Size: (n_tokens,)
            score_attn_mask (LongTensor): Self-attention mask for the score encoder.
                Size: (n_tokens, n_tokens).
            proj_to_evt (LongTensor): Linear projection matrix that extracts only the
                event markers from the tokenization. Size: (n_events, n_tokens).
            Y (Tensor): gold alignment matrix. Size: (N_FRAMES_PER_CLIP, n_events).
        """
        id_: Tuple[int, int]
        audio_frames: Tensor
        score_ids: LongTensor
        score_event_mask: LongTensor
        score_attn_mask: LongTensor
        proj_to_evt: LongTensor
        Y: LongTensor


    class ItemWithMetadata(Item):
        """Return type of the dataset iterator when split is 'test'. Extends
        the `Item` type above.

        Args:
            audio (Tensor): Full spectrogram that the current clip belongs to.
                Size: (n_frames, N_MELS)
            midi (DataFrame): Pandas dataframe representation of the full midi.
            audio_frame_interval (Tuple[int, int]): starting frame index and 
                ending frame index of the current audio clip.
            event_interval (Tuple[int, int]): starting event index and ending
                frame index of the current event clip.
        """
        audio: Tensor
        midi: DataFrame
        audio_frame_interval: Tuple[int, int]
        event_interval: Tuple[int, int]


    class Batch(NamedTuple):
        """Return type of the dataloader when `func:collate_fn` is applied.
        The suffix  `_b` denotes that these fields are the batched versions of
        the fields in `Item`, whose 0th dimension is now the batch size.
        """
        id_b: List[Tuple[int, int]]
        audio_frames_b: Tensor              # (B, N_FRAMES_PER_CLIP, N_MELS)
        score_ids_b: LongTensor             # (B, max_n_tokens)
        score_event_mask_b: LongTensor      # (B, max_n_tokens)
        score_attn_mask_b: LongTensor       # (B, max_n_tokens, max_n_tokens)
        proj_to_evt_b: List[LongTensor]     # (B) (n_events, max_n_tokens)
        Y_b: LongTensor                     # (B, N_FRAMES_PER_CLIP, max_n_tokens)


    def _preprocess(self) -> Iterator[Item]:

        for id_, audio_uri, midi_uri in zip(self.ids, self.audio_uris, self.midi_uris):

            signal = load_spectrogram(audio_uri)
            signal_clips = unfold_spectrogram(signal)
            n_frames = len(signal)
            midi = ParsedMIDI(midi_uri, n_frames)

            for ci, signal_clip in enumerate(signal_clips):
                assert len(signal_clip) == N_FRAMES_PER_CLIP

                _afi1 = ci * N_FRAMES_PER_STRIDE
                _afi2 = _afi1 + N_FRAMES_PER_CLIP

                if self.split == 'train':
                    while True:
                        # Loop until we have a subsequence of events
                        # that doesn't exceed the max number of tokens
                        afi1, afi2 = sample_interval(0, _afi1, _afi2, n_frames)
                        ei1, ei2 = midi.fi2ei[afi1], midi.fi2ei[afi2-1]+1
                        score_ids = midi.encode(ei1, ei2, return_tuple=False) # O(1)
                        if len(score_ids) < MAX_N_TOKENS: break
                else:
                    afi1, afi2 = _afi1, _afi2
                    ei1, ei2 = midi.fi2ei[afi1], midi.fi2ei[afi2-1]+1

                encoding: ParsedMIDI.Encoding = midi.encode(ei1, ei2)
                Y: Tensor = midi.align(afi1, afi2, ei1, ei2)
                
                item = MaestroDataset.Item(
                    id_=(id_, ci),
                    audio_frames=signal_clip,
                    Y=Y,
                    **encoding._asdict(),
                )

                if self.split == 'test':
                    item = MaestroDataset.ItemWithMetadata(
                        audio=signal,
                        midi=midi.midi,
                        audio_frame_interval=(afi1, afi2),
                        event_interval=(ei1, ei2),
                        **item._asdict(),
                    )

                yield item


    def __len__(self): 
        return len(self.midi_paths)
    

    def __iter__(self) -> Iterator[Item]:
        """Iterator for the MAESTRO dataset.

        Yields:
            Item: See the doc above.
        """
        if self.split == 'train':
            return cycle(self._preprocess())
        else:
            return self._preprocess()


    @staticmethod
    def collate_fn(batch: List[Item]) -> Batch:
        
        batch_size = len(batch)
        id_b = [item.id_ for item in batch]

        # Batch spectrogram
        audio_frames_b: Tensor = torch.stack([item.audio_frames for item in batch])
        assert audio_frames_b.shape == (batch_size, N_FRAMES_PER_CLIP, N_MELS)

        # Batch score_ids
        tensors = [item.score_ids for item in batch]
        max_n_tokens = max(len(t) for t in tensors)
        score_ids_b: LongTensor = torch.stack(
            [F.pad(t, (0, max_n_tokens-len(t)), value=TOKEN_ID['[PAD]']) for t in tensors]
        )
        assert score_ids_b.shape == (batch_size, max_n_tokens)

        # Batch score_event_mask
        tensors = [item.score_event_mask for item in batch]
        score_event_mask_b: LongTensor = torch.stack(
            [F.pad(t, (0, max_n_tokens-len(t)), value=0) for t in tensors]
        )
        assert score_event_mask_b.shape == (batch_size, max_n_tokens)

        # Batch score_attn_mask
        tensors = [item.score_attn_mask for item in batch]
        score_attn_mask_b: LongTensor = torch.stack(
            [F.pad(t, 
                (0, max_n_tokens-t.size(1), 0, max_n_tokens-t.size(0)), value=0
            ) for t in tensors]
        )
        assert score_attn_mask_b.shape == (batch_size, max_n_tokens, max_n_tokens)

        # Batch alignment matrix Y
        Y_b, proj_to_evt_b = [], []
        for item in batch:
            Y, proj = item.Y, item.proj_to_evt
            proj = F.pad(proj, (0, max_n_tokens - proj.size(1)), value=0)
            Y_token = Y @ proj
            Y_b.append(Y_token)
            proj_to_evt_b.append(proj)
        Y_b: LongTensor = torch.stack(Y_b)
        assert Y_b.shape == (batch_size, N_FRAMES_PER_CLIP, max_n_tokens)

        return MaestroDataset.Batch(
            id_b=id_b,
            audio_frames_b=audio_frames_b,
            score_ids_b=score_ids_b,
            score_event_mask_b=score_event_mask_b,
            score_attn_mask_b=score_attn_mask_b,
            proj_to_evt_b=proj_to_evt_b,
            Y_b=Y_b
        )

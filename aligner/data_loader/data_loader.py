import torch
import json
import os
from itertools import cycle
from pathlib import Path
from torch.utils.data import IterableDataset
from typing import Literal, List, Dict, Iterator
from aligner.utils.constants import *
from .utils import load_spectrogram, unfold_spectrogram, \
        midi_to_pandas, midi_to_tokens, midi_to_matrix, \
        find_random_subscore_covering_timestamps
import pandas as pd


class MaestroDataset(IterableDataset):

    @classmethod
    def __init__(self, 
                 root_dir: str,
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


    @classmethod
    def _preprocess(self) -> Iterator[Dict[str, torch.Tensor]]:

        for wav_uri, midi_uri in zip(self.wav_uris, self.midi_uris):

            signal, timestamps = load_spectrogram(wav_uri)
            signal_clips, timestamps_clips = unfold_spectrogram(signal, timestamps)
            midi_df: pd.DataFrame = midi_to_pandas(midi_uri, EVENT_RESOLUTION)

            for signal_clip, timestamps_clip in zip(signal_clips, timestamps_clips):

                midi_subdf = find_random_subscore_covering_timestamps(midi_df, timestamps)
                encoding: Dict[str, torch.Tensor] = midi_to_tokens(midi_subdf)
                Y = midi_to_matrix(midi_subdf, timestamps_clip)

                row = {
                    "audio_clip": signal_clip,
                    "Y": Y
                } | encoding

                yield row


    def __len__(self) -> int: 
        return len(self.midi_paths)
    

    def __iter__(self):
        if self.split == 'train':
            return cycle(self._preprocess())
        else:
            return self._preprocess()

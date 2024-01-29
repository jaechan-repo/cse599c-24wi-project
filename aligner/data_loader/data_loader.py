from pathlib import Path
import torch
from torch.utils.data import IterableDataset
import torch.nn.functional as F
import torchaudio
from typing import Literal, List, Tuple
import json
import os
import numpy as np
from ..utils.constants import *
import pretty_midi as pm
import pandas as pd
from .utils import load_spectrogram, unfold_spectrogram, midi_to_pandas, \
        match_clip, encode_midi

# TODO: work in progress

class MaestroDataset(IterableDataset):

    def __init__(self, 
                 root_dir: str,
                 split: Literal['train', 'val', 'test'] = 'train'):
        self.split = split
        self.metadata = json.load(open(os.path.join(root_dir, 'maestro-v3.json')))
        
        all_paths = list(Path(root_dir).glob("**/*.mid")) + list(Path(root_dir).glob('**/*.midi'))
        
        # Load midi paths corresponding to split
        self.midi_paths: List[str] = []
        for idx in range(len(self.paths)):
            if self.metadata['split'][str(idx)] == split:
                self.midi_paths.append(str(all_paths[idx]))
        
        # Load wav paths corresponding to split
        self.wav_paths: List[str] = []
        for path in self.midi_paths:
            self.wav_paths.append(str(path).replace('.midi', '.wav'))

    

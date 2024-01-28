from pathlib import Path, PosixPath
import torch
from torch.utils.data import IterableDataset
import torchaudio
from typing import Literal, List
import json
import os
import librosa
import numpy as np
from ..utils.constants import *
import pretty_midi as pm


"""
TODO: work in progress
"""


_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=FFT_SIZE,
    hop_length=HOP_WIDTH,
    n_mels=NUM_MEL_BINS
)


def load_mel_spectrogram(audio_path: str) -> torch.Tensor:
    """Loads Mel spectrogram given the audio path

    :param str audio_path: Path to audio file, e.g. *.wav, *.mp3
    :return torch.Tensor: 2D Mel spectrogram of size (n_mels, num_frames)
    """
    signal, sr = torchaudio.load(audio_path)

    # Resample if the audio's sr differs from our target sr
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        signal = resampler(signal)

    # Mix down if multi-channel
    signal = torch.mean(signal, dim=0)
    
    # Extract mel spectrogram
    signal = _transform(signal)
    return signal


def encode_midi(midi_path: str,
                resolution: float = EVENT_RESOLUTION) -> torch.Tensor:
    """Convert MIDI file to integer ids

    :param str midi_path: Path to midi file, e.g. *.mid, *.midi
    :param float resolution: resolution for which multiple MIDI note events may be 
        considered as the same event in our encoding. Defaults to EVENT_RESOLUTION
    :return torch.Tensor: 1D tensor of integers
    """
    # FIXME optimize further

    midi_data = pm.PrettyMIDI(midi_path)

    # (onset, offset, pitch)
    df_oop = []
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                df_oop.append({
                    'onset': note.start,    # in miliseconds
                    'offset': note.end,     # in miliseconds
                    'pitch': int(note.pitch)
                })
    df_oop = pd.DataFrame(df_oop)
    df_oop = df_oop.sort_values("onset").reset_index(drop=True)

    # (onset, pitch) - includes silence pitch "[null]"
    h_off = [(0, token_id['[null]'])]
    df_op = []
    for i, row in df_oep.iterrows():
        onset, offset = row['onset'], row['offset']
        prev_offset = None
        while h_off and h_off[0][0] <= row['onset']:
            prev_offset, _ = heappop(h_off)
        if not h_off and prev_offset is not None:
            df_op.append((prev_offset, token_id['[null]']))  # onset of silence (-1) is previous offset
        heappush(h_off, (offset, i))        # store (offset, index) as key for heap
        df_op.append((onset, int(row['pitch'])))
    df_op = pd.DataFrame(df_op, columns=['onset', 'pitch'])

    # approximate the onset timestamps by the resolution
    df_op['onset'] = df_op['onset'].apply(lambda x: round(x / resolution) * resolution)

    # convert to encoding
    encoded_midi = []
    for i, row in df_op.iterrows():
        if i == 0 or row['onset'] != prev_onset:
            encoded_midi.append(token_id['[event]'])
        encoded_midi.append(int(row['pitch']))
        prev_onset = row['onset']

    return torch.tensor(encoded_midi, dtype=torch.int)


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

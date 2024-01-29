import torch
import torch.nn.functional as F
import torchaudio
from typing import List, Optional, Tuple
from ..utils.constants import *
import pretty_midi as pm
import pandas as pd


def _find_range(arr: List[float], min_val: float, max_val: float):
    
    def binary_search_left(arr, target):
        low, high = 0, len(arr) - 1
        while low <= high:
            mid = low + (high - low) // 2
            if arr[mid] < target:
                low = mid + 1
            else:
                high = mid - 1
        return low

    def binary_search_right(arr, target):
        low, high = 0, len(arr) - 1
        while low <= high:
            mid = low + (high - low) // 2
            if arr[mid] <= target:
                low = mid + 1
            else:
                high = mid - 1
        return high

    start = binary_search_left(arr, min_val)
    end = binary_search_right(arr, max_val)

    if start >= len(arr) or arr[start] > max_val:
        return -1, -1
    if end < 0 or arr[end] < min_val:
        return -1, -1

    return start, end


_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    center=True     # pad waveform on both sides s.t. t'th frame centered at t * hop_length
)


def load_spectrogram(uri: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Loads Mel spectrogram given the audio path

    :param str uri: Path to audio file, e.g. *.wav, *.mp3
    :return Tuple[torch.Tensor, torch.Tensor]: 
        (0) Mel spectrogram of size (n_mels, n_frames)
        (1) frame timings (in ms) of size (n_frames)
    """

    signal, sr = torchaudio.load(uri)

    # Resample if the audio's sr differs from our target sr
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        signal = resampler(signal)

    # Mix down if multi-channel
    signal = torch.mean(signal, dim=0)
    
    # Extract mel spectrogram
    signal: torch.Tensor = _transform(signal)
    timing = torch.tensor([i * AUDIO_RESOLUTION for i in range(signal.size()[-1])])

    return signal, timing


def _pad_spectrogram(signal: torch.Tensor, timing: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    
    n_frames = len(timing)
    min_size = max(N_FRAMES_PER_CLIP, n_frames)

    if min_size % N_FRAMES_PER_STRIDE != 0:
        padding_size = min_size + N_FRAMES_PER_STRIDE - (min_size % N_FRAMES_PER_STRIDE)
    else:
        padding_size = min_size
    
    signal = F.pad(signal, (0, padding_size), value=0)
    timing = F.pad(timing, (padding_size), value=timing[-1])

    return signal, timing


def unfold_spectrogram(signal: torch.Tensor, timing: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unfolds the spectrogram into overlapping clips.
    Applies approrpiate padding before unfolidng.

    :param torch.Tensor signal: Mel spectrogram of size (n_mels, n_frames)
    :param torch.Tensor timing: Timing in ms of size (n_frames)
    :return Tuple[torch.Tensor, torch.Tensor]:
        (0) Clipped spectrogram. Size: (n_clips, n_mels, n_frames)
        (1) Clipped timing. Size: (n_clips, n_frames)
    """
    signal, timing = _pad_spectrogram(signal, timing)
    signals = signal.unfold(-1, N_FRAMES_PER_CLIP, N_FRAMES_PER_STRIDE)
    timings = timing.unfold(-1, N_FRAMES_PER_CLIP, N_FRAMES_PER_STRIDE)
    return signals, timings


def midi_to_pandas(uri: str, resolution: Optional[float] = None) -> pd.DataFrame:
    """Read MIDI and convert to Pandas DataFrame.

    :param str uri: Path to midi file
    :param Optional[float] resolution: Defaults to None. 
        If specified, apply the resolution and groups the notes by time events.
            columns=['event': float, 'onset': List[float], 'pitch': List[int]]
        If NOT specified, simply return:
            columns=['onset': float, 'pitch': int]
    :return pd.DataFrame:
    """

    midi_pretty = pm.PrettyMIDI(uri)

    midi = []
    instrument = midi_pretty.instruments[0]   # TODO for v1: multi-instrument support
    if not instrument.is_drum:
        for note in instrument.notes:
            midi.append((note.start, int(note.pitch)))
    midi = pd.DataFrame(midi, columns=['onset', 'pitch']).sort_values('onset').reset_index(drop=True)

    if resolution is not None:
        # round the onset timestamps to the NEAREST MULTIPLE of the resolution
        midi['event'] = midi['onset'].apply(
                lambda x: round(x / EVENT_RESOLUTION) * EVENT_RESOLUTION)
        midi = midi.groupby('event').agg({'onset': list, 'pitch': list}).reset_index()   
        midi = midi.sort_values('event').reset_index(drop=True)

    return midi


def match_clip(timing_clip: torch.Tensor, onsets: List[float]) -> List[Tuple[int,int]]:
    """Obtain where in the score the audio clip occurs

    :param torch.Tensor timing_clip: timestamps of the frames of the audio clip. Size (N_FRAMES_PER_CLIP)
    :param List[float] onsets: SORTED list of event timestamps, extracted from the score
    :return Tuple[int,int]: the boundary indices of the smallest onset interval (inclusive) that covers the timing
    """
    start, end = timing_clip[0], timing_clip[-1]
    return _find_range(onsets, start, end)


def encode_midi(midi: str | pd.DataFrame) -> torch.Tensor:

    if isinstance(midi, str):
        midi = midi_to_pandas(midi, EVENT_RESOLUTION)

    encoded_midi: List[int] = []
    for _, row in midi.iterrows():
        encoded_midi.append(TOKEN_ID['[event]'])
        encoded_midi += row['pitch']

    return torch.tensor(encoded_midi, dtype=torch.int)

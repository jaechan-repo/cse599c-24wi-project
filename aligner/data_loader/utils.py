import torch
import torch.nn.functional as F
import torchaudio
from typing import List, Optional, Tuple, Dict
from aligner.utils.constants import *
import pretty_midi as pm
import pandas as pd
from random import randrange


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
        (1) frame timestamps (in ms) of size (n_frames)
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
    timestamps = torch.tensor([i * AUDIO_RESOLUTION for i in range(signal.size()[-1])])

    return signal, timestamps


def pad_spectrogram(signal: torch.Tensor, timestamps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """TODO comment

    :param torch.Tensor signal: _description_
    :param torch.Tensor timestamps: _description_
    :return Tuple[torch.Tensor, torch.Tensor]: _description_
    """
    
    n_frames = len(timestamps)
    min_size = max(N_FRAMES_PER_CLIP, n_frames)

    if min_size % N_FRAMES_PER_STRIDE != 0:
        padding_size = min_size + N_FRAMES_PER_STRIDE - (min_size % N_FRAMES_PER_STRIDE) - n_frames
    else:
        padding_size = min_size - n_frames
    
    signal = F.pad(signal, (0, padding_size), value=0)                      # pad with 0 (silence)
    timestamps = F.pad(timestamps, (0, padding_size), value=timestamps[-1]) # pad with the last timestamp

    return signal, timestamps


def unfold_spectrogram(signal: torch.Tensor, timestamps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unfolds the spectrogram into overlapping clips.
    Applies appropriate padding before unfolidng.

    :param torch.Tensor signal: Mel spectrogram of size (n_mels, n_frames)
    :param torch.Tensor timestamps: timestamps in ms of size (n_frames)
    :return Tuple[torch.Tensor, torch.Tensor]:
        (0) Clipped spectrogram. Size: (n_clips, n_mels, n_frames)
        (1) Clipped timestamps. Size: (n_clips, n_frames)
    """
    signal, timestamps = pad_spectrogram(signal, timestamps)
    signal_clips = signal.unfold(-1, N_FRAMES_PER_CLIP, N_FRAMES_PER_STRIDE)
    timestamps_clips = timestamps.unfold(-1, N_FRAMES_PER_CLIP, N_FRAMES_PER_STRIDE)
    return signal_clips, timestamps_clips


def midi_to_pandas(uri: str, resolution: Optional[float] = None) -> pd.DataFrame:
    """Read MIDI and convert to Pandas DataFrame.

    :param str uri: Path to MIDI file
    :param Optional[float] resolution: Defaults to None. 
        If specified, apply the resolution and groups the notes by time events.
            columns=['event': float, 'onset': List[float], 'pitch': List[int]]
        If NOT specified, simply return:
            columns=['onset': float, 'pitch': int]
    :return pd.DataFrame: dataframe/csv representation of the MIDI file
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
                lambda x: round(x / resolution) * resolution)
        midi = midi.groupby('event').agg({'onset': list, 'pitch': list}).reset_index()   
        midi = midi.sort_values('event').reset_index(drop=True)

    return midi


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


def _find_smallest_event_interval_covering_timestamps(
        events: List[float], timestamps: torch.Tensor) -> List[Tuple[int,int]]:
    start, end = timestamps[0], timestamps[-1]
    return _find_range(events, start, end)


def find_random_subscore_covering_timestamps(
        midi: str | pd.DataFrame, timestamps: torch.Tensor) -> pd.DataFrame:
    """Finds a random subscore covering the timestamps.
    WARNING: This function contains an RNG. Set seed beforehand for reproducibility.

    :param str | pd.DataFrame midi:
    :param torch.Tensor timestamps:
    :return pd.DataFrame:
    """
    if isinstance(midi, str):
        midi = midi_to_pandas(midi, EVENT_RESOLUTION)

    start, end = _find_smallest_event_interval_covering_timestamps(
            midi['event'], timestamps)      # inclusive
    rand_start = randrange(0, start + 1)    # rand_start in [0, start]
    rand_end = randrange(end, len(midi))    # rand_end in [end, len(midi))

    return midi.iloc[rand_start: rand_end + 1]
    

def midi_to_matrix(midi: str | pd.DataFrame, timestamps: torch.Tensor) -> torch.Tensor:
    """TODO comment

    :param str | pd.DataFrame midi: _description_
    :param torch.Tensor timestamps: _description_
    :return torch.Tensor: _description_
    """
    if isinstance(midi, str):
        midi = midi_to_pandas(midi, EVENT_RESOLUTION)

    events = set(midi['event'])   # set of event timestamps in ms
    midi_matrix = torch.zeros((len(timestamps), len(midi))) # rows are frames, columns are events

    event_idx = 0  # 0th event is [BOS]
    for t, timestamp in enumerate(timestamps):
        if timestamp in events:
            event_idx += 1
        midi_matrix[t,event_idx] = 1
    
    return midi_matrix


def midi_to_tokens(midi: str | pd.DataFrame) -> Dict[str, torch.Tensor]:
    """TODO comment

    :param str | pd.DataFrame midi: _description_
    :return Dict[str, torch.Tensor]: _description_
    """

    if isinstance(midi, str):
        midi = midi_to_pandas(midi, EVENT_RESOLUTION)

    # 0th event is [BOS]
    event_indices: List[int] = [0]
    encoded_midi: List[int] = [TOKEN_ID['[BOS]']]

    for _, row in midi.iterrows():

        # Add the event index
        event_indices.append(len(encoded_midi))

        # Add tokens to the encoding
        encoded_midi.append(TOKEN_ID['[event]'])
        encoded_midi += row['pitch']

    encoded_midi, event_indices = torch.tensor(encoded_midi), torch.tensor(event_indices)
    n_tokens, n_events = len(encoded_midi), len(event_indices)

    # Construct the attention mask for within-event
    single_event_attn_mask = torch.zeros((n_tokens, n_tokens), dtype=int)
    for i, start in enumerate(event_indices):
        if i == n_events - 1:   # last event
            end = n_tokens
        else:
            end = event_indices[i+1]
        single_event_attn_mask[start:end,start:end] = 1

    # Construct the attention mask for between-event
    event_mask = torch.zeros((n_tokens,), dtype=int)
    event_mask[event_indices] = 1
    multi_event_attn_mask = torch.outer(event_mask, event_mask)

    return {
        "input_ids": encoded_midi,
        "event_indices": event_indices,
        "event_mask": event_mask,
        "single_event_attn_mask": single_event_attn_mask,
        "multi_event_attn_mask" : multi_event_attn_mask
    }

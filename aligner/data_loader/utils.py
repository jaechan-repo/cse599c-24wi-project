import torch
import torch.nn.functional as F
import torchaudio
from typing import List, Optional, Tuple, Dict
from ..utils.constants import *
import pretty_midi as pm
import pandas as pd
from random import randrange


_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    center=True
    # pad waveform on both sides s.t. t'th frame centered at t * hop_length
)


def load_spectrogram(uri: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Given path to audio file, computes its Mel spectrogram.

    Args:
        uri (str): Path to audio file. *.wav, *.mp3.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            (0) Mel spectrogram. Size: (n_mels, n_frames).
            (1) Frame timestamps in seconds. Size: (n_frames,).
                Example: [0.00, 0.01, 0.02, ...] for frame size of 10 ms.
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


def _pad_spectrogram(signal: torch.Tensor, 
                    timestamps: torch.Tensor
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pads the spectrogram as part of the implementation of
    :func:`~unfold_spectrogram`.
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


def unfold_spectrogram(signal: torch.Tensor,
                       timestamps: torch.Tensor
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unfolds the spectrogram into overlapping clips.

    The amount of overlapping and the number of clips are determined
    by N_FRAMES_PER_STRIDE. Pads if the number of frames in the 
    spectrogram is not divisible by N_FRAMES_PER_STRIDE.
    For fully expected behavior, first load the spectrogram with
    :func:`load_spectrogram` and apply this function to the
    loaded spectrogram and timestamps.

    Args:
        signal (torch.Tensor): Spectrogram. Size: (n_mels, n_frames)
        timestamps (torch.Tensor): Timestamps. Size: (n_frames,)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            (0) Clipped spectrogram. Size (n_clips, n_mels, n_frames)
            (1) Clipped timstamps. Size (n_clips, n_frames)
    """
    signal, timestamps = _pad_spectrogram(signal, timestamps)
    signal_clips = signal.unfold(-1, N_FRAMES_PER_CLIP, N_FRAMES_PER_STRIDE)
    timestamps_clips = timestamps.unfold(-1, N_FRAMES_PER_CLIP, N_FRAMES_PER_STRIDE)
    return signal_clips, timestamps_clips


def midi_to_pandas(uri: str,
                   resolution: Optional[float] = None
                   ) -> pd.DataFrame:
    """Reads MIDI file and converts to pandas dataframe.

    If the resolution is NOT specified (default), this function returns a
    dataframe with `columns=['onset', 'pitch']`, where 'onset' denotes
    the timestamps of note onsets measured in seconds, and 'pitch' denotes
    the MIDI pitch scale ranging from 0 to 121. For example,
    |   onset   |   pitch   |
    |   ---     |   ---     |
    |   0.5     |    81     |
    |   0.52    |    83     |
    If the resolution IS specified in seconds, this function approximates
    the onset timestamps to the nearest multiple of the resolution, and
    groups the notes by the these approximated time events. The returned
    dataframe looks like:
    |   event   |       onset   |       pitch   |
    |   ---     |       ---     |       ---     |
    |   0.5     |[0.49,0.5,0.51]|  [81, 83, 85] |
    |   0.9     |  [0.9, 0.91]  |   [72, 74]    |

    Args:
        uri (str): Path to MIDI file. *.mid, *.midi.
        resolution (Optional[float]): Event resolution in seconds.

    Returns:
        pd.DataFrame: dataframe representation of the MIDI file.
    """
    midi_pretty = pm.PrettyMIDI(uri)

    midi = []
    instrument = midi_pretty.instruments[0]   # TODO for v1: multi-instrument support
    if not instrument.is_drum:
        for note in instrument.notes:
            midi.append((note.start, int(note.pitch)))
    midi = pd.DataFrame(midi, columns=['onset', 'pitch']).sort_values('onset').reset_index(drop=True)

    if resolution is not None:
        midi['event'] = midi['onset'].apply(
                lambda x: round(x / resolution) * resolution)
        midi = midi.groupby('event').agg({'onset': list, 'pitch': list}).reset_index()   
        midi = midi.sort_values('event').reset_index(drop=True)

    return midi


def _find_range(arr: List[float], min_val: float, max_val: float):
    """Binary search"""
    
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


def _find_smallest_event_interval_covering_timestamps(events: List[float],
                                                      timestamps: torch.Tensor
                                                      ) -> List[Tuple[int,int]]:
    start, end = timestamps[0], timestamps[-1]
    return _find_range(events, start, end)


def find_random_subscore_covering_timestamps(midi: str | pd.DataFrame,
                                             timestamps: torch.Tensor
                                             )-> pd.DataFrame:
    """Extracts a random subsequence of MIDI such that it covers the timestamps.
    
    WARNING: This function contains an RNG.
    The two bounds of the random subsequence are determined by first finding the
    a subsequence that minimally covers the timestamps and then uniformly sampling
    the left and right bounds from the original MIDI outside of the minimal
    subsequence.

    Args:
        midi (str | pd.DataFrame): Path to MIDI file or return value of
                                   :func:`midi_to_pandas` with resolution specified.
        timestamps (torch.Tensor): Timestamps in seconds, as given by 
                                   :func:`load_spectrogram`.
                                   Assumed to be in SORTED (ascending) order.

    Returns:
        pd.DataFrame: Random subsequence of MIDI that covers the timestamps.
    """
    if isinstance(midi, str):
        midi = midi_to_pandas(midi, EVENT_RESOLUTION)

    start, end = _find_smallest_event_interval_covering_timestamps(
            midi['event'], timestamps)      # inclusive
    rand_start = randrange(0, start + 1)    # rand_start in [0, start]
    rand_end = randrange(end, len(midi))    # rand_end in [end, len(midi))

    return midi.iloc[rand_start: rand_end + 1]
    

def midi_to_matrix(midi: str | pd.DataFrame, timestamps: torch.Tensor) -> torch.Tensor:
    """Converts MIDI to its alignment matrix representation.

    ##### IMPORTANT #####
    NOTE: The rows of this matrix correspond to the audio frames, and the columns 
        correspond to the note events (NOT the individual tokens).
    NOTE: The zero-th column of this matrix corresponds to the [BOS] token.

    Args:
        midi (str | pd.DataFrame): Path to MIDI file or return value of
                                   :func:`midi_to_pandas` with resolution specified.
        timestamps (torch.Tensor): Timestamps in seconds, as given by 
                                   :func:`load_spectrogram`.
                                   Assumed to be in SORTED (ascending) order.

    Returns:
        torch.Tensor: Alignment matrix of size (n_frames, n_events)
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
    """Converts MIDI to various forms of encoding, including its tokenized form.

    The returned dictionary contains five fields:
    (0) "input_ids": Integer array representation (tokenization) of the MIDI.
        This is the encoding that the transformer model receives. During
        tokenization, MIDI-scaled pitches [0, 121] are directly taken as unique 
        tokens, and special tokens are attached such as [BOS] and [event]. See
        `constants.py` for their integer codes. Size: (n_tokens,).
    (2) "event_mask": Array filled with 0s and 1s, where 1s fill positions where
        event markers lie in "input_ids". Size: (n_tokens,).
    (3) "attn_mask": mask for self-attention. Size: (n_tokens, n_tokens).

    Args:
        midi (str | pd.DataFrame): Path to MIDI file or return value of
                                   :func:`midi_to_pandas` with resolution specified.

    Returns:
        Dict[str, torch.Tensor]: Dictionary of the encoded MIDI.
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

    # Merge the two attention masks
    attn_mask = (single_event_attn_mask + multi_event_attn_mask).clamp(max=1)

    return {
        "input_ids": encoded_midi,
        # "event_indices": event_indices,
        "event_mask": event_mask,
        # "single_event_attn_mask": single_event_attn_mask,
        # "multi_event_attn_mask" : multi_event_attn_mask,
        "attn_mask": attn_mask
    }

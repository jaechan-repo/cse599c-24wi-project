import torch
import torch.nn.functional as F
from torch import Tensor
import torchaudio
from typing import List, Optional, Tuple, NamedTuple
from ..utils.constants import *
import pretty_midi as pm
import pandas as pd
from random import randrange
import math


_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    center=True
    # pad waveform on both sides s.t. t'th frame centered at t * hop_length
)


def t2fi(timestamp: float) -> int:
    return timestamp // AUDIO_RESOLUTION


def sample_interval(imin: int, imax: int,
                    jmin: int, jmax: int) -> Tuple[int, int]:
    assert imin <= imax < jmin <= jmax
    i = randrange(imin, imax+1)
    j = randrange(jmin, jmax+1)
    return i, j


def load_spectrogram(uri: str) -> torch.Tensor:
    """Given path to audio file, computes a Mel spectrogram.

    Args:
        uri (str): Path to audio file. *.wav, *.mp3.

    Returns:
        Mel spectrogram. Size: (n_frames, N_MELS).
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

    return signal.transpose(0, 1)


def _pad_spectrogram(signal: torch.Tensor) -> torch.Tensor:
    n_frames = len(signal)
    min_size = max(N_FRAMES_PER_CLIP, n_frames)

    if min_size % N_FRAMES_PER_STRIDE != 0:
        padding_size = min_size + N_FRAMES_PER_STRIDE - (min_size % N_FRAMES_PER_STRIDE) - n_frames
    else:
        padding_size = min_size - n_frames

    signal = F.pad(signal, (0, 0, 0, padding_size), value=0)

    assert signal.shape[-1] == N_MELS
    return signal


def unfold_spectrogram(signal: torch.Tensor) -> torch.Tensor:
    """Unfolds the spectrogram into overlapping clips.

    The number of frames allocated to each clip is given by
    the constant `N_FRAMES_PER_CLIP`. The number of overlapping
    frames between two neighboring clips is given by
    `N_FRAMES_PER_CLIP - N_FRAMES_PER_STRIDE`. The function pads
    to the multiple of `N_FRAMES_PER_STRIDE` if the number of frames
    in the spectrogram is not divisible by `N_FRAMES_PER_STRIDE`.

    For fully expected behavior, first load the spectrogram with
    :func:`load_spectrogram` and apply this function to the
    loaded spectrogram.

    Args:
        signal (torch.Tensor): Spectrogram. 
                               Size: (n_frames, N_MELS)

    Returns:
        Clipped spectrogram. Size (n_clips, N_FRAMES_PER_CLIP, N_MELS)
    """
    signal = _pad_spectrogram(signal)
    print(signal.shape)
    signal_clips = signal.unfold(dimension=0,
                                 size=N_FRAMES_PER_CLIP,
                                 step=N_FRAMES_PER_STRIDE)
    signal_clips = signal_clips.permute((0, 2, 1))
    
    assert signal_clips.shape[1] == N_FRAMES_PER_CLIP
    assert signal_clips.shape[-1] == N_MELS
    return signal_clips


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


class ParsedMIDI:

    def __init__(self,
                 midi: str | pd.DataFrame,
                 n_frames: Optional[int] = None):

        if isinstance(midi, str):
            self.midi = midi_to_pandas(midi, EVENT_RESOLUTION)
        else:
            self.midi = midi

        assert len(self.midi) != 0, "MIDI cannot be empty."

        self.events: List[float] = self.midi['event'].tolist()
        self.events.insert(0, 0)    # BOS token
        n_events = len(self.events)

        ##########################################
        ### Creating the full alignment matrix ###
        ##########################################

        if n_frames is None:
            # index of last frame plus one
            n_frames = t2fi(self.events[-1]) + 1

        assert n_frames > t2fi(self.events[-1]), \
            "n_frames must exceed the span of frames inferred from the MIDI"

        self.fi2ei = {}     # frame index to event index
        self.alignment_matrix = torch.zeros((n_frames, n_events), dtype=int)
        ei = 1 if self.events[1] == 0 else 0

        for fi in range(n_frames):
            timestamp = fi * AUDIO_RESOLUTION

            if ei == n_events - 1:
                # Last event reached.
                # Map all the remaining frames to this event.
                for fi_ in range(fi, n_frames):
                    self.fi2ei[fi_] = ei
                    self.alignment_matrix[fi_, ei] = 1
                break

            if self.events[ei + 1] == timestamp:
                ei += 1

            self.fi2ei[fi] = ei
            self.alignment_matrix[fi, ei] = 1

        ##############################################
        ### Creating the full encoding of the MIDI ###
        ##############################################

        # Encoding
        input_ids = [TOKEN_ID['[BOS]']]     # BOS
        event_idxes = [0]                   # BOS

        for _, row in self.midi.iterrows():
            event_idxes.append(len(input_ids))
            input_ids.append(TOKEN_ID['[event]'])
            input_ids += row['pitch']

        n_tokens = len(input_ids)
        self.input_ids = torch.tensor(input_ids, dtype=int)
        self.event_idxes = torch.tensor(event_idxes, dtype=int)
        assert self.event_idxes.shape == (n_events,)
        
        # Local event attention mask
        local_event_attn_mask = torch.zeros((n_tokens, n_tokens), dtype=int)
        for i, start in enumerate(self.event_idxes):
            end = n_tokens if i == n_events - 1 else self.event_idxes[i+1]
            local_event_attn_mask[start:end, start:end] = 1
        
        # Global event attention mask
        event_mask = torch.zeros((n_tokens,), dtype=int)
        event_mask[self.event_idxes] = 1
        global_event_attn_mask = torch.outer(event_mask, event_mask)

        # Aggregate the two attention masks
        self.attn_mask = (local_event_attn_mask + global_event_attn_mask).clamp(max=1)
        assert self.attn_mask.shape == (n_tokens, n_tokens)

        # Projection from tokens to events
        self.proj_to_evt = F.one_hot(self.event_idxes, num_classes=n_tokens)
        assert self.proj_to_evt.shape == (n_events, n_tokens)
        
        # Size variables
        self.n_frames, self.n_events, self.n_tokens = n_frames, n_events, n_tokens


    class Encoding(NamedTuple):
        input_ids: Tensor
        attn_mask: Tensor
        proj_to_evt: Tensor


    def align(self,
              afi1: int, afi2: int,
              ei1: int, ei2: int
              ) -> torch.Tensor:
        """Outputs the gold alignment matrix that corresponds to
        audio frames in the range [afi1:afi2) and event indices
        in the range [ei1:ei2).

        Args:
            afi1 (int): initial audio frame index
            afi2 (int): final audio frame index
            ei1 (int): initial event index
            ei2 (int): final event index

        Returns:
            torch.Tensor: Alignment matrix
                          Size: (afi2-afi1, ei2-ei1)
        """
        assert afi1 < afi2, "afi1 has to be less than afi2."
        assert ei1 < ei2, "ei1 has to be less ei2."
        assert afi2 <= self.n_frames, \
            "Audio frame indices cannot exceed the total number of frames."

        aei1 = self.fi2ei[afi1]
        aei2 = self.fi2ei[afi2] if afi2 < self.n_frames else self.n_events
        assert ei1 <= aei1 < aei2 <= ei2, "Audio has to occur within the events."

        Y = self.alignment_matrix[afi1:afi2, ei1:ei2]
        return Y


    def encode(self,
               ei1: int, ei2: int,
               return_tuple: bool = True
               ) -> Encoding | Tensor:
        
        assert ei1 < ei2, "ei1 has to be less than ei2."
        assert 0 <= ei1 < ei2 <= self.n_events, "Invalid event indicies."

        toki1 = self.event_idxes[ei1]
        toki2 = self.event_idxes[ei2] if ei2 < self.n_events else self.n_events

        if return_tuple:
            return ParsedMIDI.Encoding(
                input_ids=self.input_ids[toki1:toki2],
                attn_mask=self.attn_mask[toki1:toki2, toki1:toki2],
                proj_to_evt=self.proj_to_evt[ei1:ei2, toki1:toki2]
            )

        return self.input_ids[toki1:toki2]

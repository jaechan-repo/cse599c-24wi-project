import torch
import torch.nn.functional as F
from torch import Tensor, LongTensor, BoolTensor
import torchaudio
from typing import List, Optional, Tuple, NamedTuple
from ..utils.constants import *
import pretty_midi as pm
import pandas as pd
from random import randrange
from bisect import bisect_left
from memory_profiler import profile


_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    f_min=F_MIN,
    f_max=F_MAX,
    center=True # pad waveform on both sides
)


def t2fi(timestamp: float) -> int:
    return timestamp // AUDIO_RESOLUTION


def sample_interval(imin: int, imax: int,
                    jmin: int, jmax: int, 
                    ) -> Tuple[int, int]:
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

    signal = torch.mean(signal, dim=0)
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
        midi['event'] = midi['onset'].apply(lambda x: round(x / resolution))
        midi = midi.groupby('event').agg({'onset': list, 'pitch': list}).reset_index()   
        midi = midi.sort_values('event').reset_index(drop=True)

    return midi


class ParsedMIDI:

    def __init__(self,
                 midi: str | pd.DataFrame):

        if isinstance(midi, str):
            self.midi = midi_to_pandas(midi, EVENT_RESOLUTION)
        else:
            self.midi = midi

        assert len(self.midi) != 0, "MIDI cannot be empty."

        self._ei2fi: List[int] = self.midi['event'].tolist()
        self._ei2fi.insert(0, 0)    # BOS token
        self.n_events = len(self._ei2fi)

        ##########################################
        ### Creating the full alignment matrix ###
        ##########################################
        self.last_fi = self._ei2fi[-1]
        self._fi2ei = {}

        # This matrix takes up a LARGE amount of memory!
        self.alignment_matrix: BoolTensor = torch.zeros((self.last_fi + 1, self.n_events), dtype=torch.bool)

        clip_fis = [afi for ci in range(1, self.last_fi // N_FRAMES_PER_STRIDE) \
                        for afi in (ci * N_FRAMES_PER_STRIDE - 1, ci * N_FRAMES_PER_STRIDE)]
        ci = 0
        for ei in range(self.n_events):
            self._fi2ei[self._ei2fi[ei]] = ei

            if ei < self.n_events - 1:
                self.alignment_matrix[self._ei2fi[ei]:self._ei2fi[ei+1], ei] = True
            else:
                # Last event. Only one timestamp exists.
                self.alignment_matrix[self._ei2fi[ei], ei] = True

            while ci < len(clip_fis) and (afi := clip_fis[ci]) < self._ei2fi[ei]:
                # The current clip frame index falls behind the current event index,
                # in which case we map the clip frame index to the previous event index
                self._fi2ei[afi] = ei - 1
                ci += 1

        ##############################################
        ### Creating the full encoding of the MIDI ###
        ##############################################

        # Encoding
        self.score_ids = [TOKEN_ID['[BOS]']]    # BOS
        self.event_pos = [0]                    # BOS

        for _, row in self.midi.iterrows():
            self.event_pos.append(len(self.score_ids))
            self.score_ids.append(TOKEN_ID['[event]'])
            self.score_ids += row['pitch']

        self.n_tokens = len(self.score_ids)
        self.score_ids: LongTensor = torch.tensor(self.score_ids, dtype=torch.long)
        self.event_pos: LongTensor = torch.tensor(self.event_pos, dtype=torch.long)
        assert self.event_pos.shape == (self.n_events,)
        
        # Local self-attention mask
        local_event_score_attn_mask = torch.zeros((self.n_tokens, self.n_tokens),
                                                  dtype=torch.long)
        for ei, toki1 in enumerate(self.event_pos):
            toki2 = self.n_tokens if ei == self.n_events - 1 else self.event_pos[ei+1]
            local_event_score_attn_mask[toki1:toki2, toki1:toki2] = 1

        # Global self-attention mask
        event_mask = torch.zeros((self.n_tokens,), dtype=torch.long)
        event_mask[self.event_pos] = 1
        global_event_score_attn_mask = torch.outer(event_mask, event_mask)

        # Aggregate the two attention masks
        self.score_attn_mask: BoolTensor = \
            (local_event_score_attn_mask + global_event_score_attn_mask).clamp(max=1).bool()
        assert self.score_attn_mask.shape == (self.n_tokens, self.n_tokens)

        # Project score to event
        self.score_to_event: BoolTensor = \
            F.one_hot(self.event_pos, num_classes=self.n_tokens).bool()
        assert self.score_to_event.shape == (self.n_events, self.n_tokens)


    def _fi2ei_naive(self, fi: int) -> int:
        ei = bisect_left(self._ei2fi, fi)
        if ei:
            if ei < self.n_events and self._ei2fi[ei] == fi:
                return ei
            else:
                return ei - 1
        # impossible to reach
        return None


    def fi2ei(self, fi: int) -> int:
        """Convert audio frame index to event index
        """
        if fi in self._fi2ei:
            return self._fi2ei[fi]
        if fi > self.last_fi:
            return self._fi2ei[self.last_fi]
        assert fi >= 0, "Invalid frame index."
        ei = self._fi2ei_naive(fi)
        self._fi2ei[fi] = ei
        return ei


    def ei2toki(self, ei: int) -> int:
        """Convert event index to token index
        """
        assert 0 <= ei <= self.n_events, "Invalid event index."
        toki = self.event_pos[ei] if ei < self.n_events else self.n_tokens
        return toki
    

    def find_minimal_event_interval_covering(self, fi1: int, fi2: int
                                             ) -> Tuple[int, int]:
        assert fi1 < fi2, "fi1 has to be less than fi2."
        return self.fi2ei(fi1), self.fi2ei(fi2-1)+1


    def find_maximal_event_interval_covering(self,
                                             ei1: int, ei2: int,
                                             max_n_tokens: int
                                             ) -> Tuple[int, int]:
        assert ei1 < ei2, "ei1 must be less than ei2."
        assert self.ei2toki(ei2) - self.ei2toki(ei1) <= max_n_tokens, \
                "Provided [ei1, ei2) already covers max_n_tokens."

        min_ei1 = ei1
        max_ei2 = ei2

        while min_ei1 > 0 and max_ei2 < self.n_events \
                and self.ei2toki(max_ei2 + 1) - self.ei2toki(min_ei1 - 1) <= max_n_tokens:
            min_ei1 -= 1
            max_ei2 += 1
        
        while min_ei1 > 0 \
                and self.ei2toki(max_ei2) - self.ei2toki(min_ei1 - 1) <= max_n_tokens:
            min_ei1 -= 1

        while max_ei2 < self.n_events \
                and self.ei2toki(max_ei2 + 1) - self.ei2toki(min_ei1) <= max_n_tokens:
            max_ei2 += 1

        assert 0 <= min_ei1 <= ei1 < ei2 <= max_ei2 <= self.n_events
        return min_ei1, max_ei2


    def align(self,
              afi1: int, afi2: int,
              ei1: int, ei2: int
              ) -> LongTensor:
        assert ei1 < ei2, "ei1 must be less than ei2."

        aei1, aei2 = self.find_minimal_event_interval_covering(afi1, afi2)
        assert ei1 <= aei1 < aei2 <= ei2, "Audio must occur within the events." # TODO: v1

        if (pad_size := afi2 - len(self.alignment_matrix)) > 0:
            last_frame = self.alignment_matrix[-1]
            padding = last_frame.repeat(pad_size, 1)
            self.alignment_matrix = torch.cat([self.alignment_matrix, padding])

        Y = self.alignment_matrix[afi1:afi2, ei1:ei2]
        return Y
    

    def _align_naive(self,
                     afi1: int, afi2: int,
                     ei1: int, ei2: int
                     ) -> LongTensor:
        """Warning: Quadratic in time. NOT used and NOT tested.
        """
        assert ei1 < ei2, "ei1 must be less than ei2."
        assert afi1 < afi2, "afi1 must be less than afi2."

        aei1, aei2 = self.find_minimal_event_interval_covering(afi1, afi2)
        assert ei1 <= aei1 < aei2 <= ei2, "Audio must occur within the events." # TODO: v1

        Y = torch.zeros((afi2-afi1, ei2-ei1), dtype=torch.bool)

        for ei in range(ei1, ei2):
            fi1_ei = max(self._ei2fi[ei], afi1)
            fi2_ei = min(self._ei2fi[ei], afi2)
            Y[fi1_ei:fi2_ei, ei] = True

        return Y


    class Encoding(NamedTuple):
        score_ids: LongTensor
        score_attn_mask: BoolTensor
        score_to_event: BoolTensor
        event_pos: LongTensor


    def encode(self,
               ei1: int, ei2: int,
               return_tuple: bool = True
               ) -> Encoding | LongTensor:
        assert ei1 < ei2, "ei1 has to be less than ei2."

        toki1, toki2 = self.ei2toki(ei1), self.ei2toki(ei2)

        if return_tuple:
            return ParsedMIDI.Encoding(
                score_ids=self.score_ids[toki1:toki2],
                score_attn_mask=self.score_attn_mask[toki1:toki2, toki1:toki2],
                score_to_event=self.score_to_event[ei1:ei2, toki1:toki2],
                event_pos=self.event_pos[ei1:ei2])

        return self.score_ids[toki1:toki2]

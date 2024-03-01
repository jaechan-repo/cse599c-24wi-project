import torch
import torch.nn.functional as F
from torch import LongTensor, BoolTensor
from typing import List, Optional, Tuple, NamedTuple
from ..utils.constants import *
import pretty_midi as pm
import pandas as pd
from bisect import bisect_left
from memory_profiler import profile
import sys
from random import randrange, uniform


def sample_interval(imin: int, imax: int,
                    jmin: int, jmax: int,
                    randomly_ignore_bounds: bool = False
                    ) -> Tuple[int, int]:
    assert imin <= imax < jmin <= jmax

    if not randomly_ignore_bounds:
        i = randrange(imin, imax+1)
        j = randrange(jmin, jmax+1)
        return i, j
    
    p = uniform(0, 1)
    if p < 0.7:
        # 70% of the times, sample an interval that adheres to the bounds.
        return sample_interval(imin, imax, jmin, jmax, False)

    if p < 0.9 and imax + 1 < jmin:
        # 20% of the times, sample an interval that partially ignores the bounds.
        midpoint = randrange(imax+1, jmin)
        if p < 0.8:
            i = randrange(imin, imax+1)
            return i, midpoint
        else:
            j = randrange(jmin, jmax+1)
            return midpoint, j

    # 10% of the times, ignore the bounds completely.
    if p < 0.95:
        # sample left first, then right
        i = randrange(imin, jmax)
        j = randrange(i+1, jmax+1)
        return i, j
    # sample right first, then left
    j = randrange(imin+1, jmax+1)
    i = randrange(imin, j)
    return i, j


def midi_to_pandas(uri: str,
                   event_resolution: float | None = None,
                   audio_resolution: float | None = None
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
        event_resolution (float | None): Event resolution in seconds.
        audio_resolution (float | None): Audio resolution in seconds.
                                         Defaults to event resolution, if specified.

    Returns:
        pd.DataFrame: dataframe representation of the MIDI file.
    """
    if audio_resolution is None:
        audio_resolution = event_resolution

    assert audio_resolution / event_resolution \
            == int(audio_resolution / event_resolution), \
            "audio_resolution must be an integer multiple of event_resolution"

    midi_pretty = pm.PrettyMIDI(uri)

    # TODO for v1: multi-instrument support
    instrument = midi_pretty.instruments[0]
    midi = []
    for note in instrument.notes:
        midi.append((note.start, int(note.pitch)))
    midi = pd.DataFrame(midi, columns=['onset', 'pitch']
                        ).sort_values('onset').reset_index(drop=True)

    if event_resolution is not None:
        midi['event'] = midi['onset'].apply(lambda x: round(x / event_resolution))
        midi['frame'] = midi['onset'].apply(lambda x: round(x / audio_resolution))
        midi = midi.groupby('event').agg({'onset': list,
                                          'pitch': list,
                                          'frame': 'min'}).reset_index()   
        midi = midi.sort_values('event').reset_index(drop=True)

    return midi


class ParsedMIDI:
    """
    Vocabulary
        ei: Event index. ith event corresponds to ith row in the pandas
            representation of the MIDI. Each event is determined up to
            EVENT_RESOLUTION.
        fi: Frame index. Multiply by AUDIO_RESOLUTION to obtain the timestamp
            in seconds. Synonym: afi (audio frame index).
        toki: Token index in `score_ids`.
    """

    # @profile
    def __init__(self,
                 midi: str | pd.DataFrame,
                 lazy_align: bool = True):
        """Parses the midi.

        Args:
            midi (str | pd.DataFrame): either path to MIDI or the pandas
                                       representation of the MIDI.
            lazy_align (bool, optional): When set to False, the entire
                                         (n_frames, n_events) alignment matrix is
                                         constructed during `__init__`. This makes
                                         the `self.align` operation O(1). When set to
                                         True (default), the alignment matrix is not
                                         computed until queried by the `self.align`
                                         operation. Setting to True is highly recommended
                                         for both training and batched evaluation
                                         as the full alignment matrix tends to cost
                                         lots of RAM memory. But if the call to `__init__`
                                         is sparse (e.g., real-world application where
                                         you have a fixed score and multiple `self.align`
                                         queries), then setting to False is recommended.
        """
        if isinstance(midi, str):
            self.midi = midi_to_pandas(midi, EVENT_RESOLUTION, AUDIO_RESOLUTION)
        else:
            self.midi = midi

        assert len(self.midi) != 0, "MIDI cannot be empty."

        self._ei2fi: List[int] = self.midi['frame'].tolist()
        self._ei2fi.insert(0, 0)    # BOS token
        self.n_events = len(self._ei2fi)

        #######################################################
        ### Creating the frame index to event index mapping ###
        #######################################################
        self.last_fi = self._ei2fi[-1]
        self._fi2ei = {}
        self.lazy_align = lazy_align

        if not self.lazy_align:
            self.alignment_matrix: BoolTensor = torch.zeros(
                    (self.last_fi + 1, self.n_events), dtype=torch.bool)

        # We only compute the event indices for the following frame indices.
        # Why? These are all we need for training.
        clip_fis = [afi for ci in range(1, self.last_fi // N_FRAMES_PER_STRIDE)
                        for afi in (ci * N_FRAMES_PER_STRIDE - 1, ci * N_FRAMES_PER_STRIDE)]
        ci = 0
        for ei in range(self.n_events):
            self._fi2ei[self._ei2fi[ei]] = ei

            if not self.lazy_align:
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
            self.score_ids += row['pitch']      # list of pitch values

        self.n_tokens = len(self.score_ids)
        self.score_ids = torch.tensor(self.score_ids, dtype=torch.long)
        self.event_pos = torch.tensor(self.event_pos, dtype=torch.long)
        assert self.event_pos.shape == (self.n_events,)

        # LOCAL self-attention mask
        self.local_event_attn_mask = torch.zeros((self.n_tokens, self.n_tokens),
                                                 dtype=torch.bool)
        for ei, toki1 in enumerate(self.event_pos):
            # Get the next token index
            toki2 = self.n_tokens if ei == self.n_events - 1 else self.event_pos[ei+1]
            # Def. of local event attention          
            self.local_event_attn_mask[toki1:toki2, toki1:toki2] = True

        # GLOBAL self-attention mask
        event_mask = torch.zeros((self.n_tokens,), dtype=torch.bool)
        event_mask[self.event_pos] = True
        self.global_event_attn_mask = torch.outer(event_mask, event_mask)
        self.global_event_attn_mask[
            torch.arange(self.n_tokens), torch.arange(self.n_tokens)
        ] = True    # diagonal entires

        # Projection matrix from score to events
        self.score_to_event = F.one_hot(
                self.event_pos, num_classes=self.n_tokens).bool()
        assert self.score_to_event.shape == (self.n_events, self.n_tokens)


    def get_frame_indices(self) -> List[int]:
        """Outputs the frame indices of the score events.

        Returns:
            List[int]: Audio frame indices. Size: (n_events, )
        """
        return self._ei2fi.copy()


    def _fi2ei_naive(self, fi: int) -> int:
        # O(log self.n_events)
        ei = bisect_left(self._ei2fi, fi)
        if ei:
            if ei < self.n_events and self._ei2fi[ei] == fi:
                return ei
            else:
                return ei - 1
        # impossible to reach
        return None


    def fi2ei(self, fi: int) -> int:
        """Converts the given audio frame index to its closest preceding
        event index.
        """
        if fi in self._fi2ei:
            return self._fi2ei[fi]
        if fi > self.last_fi:
            return self._fi2ei[self.last_fi]
        assert fi >= 0, "Invalid frame index."
        ei = self._fi2ei_naive(fi)  # Naive O(log N) binary search
        self._fi2ei[fi] = ei        # Cache for further usage
        return ei


    def ei2toki(self, ei: int) -> int:
        """Converts the given event index to its corresponding token index,
        i.e. its location in `self.score_ids`.
        """
        assert 0 <= ei <= self.n_events, "Invalid event index."
        toki = self.event_pos[ei] if ei < self.n_events else self.n_tokens
        return toki
    

    def find_minimal_event_interval_covering(self, fi1: int, fi2: int
                                             ) -> Tuple[int, int]:
        """Finds an event interval that minimally covers the given audio frame interval.
        Both the input and output intervals are inclusive on the left and exclusive on
        the right. In other words, the returned right-hand side bound of the event
        interval might be the number of events (`self.n_events`), instead of a valid 
        event index.
        """
        assert fi1 < fi2, "fi1 has to be less than fi2."
        return self.fi2ei(fi1), self.fi2ei(fi2-1)+1


    def find_maximal_event_interval_covering(self,
                                             fi1: int, fi2: int,
                                             max_n_tokens: int
                                             ) -> Tuple[int, int]:
        assert fi1 < fi2, "fi1 has to be less than fi2."
        ei1, ei2 = self.find_minimal_event_interval_covering(fi1, fi2)
        min_ei1, max_ei2 = ei1, ei2

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

        assert 0 <= min_ei1 <= ei1 < ei2 <= max_ei2 <= self.n_events, "Error!"
        return min_ei1, max_ei2
    

    def sample_event_interval(self,
                              afi1: int, afi2: int,
                              random_sample: bool = False,
                              include_null: bool = False
                              ) -> Tuple[int, int]:
        assert random_sample or not include_null, \
            "You can't set include_null to True without also setting random_sample to True."

        # Event interval corresponding to [afi1, afi2).
        # This is the most MINIMAL interval of events that covers the span of
        # audio from afi1 to afi2.
        ei1, ei2 = self.find_minimal_event_interval_covering(afi1, afi2)
        n_tokens = self.ei2toki(ei2) - self.ei2toki(ei1)
        if n_tokens > MAX_N_TOKENS:
            return ei1, ei2

        # [min_ei1, max_ei2) is the MAXIMAL event interval
        # such that the interval length doesn't exceed MAX_N_TOKENS.
        min_ei1, max_ei2 = self.find_maximal_event_interval_covering(afi1, afi2, MAX_N_TOKENS)
        assert 1 <= self.ei2toki(max_ei2) - self.ei2toki(min_ei1) <= MAX_N_TOKENS
        if random_sample:
            # Randomly sample an event interval
            ei1, ei2 = sample_interval(min_ei1, ei1, ei2, max_ei2,
                                       randomly_ignore_bounds=include_null)
        else:
            # Otherwise, take the maximal interval.
            ei1, ei2 = min_ei1, max_ei2

        return ei1, ei2


    def encode(self,
               ei1: int, ei2: int,
               include_null: bool = False
               ) -> Tuple | LongTensor:
        assert ei1 < ei2, "ei1 has to be less than ei2."

        class Encoding(NamedTuple):
            score_ids: LongTensor
            local_event_attn_mask: BoolTensor
            global_event_attn_mask: BoolTensor
            score_to_event: BoolTensor
            event_pos: LongTensor

        toki1, toki2 = self.ei2toki(ei1), self.ei2toki(ei2)

        if not include_null:
            return Encoding(
                score_ids=self.score_ids[toki1:toki2],
                local_event_attn_mask=self.local_event_attn_mask[toki1:toki2, toki1:toki2],
                global_event_attn_mask=self.global_event_attn_mask[toki1:toki2, toki1:toki2],
                score_to_event=self.score_to_event[ei1:ei2, toki1:toki2],
                event_pos=self.event_pos[ei1:ei2])

        raise NotImplementedError

        # Add NULL token to the end
        score_ids = self.score_ids[toki1: toki2]
        score_ids = F.pad(score_ids, (0, 1), value=TOKEN_ID['[NULL]'])

        # TODO
        score_attn_mask = self.score_attn_mask[toki1:toki2, toki1:toki2]
        score_attn_mask = F.pad(score_attn_mask, (0, 1, 0, 1), value=True)

        score_to_event = self.score_to_event[ei1:ei2, toki1:toki2]
        score_to_event = F.pad(score_to_event, (0, 1, 0, 1), value=False)
        score_to_event[ei2, toki2] = True   # NULL event

        event_pos = self.event_pos[ei1:ei2]
        event_pos = F.pad(event_pos, (0, 1), value=toki2)

        return Encoding(
            score_ids, score_attn_mask, score_to_event, event_pos
        )


    # @profile
    def align(self,
              afi1: int, afi2: int,
              ei1: int, ei2: int,
              include_null: bool = False
              ) -> BoolTensor:
        """Outputs the alignment matrix that corresponds to the given bounding box
        of audio frame interval and event interval. The input intervals are expected
        to be inclusive on the left and exclusive on the right.
        Runs in O(1) if `self.lazy_align` set to False, O(self.n_events) otherwise.
        """
        assert ei1 < ei2, "ei1 must be less than ei2."

        aei1, aei2 = self.find_minimal_event_interval_covering(afi1, afi2)

        if not include_null:
            assert ei1 <= aei1 < aei2 <= ei2, \
                    "audio interval should be contained within event interval."
            Y = self._align_eager(afi1, afi2, ei1, ei2) \
                if not self.lazy_align \
                else self._align_lazy(afi1, afi2, ei1, ei2)
            return Y

        fi_ei1, fi_ei2 = self._ei2fi[ei1], self._ei2fi[ei2]
        Y = self._align_eager(max(fi_ei1, afi1), min(fi_ei2, afi2), ei1, ei2) \
            if not self.lazy_align \
            else self._align_lazy(max(fi_ei1, afi1), min(fi_ei2, afi2), ei1, ei2)        

        left_pad, right_pad = max(fi_ei1 - afi1, 0), max(afi2 - fi_ei2, 0)

        # add the NULL event marker along the event axis
        # add the padding along the audio axis if the score doesn't cover the audio
        Y = F.pad(Y, (0, 1, left_pad, right_pad), value=False)

        # Map the overflowing audio frames to the NULL event
        Y[:left_pad, ei2] = True
        Y[-right_pad:, ei2] = True
        return Y


    def _align_eager(self,
                     afi1: int, afi2: int,
                     ei1: int, ei2: int
                     ) -> BoolTensor:
        if (pad_size := afi2 - len(self.alignment_matrix)) > 0:
            last_frame = self.alignment_matrix[-1]
            padding = last_frame.repeat(pad_size, 1)
            self.alignment_matrix = torch.cat([self.alignment_matrix, padding])

        Y = self.alignment_matrix[afi1:afi2, ei1:ei2]
        return Y
    

    def _align_lazy(self,
                    afi1: int, afi2: int,
                    ei1: int, ei2: int
                    ) -> BoolTensor:
        Y: BoolTensor = torch.zeros((afi2-afi1, ei2-ei1), dtype=torch.bool)
        ei_afi1, ei_afi2 = self.find_minimal_event_interval_covering(afi1, afi2)

        for ei in range(max(ei_afi1, ei1), min(ei_afi2, ei2)):
            # fi_ei1 is the frame that corresponds to ei (curr event).
            # fi_ei2 is the frame that corresponds to ei+1 (next event).
            # Our goal is to map all frames inbetween to ei (curr event).
            fi1_ei = self._ei2fi[ei]
            fi2_ei = self._ei2fi[ei+1] if ei+1 < self.n_events else sys.maxsize

            # Shift according to the given bounding box
            fi1_ei_rel, fi2_ei_rel = \
                    max(fi1_ei, afi1) - afi1, min(fi2_ei, afi2) - afi1
            ei_rel = ei - ei1

            assert fi1_ei_rel < fi2_ei_rel
            Y[fi1_ei_rel: fi2_ei_rel, ei_rel] = True

        return Y


    def perturb(self):
        ei2fi = self.get_frame_indices()
        



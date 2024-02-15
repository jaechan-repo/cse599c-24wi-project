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
                 midi: str | pd.DataFrame,
                 lazy_align: bool = True):

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
        self.lazy_align = lazy_align

        if not self.lazy_align:
            # This matrix takes up a LARGE amount of memory!
            self.alignment_matrix: BoolTensor = torch.zeros((self.last_fi + 1, self.n_events), dtype=torch.bool)

        clip_fis = [afi for ci in range(1, self.last_fi // N_FRAMES_PER_STRIDE) \
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
        """O(log n_events).
        """
        ei = bisect_left(self._ei2fi, fi)
        if ei:
            if ei < self.n_events and self._ei2fi[ei] == fi:
                return ei
            else:
                return ei - 1
        # impossible to reach
        return None


    def fi2ei(self, fi: int) -> int:
        """Converts audio frame index to closest preceding event index.
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
        """Converts event index to token index in self.score_ids.
        """
        assert 0 <= ei <= self.n_events, "Invalid event index."
        toki = self.event_pos[ei] if ei < self.n_events else self.n_tokens
        return toki
    

    def find_minimal_event_interval_covering(self, fi1: int, fi2: int
                                             ) -> Tuple[int, int]:
        """Finds an event interval that minimally covers the given audio frame interval.
        Both the input and output intervals are inclusive on the left and exclusive on
        the right. In other words, the returned right-hand side bound of the event
        interval might be the number of events (`self.n_events`), not a valid event index.
        """
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

    def align(self,
              afi1: int, afi2: int,
              ei1: int, ei2: int
              ) -> BoolTensor:
        """Outputs the alignment matrix that corresponds to the given bounding box
        of audio frame interval and event interval. The input intervals are expected
        to be inclusive on the left and exclusive on the right.
        Runs in O(1) if `self.lazy_align` set to False, O(self.n_events) otherwise.
        """
        assert ei1 < ei2, "ei1 must be less than ei2."

        aei1, aei2 = self.find_minimal_event_interval_covering(afi1, afi2)
        assert ei1 <= aei1 < aei2 <= ei2, "Audio must occur within the events." # TODO: v1

        return self._align_eager(afi1, afi2, ei1, ei2) if not self.lazy_align \
                else self._align_lazy(afi1, afi2, ei1, ei2)


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

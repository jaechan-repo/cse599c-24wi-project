from itertools import cycle
from pathlib import Path
from collections.abc import Sequence
from typing import Literal, List, Iterator, NamedTuple, Tuple, Optional, Any

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from torch import Tensor, LongTensor, BoolTensor

import pandas as pd
from memory_profiler import profile
import os

from ..utils.constants import *
from .score_tools import ParsedMIDI
from .audio_tools import load_spectrogram, unfold_spectrogram, sample_interval

import numpy as np


class AlignDataset(IterableDataset):

    class Item(NamedTuple):
        id_: Tuple[int, int]
        audio_frames: Tensor
        score_ids: LongTensor
        score_attn_mask: BoolTensor
        score_to_event: BoolTensor
        event_pos: LongTensor
        Y: BoolTensor


    class ItemWithMetadata(Item):
        audio: Tensor
        events: List[int]
        audio_resolution: int
        event_resolution: int
        audio_frame_interval: Tuple[int, int]
        event_interval: Tuple[int, int]


    class Batch(NamedTuple):
        id_: List[Tuple[int, int]]
        audio_frames: Tensor                # (B, N_FRAMES_PER_CLIP, N_MELS)
        score_ids: LongTensor               # (B, max_n_tokens)
        score_attn_mask: BoolTensor         # (B, max_n_tokens, max_n_tokens)
        score_to_event: BoolTensor          # (B, max_n_events, max_n_tokens)
        event_pos: List[LongTensor]         # (B) * (n_events,)
        event_padding_mask: BoolTensor      # (B, max_n_events)
        Y: BoolTensor                       # (B, N_FRAMES_PER_CLIP, max_n_events)


    def __init__(self,
                 ids: Sequence,
                 midi_uris: Sequence[str],
                 audio_uris: Sequence[str],
                 split: Literal['train', 'validation', 'test'],
                 shuffle: bool = True
                 ):
        self.split = split
        self.shuffle = shuffle

        assert len(ids) == len(midi_uris) == len(audio_uris)
        self.ids, self.midi_uris, self.audio_uris = \
                np.array(ids), np.array(midi_uris), np.array(audio_uris)
        self.n_uris = len(self.ids)

        if split == 'train' and shuffle:
            self._randomize()

        # Current variables
        self.idx = -1
        self.clip_idx = -1


    def _randomize(self):
        # Sample the permutation
        perm = np.random.permutation(self.n_uris)
        self.ids = self.ids[perm]
        self.midi_uris = self.midi_uris[perm]
        self.audio_uris = self.audio_uris[perm]


    def _update(self):
        if self.clip_idx == -1 or self.clip_idx == len(self.audio_clips) - 1:

            if self.idx == self.n_uris - 1:
                if self.split != 'train':
                    raise StopIteration

                self.idx = self.clip_idx = 0
                if self.shuffle:
                    self._randomize()

            elif self.idx == -1 and self.clip_idx == -1:
                # Starting
                self.idx = 0
                self.clip_idx = 0

            else:
                # Move to next idx and reset clip_idx
                self.idx += 1
                self.clip_idx = 0

            self.id_ = self.ids[self.idx]
            self.midi: ParsedMIDI = ParsedMIDI(self.midi_uris[self.idx], lazy_align=True)
            self.audio: Tensor = load_spectrogram(self.audio_uris[self.idx])
            self.audio_clips: List[Tensor] = unfold_spectrogram(self.audio)
            if self.split == 'train':
                clip_perm = torch.randperm(len(self.audio_clips))
                self.audio_clips = self.audio_clips[clip_perm]
        else:
            self.clip_idx += 1


    def __next__(self) -> Item:
        self._update()

        # Audio frame interval.
        # Inclusive on the left, exclusive on the right.
        afi1 = self.clip_idx * N_FRAMES_PER_STRIDE
        afi2 = afi1 + N_FRAMES_PER_CLIP

        # Event interval corresponding to [afi1, afi2). This is the most MINIMAL 
        # interval of events that covers the span of music from afi1 and afi2.
        ei1, ei2 = self.midi.find_minimal_event_interval_covering(afi1, afi2)
        n_tokens = self.midi.ei2toki(ei2) - self.midi.ei2toki(ei1)

        # Search for a larger event interval if this minimal event interval
        # doesn't exceed MAX_N_TOKENS.
        if n_tokens <= MAX_N_TOKENS:
            # [min_ei1, max_ei2) is the MAXIMAL event interval such that
            # the interval length doesn't exceed MAX_N_TOKENS.
            min_ei1, max_ei2 = self.midi.find_maximal_event_interval_covering(ei1, ei2, MAX_N_TOKENS)
            assert self.midi.ei2toki(max_ei2) - self.midi.ei2toki(min_ei1) <= MAX_N_TOKENS

            if self.split == 'train':
                # If train, randomly sample between the minimal and maximal interval.
                ei1, ei2 = sample_interval(min_ei1, ei1, ei2, max_ei2)
            else:
                # Otherwise, we take the minimal interval.
                ei1, ei2 = min_ei1, max_ei2

        Y: Tensor = self.midi.align(afi1, afi2, ei1, ei2)
        item = AlignDataset.Item(id_=(self.id_, self.clip_idx),
                                 audio_frames=self.audio_clips[self.clip_idx],
                                 Y=Y,
                                 **(self.midi.encode(ei1, ei2)._asdict()))

        if self.split == 'test':
            item = AlignDataset.ItemWithMetadata(audio=self.audio,
                                                 events=self.midi.get_frame_indices(),
                                                 audio_resolution=AUDIO_RESOLUTION,
                                                 event_resolution=EVENT_RESOLUTION,
                                                 audio_frame_interval=(afi1, afi2),
                                                 event_interval=(ei1, ei2),
                                                 **item._asdict())
        return item


    def __iter__(self) -> Iterator[Item]:
        return self


    # @profile
    def collate_fn(batch: List[Item]) -> Batch:
        batch_size = len(batch)
        id_= [item.id_ for item in batch]

        # audio_frames
        audio_frames: Tensor = torch.stack([item.audio_frames for item in batch])
        assert audio_frames.shape == (batch_size, N_FRAMES_PER_CLIP, N_MELS)

        # score_ids
        tensors = [item.score_ids for item in batch]
        max_n_tokens = max(len(t) for t in tensors)
        score_ids: LongTensor = torch.stack(
            [F.pad(t, (0, max_n_tokens-len(t)), value=TOKEN_ID['[PAD]']) for t in tensors])
        assert score_ids.shape == (batch_size, max_n_tokens)

        # score_attn_mask
        tensors = [item.score_attn_mask for item in batch]

        # Prevent non-padding queries from attending to padding keys
        score_attn_mask = [
            F.pad(t, (0, max_n_tokens-t.size(1)), value=False)
            for t in tensors
        ]
        # Allow padding keys to attend to anything (doesn't matter)
        score_attn_mask: BoolTensor = torch.stack([
            F.pad(t, (0, 0, 0, max_n_tokens-t.size(0)), value=True) 
            for t in score_attn_mask
        ])
        assert score_attn_mask.shape == (batch_size, max_n_tokens, max_n_tokens)

        # event_pos
        event_pos = [item.event_pos for item in batch]
        max_n_events = max(len(t) for t in event_pos)

        # event_padding_mask
        tensors = [torch.ones(len(t), dtype=torch.bool) for t in event_pos]
        event_padding_mask: BoolTensor = torch.stack([
            F.pad(t, (0, max_n_events-len(t)), value=False) for t in tensors])
        assert event_padding_mask.shape == (batch_size, max_n_events)

        # score_to_events
        tensors = [item.score_to_event for item in batch]
        score_to_event: BoolTensor = torch.stack([
            F.pad(t, (0, max_n_tokens - t.size(1), 0, max_n_events - t.size(0)),
            value=False) for t in tensors])
        assert score_to_event.shape == (batch_size, max_n_events, max_n_tokens)

        # Y
        tensors = [item.Y for item in batch]
        Y: BoolTensor = torch.stack([
            F.pad(t, (0, max_n_events-t.size(1)), value=False
            ) for t in tensors])
        assert Y.shape == (batch_size, N_FRAMES_PER_CLIP, max_n_events)

        return AlignDataset.Batch(
            id_=id_,
            audio_frames=audio_frames,
            score_ids=score_ids,
            score_attn_mask=score_attn_mask,
            score_to_event=score_to_event,
            event_pos=event_pos,
            event_padding_mask=event_padding_mask,
            Y=Y)


class MaestroDataset(AlignDataset):

    def __init__(self,
                 root_dir: str,
                 split: Literal['train', 'validation', 'test'],
                 shuffle: bool = True):
        
        assert split in {'train', 'validation', 'test'}

        metadata = pd.read_csv(os.path.join(root_dir, "maestro-v3.0.0.csv"))
        metadata_split = metadata[metadata['split'] == split]
        ids: Sequence[int] = metadata_split.index
        midi_uris: Sequence[str] = metadata_split['midi_filename'].apply(
                lambda filename: os.path.join(root_dir, filename))
        audio_uris: Sequence[str] = metadata_split['audio_filename'].apply(
                lambda filename: os.path.join(root_dir, filename))

        super().__init__(ids, midi_uris, audio_uris, split, shuffle)

        ### LENGTH COMPUTING ###
        # Convert seconds to audio frames
        duration = metadata_split['duration'] / AUDIO_RESOLUTION
        # How many strides fit into the duration
        duration = np.ceil(duration / N_FRAMES_PER_STRIDE)
        # Clips are bigger than strides, but they should count as one
        shift = (N_FRAMES_PER_CLIP - N_FRAMES_PER_STRIDE) // N_FRAMES_PER_STRIDE
        duration = (duration - shift).apply(lambda x: max(x, 1))
        self.len = int(duration.sum())


    def __len__(self):
        return self.len

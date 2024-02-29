from collections.abc import Sequence
from typing import Literal, List, Iterator, NamedTuple, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from torch import Tensor, LongTensor, BoolTensor

import pandas as pd
import os

from ..utils.constants import *
from .score_tools import ParsedMIDI
from .audio_tools import load_spectrogram, unfold_spectrogram, sample_interval, get_num_frames
from .dataset_preprocessing_tools import generate_splits

import numpy as np

import math
from tqdm import tqdm


class AlignDataset(IterableDataset):
    """
    Note 1: If you want to shuffle, pass `shuffle = True` in the initialization.
    Shuffling outside of this class definition (e.g., as in DataLoader) is HIGHLY
    DISCOURAGED, since the `__getitem__()` operation of this class is no longer
    asymtotically O(1).

    Note 2: Why do we use `IterableDataset`? Performance degradation is huge if we
    use the map-based `Dataset` module instead.
    """

    class Item(NamedTuple):
        id_: Tuple[int, int]
        clip_idx: int
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
                 midi_uris: Sequence[str],
                 audio_uris: Sequence[str],
                 split: Literal['train', 'validation', 'test'],
                 shuffle: bool = False,
                 ids: Sequence | None = None,
                 ns_frames: Sequence[int] | None = None,
                 start_at: int = 0  # For fault tolerance
                 ):

        self.start_at = start_at
        self.n_uris = len(midi_uris)
        assert len(audio_uris) == self.n_uris

        if ids is None:
            ids = range(self.n_uris)
        assert len(ids) == len(set(ids)) == self.n_uris

        self.split = split

        if shuffle:
            ids, midi_uris, audio_uris = np.array(ids), np.array(midi_uris), np.array(audio_uris)
            perm = np.random.permutation(self.n_uris)
            ids, midi_uris, audio_uris = ids[perm], midi_uris[perm], audio_uris[perm]
            if ns_frames is not None:
                ns_frames = np.array(ns_frames)[perm]

        if ns_frames is None:
            print("Preprocessing...")

        self.data: List[Tuple] = []     # (id_, midi_uri, audio_uri, clip_idx)\
        for i in range(self.n_uris):

            id_, midi_uri, audio_uri = ids[i], midi_uris[i], audio_uris[i]
            if ns_frames is not None:
                n_frames = ns_frames[i]
            else:
                n_frames = get_num_frames(audio_uri)

            n_clips = math.ceil(
                max(n_frames - N_FRAMES_PER_CLIP, 0) / N_FRAMES_PER_STRIDE
            ) + 1

            for clip_idx in range(n_clips):
                self.data.append((
                    id_, midi_uri, audio_uri, clip_idx
                ))

        self.curr_idx: int = None
        self.curr_id_ = None
        self.curr_midi: ParsedMIDI = None
        self.curr_audio_clips: Tensor = None


    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, i: int) -> Item:
        id_, midi_uri, audio_uri, clip_idx = self.data[i]

        if id_ != self.curr_id_:
            self.curr_id_ = id_
            audio = load_spectrogram(audio_uri)
            self.curr_audio_clips = unfold_spectrogram(audio)
            self.curr_midi = ParsedMIDI(midi_uri, lazy_align=True)

        midi, audio_clips = self.curr_midi, self.curr_audio_clips

        # Audio frame interval, [afi1, afi2).
        # INCLUSIVE on the left, EXCLUSIVE on the right.
        afi1 = clip_idx * N_FRAMES_PER_STRIDE
        afi2 = afi1 + N_FRAMES_PER_CLIP

        # Event interval corresponding to [afi1, afi2).
        # This is the most MINIMAL interval of events that covers the span of
        # audio from afi1 to afi2.
        ei1, ei2 = midi.find_minimal_event_interval_covering(afi1, afi2)
        n_tokens = midi.ei2toki(ei2) - midi.ei2toki(ei1)

        # Search for a larger event interval
        # if this minimal event interval doesn't exceed MAX_N_TOKENS.
        if n_tokens <= MAX_N_TOKENS:
            # [min_ei1, max_ei2) is the MAXIMAL event interval
            # such that the interval length doesn't exceed MAX_N_TOKENS.
            min_ei1, max_ei2 = midi.find_maximal_event_interval_covering(ei1, ei2, MAX_N_TOKENS)
            assert 1 <= midi.ei2toki(max_ei2) - midi.ei2toki(min_ei1) <= MAX_N_TOKENS

            if self.split == 'train':
                # If the split is train, randomly sample an event interval
                # between the minimal and maximal intervals.
                ei1, ei2 = sample_interval(min_ei1, ei1, ei2, max_ei2)
            else:
                # Otherwise, take the maximal interval.
                ei1, ei2 = min_ei1, max_ei2

        Y = midi.align(afi1, afi2, ei1, ei2)
        item = AlignDataset.Item(id_=id_,
                                 clip_idx=clip_idx, 
                                 audio_frames=audio_clips[clip_idx],
                                 Y=Y,
                                 **(midi.encode(ei1, ei2)._asdict()))

        if self.split == 'test':
            item = AlignDataset.ItemWithMetadata(audio_uri=audio_uri,
                                                 audio_resolution=AUDIO_RESOLUTION,
                                                 audio_frame_interval=(afi1, afi2),
                                                 midi_uri=midi_uri,
                                                 events=midi.get_frame_indices(),
                                                 event_resolution=EVENT_RESOLUTION,
                                                 event_interval=(ei1, ei2),
                                                 **item._asdict())
        return item


    def __iter__(self) -> Iterator[Item]:
        for idx in range(self.start_at, self.__len__()):
            self.idx = idx
            yield self.__getitem__(idx)


    # @profile
    @staticmethod
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
                 shuffle: bool = False,
                 start_at: int = 0):
        assert split in {'train', 'validation', 'test'}
        metadata = pd.read_csv(os.path.join(root_dir, "maestro-v3.0.0.csv"))

        if 'n_frames' not in metadata.columns:
            print("Preprocessing...")
            ns_frames = []
            for audio_uri in tqdm(metadata['audio_filename']):
                ns_frames.append(
                    get_num_frames(os.path.join(root_dir, audio_uri))
                )
            metadata['n_frames'] = ns_frames
            metadata.to_csv(os.path.join(root_dir, "maestro-v3.0.0.csv"), index=False)

        metadata_split = metadata[metadata['split'] == split]

        ids = list(metadata_split.index)
        midi_uris = list(metadata_split['midi_filename'].apply(
                lambda filename: os.path.join(root_dir, filename)))
        audio_uris = list(metadata_split['audio_filename'].apply(
                lambda filename: os.path.join(root_dir, filename)))
        ns_frames = list(metadata_split['n_frames'])
        assert len(ids) == len(midi_uris) == len(audio_uris) == len(ns_frames)

        super().__init__(midi_uris, audio_uris, split, shuffle, ids, ns_frames)

class MusicNetDataset(AlignDataset):

    def __init__(self,
                 root_dir: str,
                 split: Literal['train', 'validation', 'test'],
                 shuffle: bool = False,
                 start_at: int = 0):
        assert split in {'train', 'validation', 'test'}
        metadata = pd.read_csv(os.path.join(root_dir, "musicnet_metadata.csv"))

        if 'split' not in metadata.columns:
            metadata = generate_splits(metadata, 75, 10, 15)
            metadata.to_csv(os.path.join(root_dir, "musicnet_metadata.csv"), index=False)

        if 'n_frames' not in metadata.columns:
            print("Preprocessing...")
            ns_frames = []
            for id_ in tqdm(metadata['id']):
                ns_frames.append(
                    get_num_frames(os.path.join(root_dir, f"{split}_labels/{id_}.wav"))
                )
            metadata['n_frames'] = ns_frames
            metadata.to_csv(os.path.join(root_dir, "musicnet_metadata.csv"), index=False)

        metadata_split = metadata[metadata['split'] == split]

        ids = list(metadata_split.index)
        midi_uris = list(metadata_split['midi_filename'].apply(
                lambda filename: os.path.join(root_dir, filename)))
        audio_uris = list(metadata_split['audio_filename'].apply(
                lambda filename: os.path.join(root_dir, filename)))
        ns_frames = list(metadata_split['n_frames'])
        assert len(ids) == len(midi_uris) == len(audio_uris) == len(ns_frames)

        super().__init__(midi_uris, audio_uris, split, shuffle, ids, ns_frames)
from itertools import cycle
from pathlib import Path
from torch.utils.data import IterableDataset
from collections.abc import Sequence
from typing import Literal, List, Iterator, NamedTuple, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor, LongTensor, BoolTensor
import os
from ..utils.constants import *
from .toolbox import ParsedMIDI, load_spectrogram, unfold_spectrogram, sample_interval
import pandas as pd
from pandas import DataFrame
import random
from memory_profiler import profile


class MaestroDataset(IterableDataset):

    class Item(NamedTuple):
        """Return type of the dataset iterator when split is 'train' or 
        'validation'.

        Args:
            id_ (Tuple[int, int]): Unique identifier of the item. First index
                corresponds to the index of the audio file as written in
                metadata.csv. The second index corresponds to the clip number
                of this audio clip within the whole audio file that contains the 
                clip.
            audio_frames (Tensor): ~10 sec spectrogram. 
                Size: (N_MELS, N_FRAMES_PER_CLIP).
            score_ids (BoolTensor): Tokenization (int array representation) of the MIDI. 
                This is the encoding that the transformer receives. Size: (n_tokens,).
            event_padding_mask (BoolTensor): Marks the positions of event markers in the
                score_ids with 1s, the rest with 0s. Size: (n_tokens,)
            score_attn_mask (BoolTensor): Self-attention mask for the score encoder.
                Size: (n_tokens, n_tokens).
            Y (Tensor): gold alignment matrix. Size: (N_FRAMES_PER_CLIP, n_events).
        """
        id_: Tuple[int, int]
        audio_frames: Tensor

        score_ids: LongTensor
        score_attn_mask: BoolTensor

        score_to_event: BoolTensor
        event_pos: LongTensor

        Y: BoolTensor


    class ItemWithMetadata(Item):
        """Return type of the dataset iterator when split is 'test'. Extends
        the `Item` type above.

        Args:
            audio (Tensor): Full spectrogram that the current clip belongs to.
                Size: (n_frames, N_MELS)
            midi (DataFrame): Pandas dataframe representation of the full midi.
            audio_frame_interval (Tuple[int, int]): starting frame index and 
                ending frame index of the current audio clip.
            event_interval (Tuple[int, int]): starting event index and ending
                frame index of the current event clip.
        """
        audio: Tensor
        midi: DataFrame
        audio_frame_interval: Tuple[int, int]
        event_interval: Tuple[int, int]


    class Batch(NamedTuple):
        """Return type of the dataloader when `func:collate_fn` is applied.
        """
        id_: List[Tuple[int, int]]
        audio_frames: Tensor                # (B, N_FRAMES_PER_CLIP, N_MELS)

        score_ids: LongTensor               # (B, max_n_tokens)
        score_attn_mask: BoolTensor         # (B, max_n_tokens, max_n_tokens)

        score_to_event: BoolTensor          # (B, max_n_events, max_n_tokens)
        event_pos: List[LongTensor]         # (B) * (n_events,)
        event_padding_mask: BoolTensor      # (B, max_n_events)

        Y: BoolTensor                       # (B, N_FRAMES_PER_CLIP, max_n_events)


    def __init__(self, 
                 root_dir: str, # path to MAESTRO dataset
                 split: Literal['train', 'validation', 'test'] = 'train'):

        if split not in {'train', 'validation', 'test'}:
            raise TypeError("Allowed values for split: 'train', 'validation', 'test'")

        self.split = split
        self.metadata = pd.read_csv(os.path.join(root_dir, "maestro-v3.0.0.csv"))
        
        metadata_split = self.metadata[self.metadata['split'] == split]

        self.randomized = split == 'train'
        if self.randomized:
            metadata_split = metadata_split.sample(frac=1)

        self.ids: Sequence[int] = metadata_split.index
        self.midi_uris: Sequence[str] = metadata_split['midi_filename'].apply(
                lambda filename: os.path.join(root_dir, filename))
        self.audio_uris: Sequence[str] = metadata_split['audio_filename'].apply(
                lambda filename: os.path.join(root_dir, filename))


    def _preprocess(self) -> Iterator[Item]:

        for id_, audio_uri, midi_uri in zip(self.ids, self.audio_uris, self.midi_uris):

            audio = load_spectrogram(audio_uri)
            audio_clips: List[Tuple[int, Tensor]] = list(enumerate(unfold_spectrogram(audio)))
            n_frames = N_FRAMES_PER_STRIDE * (len(audio_clips)-1) \
                     + N_FRAMES_PER_CLIP

            if self.randomized:
                random.shuffle(audio_clips)

            midi = ParsedMIDI(midi_uri)

            for ci, audio_clip in audio_clips:
                assert len(audio_clip) == N_FRAMES_PER_CLIP

                # Audio frame interval.
                # Inclusive on the left, exclusive on the right.
                afi1 = ci * N_FRAMES_PER_STRIDE
                afi2 = afi1 + N_FRAMES_PER_CLIP

                # Event interval corresponding to [afi1, afi2). This is the most MINIMAL 
                # interval of events that covers the span of music from afi1 and afi2.
                ei1, ei2 = midi.find_minimal_event_interval_covering(afi1, afi2)
                n_tokens = midi.ei2toki(ei2) - midi.ei2toki(ei1)

                # Search for a larger event interval if this minimal event interval
                # doesn't exceed MAX_N_TOKENS.
                if n_tokens <= MAX_N_TOKENS:
                    # [min_ei1, max_ei2) is the MAXIMAL event interval such that
                    # the interval length doesn't exceed MAX_N_TOKENS.
                    min_ei1, max_ei2 = midi.find_maximal_event_interval_covering(ei1, ei2, MAX_N_TOKENS)
                    assert midi.ei2toki(max_ei2) - midi.ei2toki(min_ei1) <= MAX_N_TOKENS

                    if self.randomized:
                        # If train, randomly sample between the minimal and maximal interval.
                        ei1, ei2 = sample_interval(min_ei1, ei1, ei2, max_ei2)
                    else:
                        # Otherwise, we take the minimal interval.
                        ei1, ei2 = min_ei1, max_ei2


                encoding: ParsedMIDI.Encoding = midi.encode(ei1, ei2)
                Y: Tensor = midi.align(afi1, afi2, ei1, ei2)

                item = MaestroDataset.Item(id_=(id_, ci),
                                           audio_frames=audio_clip,
                                           Y=Y,
                                           **encoding._asdict())

                if self.split == 'test':
                    item = MaestroDataset.ItemWithMetadata(audio=audio,
                                                           midi=midi.midi,
                                                           audio_frame_interval=(afi1, afi2),
                                                           event_interval=(ei1, ei2),
                                                           **item._asdict())
                yield item


    def __iter__(self) -> Iterator[Item]:
        """Iterator for the MAESTRO dataset.

        Yields:
            Item: See the doc above.
        """
        if self.split == 'train':
            return cycle(self._preprocess())
        else:
            return self._preprocess()


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
        score_attn_mask: BoolTensor = torch.stack([
            F.pad(t, (0, max_n_tokens-t.size(1), 0, max_n_tokens-t.size(0)),
                  value=False) for t in tensors])
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

        return MaestroDataset.Batch(
            id_=id_,
            audio_frames=audio_frames,
            score_ids=score_ids,
            score_attn_mask=score_attn_mask,
            score_to_event=score_to_event,
            event_pos=event_pos,
            event_padding_mask=event_padding_mask,
            Y=Y)

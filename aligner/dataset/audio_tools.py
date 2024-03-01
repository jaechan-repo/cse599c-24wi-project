import math
import torch
import torch.nn.functional as F
import torchaudio
from typing import Tuple
from ..utils.constants import *
from random import randrange


def t2fi(timestamp: float) -> int:
    return timestamp // AUDIO_RESOLUTION


def sample_interval(imin: int, imax: int,
                    jmin: int, jmax: int, 
                    ) -> Tuple[int, int]:
    assert imin <= imax < jmin <= jmax
    i = randrange(imin, imax+1)
    j = randrange(jmin, jmax+1)
    return i, j


_empty_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=0,
    f_min=F_MIN,
    f_max=F_MAX,
    center=True # pad waveform on both sides
)


_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    f_min=F_MIN,
    f_max=F_MAX,
    center=True # pad waveform on both sides
)


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


def get_num_frames(uri: str) -> int:
    signal, sr = torchaudio.load(uri)
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        signal = resampler(signal)
    signal = torch.mean(signal, dim=0)
    signal: torch.Tensor = _empty_transform(signal)
    return signal.shape[1]


def _pad_spectrogram(signal: torch.Tensor) -> torch.Tensor:
    n_frames = len(signal)
    new_n_frames = math.ceil(n_frames / N_FRAMES_PER_STRIDE) * N_FRAMES_PER_STRIDE  # Multiple of stride size
    new_n_frames = max(new_n_frames, N_FRAMES_PER_CLIP)     # One CLIP at the very least
    signal = F.pad(signal, (0, 0, 0, new_n_frames - n_frames), value=0)
    assert signal.shape == (new_n_frames, N_MELS)
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

import numpy as np
import librosa, pretty_midi
import midi as midi
import util as util
from dtw import *

import sys
sys.path.append('..')
from aligner.utils.constants import SAMPLE_RATE, HOP_LENGTH, AUDIO_RESOLUTION, N_FFT


def align_chroma(score_fp, audio_fp, audio_start, duration):
    """Align a score and audio file using chroma features.
    
    Args:
        score_fp (str): Filepath to the .midi score file.
        audio_fp (str): Filepath to the .wav audio file.
        audio_start (int): Starting timestamp in seconds of the audio clip to align.
        duration (int): Duration in seconds of the audio clip to align."""

    score_synth = pretty_midi.PrettyMIDI(score_fp).fluidsynth(fs=SAMPLE_RATE)
    perf,_ = librosa.load(audio_fp, sr=SAMPLE_RATE, offset=audio_start, duration=duration)
    score_chroma = librosa.feature.chroma_stft(y=score_synth, sr=SAMPLE_RATE, tuning=0, norm=2,
                                               hop_length=HOP_LENGTH, n_fft=N_FFT)
    score_logch = librosa.power_to_db(score_chroma, ref=score_chroma.max())
    perf_chroma = librosa.feature.chroma_stft(y=perf, sr=SAMPLE_RATE, tuning=0, norm=2,
                                              hop_length=HOP_LENGTH, n_fft=N_FFT)
    perf_logch = librosa.power_to_db(perf_chroma, ref=perf_chroma.max())
    
    alignment = dtw(perf_logch, score_logch, keep_internals=False, open_begin=True, open_end=True)
    wp = np.array(list(zip(alignment.index1, alignment.index2)))

    return wp


def align_spectra(score_fp, audio_fp, audio_start, duration):
    """Align a score and audio file using spectra features.
    
    Args:
        score_fp (str): Filepath to the .midi score file.
        audio_fp (str): Filepath to the .wav audio file.
        audio_start (int): Starting timestamp in seconds of the audio clip to align.
        duration (int): Duration in seconds of the audio clip to align."""

    score_synth = pretty_midi.PrettyMIDI(score_fp).fluidsynth(fs=SAMPLE_RATE)
    perf,_ = librosa.load(audio_fp, sr=SAMPLE_RATE, offset=audio_start, duration=duration)
    score_spec = np.abs(librosa.stft(y=score_synth, hop_length=HOP_LENGTH, n_fft=N_FFT))**2
    score_logspec = librosa.power_to_db(score_spec, ref=score_spec.max())
    perf_spec = np.abs(librosa.stft(y=perf, hop_length=HOP_LENGTH, n_fft=N_FFT))**2
    perf_logspec = librosa.power_to_db(perf_spec, ref=perf_spec.max())

    alignment = dtw(perf_logspec, score_logspec, keep_internals=False, open_begin=True, open_end=True)
    wp = np.array(list(zip(alignment.index1, alignment.index2)))

    return wp


def align_prettymidi(score_fp, audio_fp, start_time, duration, note_start=36, n_notes=48, penalty=None):
    '''Align a MIDI object in-place to some audio data.
    
    Args:
        midi_object (pretty_midi.PrettyMIDI): A pretty_midi.PrettyMIDI class instance describing some MIDI content
        audio_data (np.ndarray): Samples of some audio data
        audio_start (int): Starting timestamp in seconds of the audio clip to align.
        duration (int): Duration in seconds of the audio clip to align.
        note_start (int): Lowest MIDI note number for CQT
        n_notes (int): Number of notes to include in the CQT
        penalty (float): DTW non-diagonal move penalty'''
    
    def extract_cqt(audio_data, note_start, n_notes):
        # Compute CQT
        cqt = librosa.cqt(
            audio_data, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
            fmin=librosa.midi_to_hz(note_start), n_bins=n_notes)
        # Transpose so that rows are spectra
        cqt = cqt.T
        # Compute log-amplitude
        cqt = librosa.amplitude_to_db(librosa.magphase(cqt)[0], ref=cqt.max())
        # L2 normalize the columns
        cqt = librosa.util.normalize(cqt, norm=2., axis=1)
        # Compute the time of each frame
        times = librosa.frames_to_time(np.arange(cqt.shape[0]), SAMPLE_RATE, HOP_LENGTH)
        return cqt, times

    audio_data, _ = librosa.load(audio_fp, SAMPLE_RATE, offset=start_time, duration=duration)
    midi_object = pretty_midi.PrettyMIDI(score_fp)
    # Get synthesized MIDI audio
    midi_audio = midi_object.fluidsynth(fs=SAMPLE_RATE)
    # Compute CQ-grams for MIDI and audio
    midi_gram, midi_times = extract_cqt(
        midi_audio, SAMPLE_RATE, HOP_LENGTH, note_start, n_notes)
    audio_gram, audio_times = extract_cqt(
        audio_data, SAMPLE_RATE, HOP_LENGTH, note_start, n_notes)
    # Compute distance matrix; because the columns of the CQ-grams are
    # L2-normalized we can compute a cosine distance matrix via a dot product
    distance_matrix = 1 - np.dot(midi_gram, audio_gram.T)
    
    alignment = dtw(distance_matrix, keep_internals=False, open_begin=True, open_end=True)
    wp = np.array(list(zip(alignment.index1, alignment.index2)))
    return wp
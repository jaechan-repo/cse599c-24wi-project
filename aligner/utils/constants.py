# Time - Audio
SAMPLE_RATE = 32000
HOP_LENGTH = 320     # 10 ms hop duration
AUDIO_RESOLUTION = HOP_LENGTH / SAMPLE_RATE # 10 ms
N_FFT = 2 * HOP_LENGTH

N_FRAMES_PER_CLIP = 1024
N_FRAMES_PER_STRIDE = N_FRAMES_PER_CLIP // 2

# Time - MIDI
EVENT_RESOLUTION = AUDIO_RESOLUTION      # 10 ms
MAX_N_TOKENS = 1024

# Frequency
N_MELS = 512
MEL_LO_HZ = 20.0
MEL_FMIN = 20.0
MEL_FMAX = 7600.0

# Token id
TOKEN_ID = {
    '[event]': 129, # event marker
    '[BOS]': 128,   # beginning of score/sequence
}
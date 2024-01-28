# Time - Audio
SAMPLE_RATE = 32000
HOP_WIDTH = 320     # 10 ms hop duration
FFT_SIZE = 2048
NUM_FRAMES = 1024

# Time - MIDI
EVENT_RESOLUTION = 0.020    # 20 ms

# Frequency
NUM_MEL_BINS = 512
MEL_LO_HZ = 20.0
MEL_FMIN = 20.0
MEL_FMAX = 7600.0

# Token id
TOKEN_ID = {
    '[event]': 129, # event marker
    '[null]': 128,  # onset of silence
}
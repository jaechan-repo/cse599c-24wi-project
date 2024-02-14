from typing import Dict

### Time - Audio ###
SAMPLE_RATE = 32000
HOP_LENGTH = 320     # 10 ms hop duration
AUDIO_RESOLUTION = HOP_LENGTH / SAMPLE_RATE # 10 ms
N_FFT = 2 * HOP_LENGTH

N_FRAMES_PER_CLIP = 1024
N_FRAMES_PER_STRIDE = N_FRAMES_PER_CLIP // 2

### Time - MIDI ###
EVENT_RESOLUTION = AUDIO_RESOLUTION      # 10 ms
MAX_N_TOKENS = 2048

### Frequency ###
N_MELS = 128
F_MIN = 20.0
F_MAX = 5000.0   
# TODO: For piano music, highest frequency is 4186Hz
# Prone to change for v1.

### Token id ###
N_PITCHES = 128
TOKEN_ID: Dict[str, int] = {str(i): i for i in range(N_PITCHES)} | {
    '[BOS]': 128,   # beginning of score
    '[event]': 129, # event marker
    '[PAD]': 130,   # end of score / padding
}

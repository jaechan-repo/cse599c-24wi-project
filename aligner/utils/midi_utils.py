from midiutil import MIDIFile
import torch

def create_MIDI(alignment_matrix: torch.Tensor, event_info: list, save_path: str, f: int = 10, duration: int = 200):
    """
    Args: 
        alignment_matrix (Tensor): (E, F), where N is the number of consecutive audio clips, E is the number of events and F is the number of frames
        event_info (list): (E), a list of lists storing the pitches played at each event 
        save_path (str): path to save MIDI file to
        f (int): length of each audio frame in ms (default 10), at most 1000
        duration (int): length of each note in ms. must be divisible by f
    Output:
        saves MIDI file in save_path
    """
    assert f <= 1000
    assert duration % f == 0
    E, F = alignment_matrix.shape
    # alignment_matrix = alignment_matrix.transpose(0, 1).reshape(E, N*F)

    mask = alignment_matrix.max(dim=1).values > 0
    indices = torch.argmax(alignment_matrix, dim=1)
    time = torch.where(mask, indices, torch.tensor(-1))
    midi_file = MIDIFile(numTracks=1, ticks_per_quarternote=int(1000/f), eventtime_is_ticks=True)
    track = 0
    midi_file.addTempo(track=track, time=0, tempo=60)


    for time, event in zip(time, event_info):
        for pitch in event:
            if time >= 0:
                midi_file.addNote(track=track, channel=0, pitch=pitch, time=time.item(), duration=duration/f, volume=64)
    with open(save_path, 'wb') as f:
        midi_file.writeFile(f)
### Evaluation code for audio-score alignment

import torch
from torch.utils.data import DataLoader
import tqdm
import argparse
from typing import Literal, Tuple, Callable

import sys
sys.path.append('..')
from aligner.utils.metrics import temporal_distance_vec, binary_accuracy_vec, monotonicity_vec, score_coverage_vec
from aligner.utils.metrics import compute_loss
from aligner.utils.decode import max_decode, DTW
from aligner.data_loader.data_loader import MaestroDataset

# TODO: Update the following imports with actual model
from aligner.model import Aligner

# TODO: Add imports for other evaluation benchmark datasets


def load_model(model_path):
    model = Aligner()   # TODO: Update with actual model + args
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def evaluate(
        model: torch.nn.Module, 
        dataloader: torch.utils.data.DataLoader, 
        decoding: Callable[[torch.Tensor], torch.Tensor], 
        tolerance: float) -> Tuple[float, float, float, float, float]:
    
    """Evaluate the model on the evaluation dataset.
    
    Args:
        model (torch.nn.Module): Model to evaluate.
        dataloader (torch.utils.data.DataLoader): Evaluation dataset.
        decoding (Callable): Decoding method to use on cross-attention alignment matrix. Either \textbf{max_decode} or \textbf{DTW}.
        tolerance (float): Tolerance threshold for alignment distance.
        
    Returns:
        float: Average loss on the evaluation dataset.
        float: Average temporal alignment distance on the evaluation dataset.
        float: Average binary alignment accuracy on the evaluation dataset.
        float: Average monotonicity on the evaluation dataset.
        float: Average score coverage on the evaluation dataset."""

    total_loss = 0
    total_distance = 0
    total_accuracy = 0
    total_coverage = 0
    total_monotonicity = 0

    # Compute loss and metrics for each batch with tqdm
    for batch in tqdm(dataloader):
        '''
        Each sample is a dictionary with the following fields:
            (0) "wav_uri": absolute path to audio file
            (1) "midi_uri": absolute path to midi file
            (2) "audio_clip": ~10 sec spectrogram. Size: (n_mels, n_frames).
            (3) "Y": alignment matrix. Size: (n_frames, n_events).
            (4) "input_ids": Integer array representation (tokenization) of the MIDI.
                This is the encoding that the transformer model receives. During
                tokenization, MIDI-scaled pitches [0, 121] are directly taken as unique 
                tokens, and special tokens such as [BOS] and [event] are attached. See
                `TOKEN_ID` in `constants.py` for their integer codes. Size: (n_tokens,).
            (5) "event_mask": Array filled with 0s and 1s, where 1s fill positions where
                event markers lie in "input_ids". Size: (n_tokens,).
            (6) "attn_mask": mask for self-attention. Size: (n_tokens, n_tokens).'''
        
        # Assuming batch of shape (N, E, X), where N is the batch size, E is the number of MIDI events in the score, and X is the number of audio frames
        # TODO: Update with actual batch format
        audio_frames, score_events, Y, midi_event_timestamps = batch

        # get model predictions given audio frames and score events
        # TODO: Update with actual model forward pass API
        Y_pred = model(audio_frames, score_events)

        # loss should use the non-decoded alignment matrix???
        total_loss += compute_loss(Y_pred, Y, midi_event_timestamps, 'mean')

        # decode soft cross-attn alignment matrix into binary alignment matrix
        Y_pred_binary = decoding(Y_pred)

        total_distance += temporal_distance_vec(Y_pred_binary, Y, midi_event_timestamps, tolerance, 'mean')
        total_accuracy += binary_accuracy_vec(Y_pred_binary, Y, midi_event_timestamps, tolerance, 'mean')
        total_monotonicity += monotonicity_vec(Y_pred_binary, midi_event_timestamps, 'mean')
        total_coverage += score_coverage_vec(Y_pred_binary, 'mean')

    return total_loss / len(dataloader), total_distance / len(dataloader), total_accuracy / len(dataloader), total_monotonicity / len(dataloader), total_coverage / len(dataloader)
        

def main():
    # Parse arguments: model_path, evaluation_data_path, tau
    parser = argparse.ArgumentParser(description='Evaluation script')
    parser.add_argument('model_path', type=str, help='Path to the trained model')
    parser.add_argument('evaluation_data_path', type=str, help='Path to the evaluation data')
    parser.add_argument('decoding', type=str, choices=['max', 'dtw'], help='Decoding method to use on cross-attention alignment matrix')
    parser.add_argument('tolerance', type=float, help='Tolerance threshold for alignment distance. Minimum threshold is the MIDI score event duration / 2')
    args = parser.parse_args()

    model_path = args.model_path
    evaluation_data_path = args.evaluation_data_path
    decoding = max_decode if args.decoding == 'max' else DTW
    tolerance = args.tolerance

    model = load_model(model_path)
    evaluation_dataset = MaestroDataset(evaluation_data_path, 'test')
    dataloader = DataLoader(evaluation_dataset, batch_size=32)
    
    loss, distance, accuracy, monotonicity, coverage = evaluate(model, dataloader, decoding, tolerance)
    print(f'Evaluation Metrics with Tau={tolerance}\nLoss: {loss:.2f}, Distance: {distance:.2f}, Accuracy: {accuracy:.2f}, Monotonicity: {monotonicity:.2f}, Coverage: {coverage:.2f}')

if __name__ == "__main__":
    main()

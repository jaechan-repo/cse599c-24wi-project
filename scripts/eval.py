### Evaluation code for audio-score alignment

import json
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

from aligner.dataset import MaestroDataset
from aligner.utils.constants import *
from aligner.model import AlignerModel, ModelConfig
from aligner.utils.constants import *


# TODO: Add imports for other evaluation benchmark datasets


def load_model(model_path):
    model = AlignerModel(ModelConfig())
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def evaluate(
        model, 
        dataloader: torch.utils.data.DataLoader, 
        decoding: Callable[[torch.Tensor], torch.Tensor], 
        tolerance: float) -> Tuple[float, float, float, float, float]:
    
    """Evaluate the model on the evaluation dataset.
    
    Args:
        model: Model to evaluate.
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
        Y = batch.Y_b.transpose(-2, -1) # (N, X, E) -> (N, E, X)

        # get model predictions given audio frames and score events
        Y_pred = model(
            batch.audio_frames_b,
            score_ids=batch.score_ids_b,
            score_attn_mask=batch.score_attn_mask_b,
            event_padding_mask=batch.event_padding_mask_b
        ).tranpose(-2, -1) # (N, X, E) -> (N, E, X)

        total_loss += compute_loss(Y_pred, batch.Y, midi_event_timestamps, 'mean')

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
    dataloader = DataLoader(
        dataset=evaluation_dataset, 
        num_workers=4,
        collate_fn=MaestroDataset.collate_fn
    )
    
    loss, distance, accuracy, monotonicity, coverage = evaluate(model, dataloader, decoding, tolerance)
    print(f'Evaluation Metrics with Tau={tolerance}\nLoss: {loss:.2f}, Distance: {distance:.2f}, Accuracy: {accuracy:.2f}, Monotonicity: {monotonicity:.2f}, Coverage: {coverage:.2f}')

    # organize metrics into a dictionary
    metrics = {
        'Loss': loss,
        'Distance': distance,
        'Accuracy': accuracy,
        'Monotonicity': monotonicity,
        'Coverage': coverage
    }

    metrics_json = json.dumps(metrics)
    
    # write the JSON string to a file
    with open('metrics.json', 'w') as f:
        f.write(metrics_json)


if __name__ == "__main__":
    main()

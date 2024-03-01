### Evaluation code for audio-score alignment

import json
import torch
from torch.utils.data import DataLoader
import tqdm
import argparse
from typing import Literal, Tuple, Callable

import sys
sys.path.append('..')
from aligner.utils.metrics import temporal_distance, binary_accuracy, monotonicity, score_coverage
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
    model: AlignerModel, 
    dataloader: torch.utils.data.DataLoader, 
    decoding: Callable[[torch.Tensor], torch.Tensor], 
    tolerance: float) -> Dict[str, float]:
    
    """Evaluate the model on the evaluation dataset.
    
    Args:
        model: Model to evaluate.
        dataloader (torch.utils.data.DataLoader): Evaluation dataset.
        decoding (Callable): Decoding method to use on cross-attention alignment matrix. Either \textbf{max_decode} or \textbf{DTW}.
        tolerance (float): Tolerance threshold for alignment distance.
        
    Returns:
        Dict[str, float]: Dictionary of evaluation metrics for 'Loss', 'Distance', 'Accuracy', 'Monotonicity', 'Coverage'."""

    total_loss = 0
    total_distance = 0
    total_accuracy = 0
    total_coverage = 0
    total_monotonicity = 0

    # Compute loss and metrics with batch size of 1
    for item in tqdm(dataloader):

        Y = item.Y.transpose(-2, -1) # (X, E) -> (E, X)
        score_event_timestamps = item.score_event_timestamps     

        # unsqueeze batch dimension to match model forward input shape
        audio_frames = item.audio_frames.unsqueeze(0)
        score_ids = item.score_ids.unsqueeze(0)
        score_attn_mask = item.score_attn_mask.unsqueeze(0)

        # get model predictions given audio frames and score events
        Y_pred = model(
            audio_frames,
            score_ids=score_ids,
            score_attn_mask=score_attn_mask,
        ).transpose(-2, -1) # (X, E) -> (E, X)

        total_loss += compute_loss(Y_pred, Y, score_event_timestamps)

        # decode soft cross-attn alignment matrix into binary alignment matrix
        Y_pred_binary = decoding(Y_pred)

        total_distance += temporal_distance(Y_pred_binary, Y, score_event_timestamps, tolerance)
        total_accuracy += binary_accuracy(Y_pred_binary, Y, score_event_timestamps, tolerance)
        total_monotonicity += monotonicity(Y_pred_binary)
        total_coverage += score_coverage(Y_pred_binary, Y)

    # organize metrics into a dictionary
    metrics = {
        'Loss': total_loss / len(dataloader),
        'Distance': total_distance / len(dataloader),
        'Accuracy': total_accuracy / len(dataloader),
        'Monotonicity': total_monotonicity / len(dataloader),
        'Coverage': total_coverage / len(dataloader)
    }

    return metrics


def main():
    # Parse arguments: model_path, evaluation_data_path, tau
    args = get_args()

    model_path = args.model_path
    evaluation_data_path = args.evaluation_data_path
    decoding = max_decode if args.decoding == 'max' else DTW
    tolerance = args.tolerance

    model = load_model(model_path)
    evaluation_dataset = MaestroDataset(evaluation_data_path, 'test')
    dataloader = DataLoader(
        dataset=evaluation_dataset,
        num_workers=4,
    )
    
    metrics = evaluate(model, dataloader, decoding, tolerance)
    
    # print evaluation metrics
    print(f"Evaluation Metrics with tolerance={tolerance}")
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}")

    metrics_json = json.dumps(metrics)
    
    # write the JSON string to a file in the "../results" directory
    with open(f'../results/{model_path}_eval_metrics.json', 'w') as f:
        f.write(metrics_json)


def get_args():
    parser = argparse.ArgumentParser(description='Evaluation script')
    parser.add_argument('model_path', type=str, help='Path to the trained model')
    parser.add_argument('evaluation_data_path', type=str, help='Path to the evaluation data')
    parser.add_argument('decoding', type=str, default='dtw', choices=['max', 'dtw'], help='Decoding method to use on cross-attention alignment matrix')
    parser.add_argument('tolerance', type=float, default=EVENT_RESOLUTION / 2, help='Tolerance threshold for alignment distance. Default / minimum threshold is half the MIDI score event resolution, defined as the duration of a single event marker in the score.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()

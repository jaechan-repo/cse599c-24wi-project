### Evaluation code for audio-score alignment

import json
import torch
from torch.utils.data import DataLoader
import tqdm
import argparse
from typing import Literal, Tuple, Callable

import sys
sys.path.append('..')
from aligner.utils.metrics import temporal_distance_v2, binary_accuracy_v2, monotonicity_v2, coverage
from aligner.utils.decode import max_decode_v2, DTW_v2

# TODO: update import below with the correct loss function
from aligner.utils.metrics import compute_loss

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
    decoding: Callable[[torch.Tensor], torch.Tensor]
) -> Dict[str, float]:
    
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

        # Get gold alignment matrix, with extra leading and trailing audio frame
        Y = item.Y.transpose(-2, -1) # (X, E) -> (E, X)

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

        # TODO: compute loss using the correct loss function
        total_loss += compute_loss(Y_pred, Y)

        # decode soft cross-attn alignment matrix into binary alignment matrix
        Y_pred_binary = decoding(Y_pred)

        # drop leading and trailing audio frame column of Y and Y_pred_binary to get aligned section
        Y = Y[:, 1:-1]
        Y_pred_binary = Y_pred_binary[:, 1:-1]

        total_distance += temporal_distance_v2(Y_pred_binary, Y)
        total_accuracy += binary_accuracy_v2(Y_pred_binary, Y)
        total_monotonicity += monotonicity_v2(Y_pred_binary)
        total_coverage += coverage(Y_pred_binary, Y)

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
    decoding = max_decode_v2 if args.decoding == 'max' else DTW_v2

    model = load_model(model_path)
    evaluation_dataset = MaestroDataset(evaluation_data_path, 'test')
    dataloader = DataLoader(
        dataset=evaluation_dataset,
        num_workers=4,
    )
    
    metrics = evaluate(model, dataloader, decoding)
    
    # print evaluation metrics
    print(f"Evaluation Metrics V2")
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}")

    metrics_json = json.dumps(metrics)
    
    # write the JSON string to a file in the "../results" directory
    with open(f'../results/{model_path}_eval_metrics_v2.json', 'w') as f:
        f.write(metrics_json)


def get_args():
    parser = argparse.ArgumentParser(description='Evaluation script')
    parser.add_argument('model_path', type=str, help='Path to the trained model')
    parser.add_argument('evaluation_data_path', type=str, help='Path to the evaluation data')
    parser.add_argument('decoding', type=str, default='dtw', choices=['max', 'dtw'], help='Decoding method to use on cross-attention alignment matrix')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
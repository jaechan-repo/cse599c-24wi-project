### Evaluation code for audio-score alignment

import json
import torch
from tqdm import tqdm
import argparse
from typing import Callable

import sys
sys.path.append('..')

from torch import Tensor, LongTensor

from aligner.inference.decode import max_decode, dtw_decode, softmax
from aligner.inference.metrics \
    import cross_entropy, rmse, temporal_distance, binary_accuracy,\
           monotonicity, score_coverage_ratio_unbatched

from aligner.dataset import MaestroDataset, ItemWithMetaData, AlignerDataset
from aligner.utils.constants import *
from aligner.utils.utils import find_file
from aligner.model import AlignerModel, AlignerLitModel
import os

# TODO: Add imports for other evaluation benchmark datasets


def load_model(model_path: str, device='cuda') -> AlignerModel:
    model = AlignerLitModel.load_from_checkpoint(model_path)
    model.eval()
    return model.model.to(device)


def evaluate(
    model: AlignerModel,
    dataset: AlignerDataset,
    decoding: Callable[[torch.Tensor], torch.Tensor], 
    tolerance: float) -> Dict[str, float]:
    """Evaluate the model on the evaluation dataset.

    Args:
        model: Model to evaluate.
        dataset (torch.utils.data.dataset): Evaluation dataset.
        decoding (Callable): Decoding method to use on cross-attention alignment matrix. Either \textbf{max_decode} or \textbf{DTW}.
        tolerance (float): Tolerance threshold for alignment distance.
        
    Returns:
        Dict[str, float]: Dictionary of evaluation metrics for 'Loss', 'Distance', 'Accuracy', 'Monotonicity', 'Coverage'.
    """
    total_cls_loss = 0
    total_mse_loss = 0
    total_distance = 0
    total_accuracy = 0
    total_coverage = 0
    total_monotonicity = 0

    # Compute loss and metrics with batch size of 1
    item: ItemWithMetaData
    for item in tqdm(dataset):
        event_timestamps = item.event_timestamps

        ### Predictions ###
        Y = item.Y.float()
        Y_tilde = model.forward_unbatched(**item._asdict(),
                                          normalization='none'
                                          ).detach().cpu()
        Y_hat = softmax(Y_tilde)
        Y_hat_binary: LongTensor = decoding(Y_tilde).float()

        ### Metrics ###
        total_cls_loss += cross_entropy(Y_hat, Y)
        total_mse_loss += rmse(Y_hat, Y)
        total_distance += temporal_distance(
                Y_hat_binary, Y, event_timestamps, tolerance)
        total_accuracy += binary_accuracy(
                Y_hat_binary, Y, event_timestamps, tolerance)
        total_monotonicity += monotonicity(Y_hat_binary)
        total_coverage += score_coverage_ratio_unbatched(Y_hat_binary, Y)

    metrics = {
        'CrossEntropyLoss': float(total_cls_loss / len(dataset)),
        'RMSELoss': float(total_mse_loss / len(dataset)),
        'Distance': float(total_distance / len(dataset)),
        'Accuracy': float(total_accuracy / len(dataset)),
        'Monotonicity': float(total_monotonicity / len(dataset)),
        'Coverage': float(total_coverage / len(dataset))
    }
    return metrics


def main():
    # Parse arguments: model_path, evaluation_data_path, tau
    args = get_args()

    model_path = find_file(args.model_path, 'ckpt')
    evaluation_data_path = args.data_path
    decoding = max_decode if args.decoding == 'max' else dtw_decode
    tolerance = args.tolerance

    model = load_model(model_path)
    dataset = MaestroDataset(evaluation_data_path, 'test')
    metrics = evaluate(model, dataset, decoding, tolerance)

    print(f"Evaluation Metrics with Tau={tolerance}\nLoss: {metrics['Loss']:.2f}, Distance: {metrics['Distance']:.2f}, Accuracy: {metrics['Accuracy']:.2f}, Monotonicity: {metrics['Monotonicity']:.2f}, Coverage: {metrics['Coverage']:.2f}")
    metrics_json = json.dumps(metrics)

    # write the JSON string to a file in the "../results" directory
    output_path = os.path.join(args.output_path, "eval_metrics.json")
    os.makedirs(output_path)
    with open(output_path, 'w') as f:
        f.write(metrics_json)


def get_args():
    parser = argparse.ArgumentParser(description='Evaluation script')
    parser.add_argument('--model_path', type=str, help='Path to folder containing the trained model')
    parser.add_argument('--data_path', type=str, help='Path to the evaluation data')
    parser.add_argument('--output_path', type=str, help='Path to output')
    parser.add_argument('--decoding', type=str, default='dtw', choices=['max', 'dtw'], help='Decoding method to use on cross-attention alignment matrix')
    parser.add_argument('--tolerance', type=float, default=EVENT_RESOLUTION / 2, help='Tolerance threshold for alignment distance. Default / minimum threshold is half the MIDI score event resolution, defined as the duration of a single event marker in the score.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()

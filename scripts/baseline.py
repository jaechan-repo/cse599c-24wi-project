from fastdtw import fastdtw
import music21
import torch
import sys
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Literal, Dict, Callable, Tuple
import argparse
import json
import os

import sys
sys.path.append('..')
from aligner.dataset import MaestroDataset
from aligner.utils.metrics import temporal_distance, binary_accuracy, monotonicity, score_coverage
from aligner.utils.constants import AUDIO_RESOLUTION, N_FRAMES_PER_CLIP

from baseline.algos import align_chroma, align_spectra, align_prettymidi


# run baseline over test set
def run_baseline(
    evaluation_data_path: torch.utils.data.Dataset, 
    align_method: Callable
) -> Dict[str, float]:

    # Load the test set
    # TODO: Replace MaestroDataset with the appropriate dataset class
    evaluation_dataset = MaestroDataset(evaluation_data_path, 'test')
    dataloader = DataLoader(dataset=evaluation_dataset)

    total_distance = 0
    total_accuracy = 0
    total_coverage = 0

    # Compute the DTW alignment for each test set
    loop = tqdm(dataloader)
    for item in tqdm(dataloader):
        Y = item.Y.transpose(-2, -1) # (X, E) -> (E, X)
        num_score_events = Y.shape[-2]

        score_fp = item.score_fp
        audio_fp = item.audio_fp

        start_time = item.start_idx * AUDIO_RESOLUTION
        duration = N_FRAMES_PER_CLIP * AUDIO_RESOLUTION

        # Align the score and audio to get the warping path (wp)
        wp = align_method(score_fp, audio_fp, start_time, duration)

        # construct the predicted alignment matrix
        Y_pred = torch.zeros(Y.shape)
        Y_pred[wp[:, 1], wp[:, 0]] = 1

        # construct the score event timestamps
        # example: for a score of duration 100 ms and resolution 10 ms, we get
        #   [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
        score_event_timestamps = torch.Tensor([(i * AUDIO_RESOLUTION) + (AUDIO_RESOLUTION / 2) for i in range(num_score_events)])

        total_distance += temporal_distance(Y_pred, Y, score_event_timestamps, tolerance=AUDIO_RESOLUTION)
        total_accuracy += binary_accuracy(Y_pred, Y, score_event_timestamps, tolerance=AUDIO_RESOLUTION)
        total_coverage += score_coverage(Y_pred)

        loop.set_description(f'Distance: {total_distance / len(dataloader):.2f}, Acc: {total_accuracy / len(dataloader):.2f}, Coverage: {total_coverage / len(dataloader):.2f}')
        loop.update(1)
    
    metrics = {
        'Distance': total_distance / len(dataloader),
        'Accuracy': total_accuracy / len(dataloader),
        'Coverage': total_coverage / len(dataloader)
    }

    return metrics


def main():
    # Parse arguments using argparse
    parser = argparse.ArgumentParser(description='Baseline script')
    parser.add_argument('evaluation_data_path', type=str, help='Path to the evaluation data')
    parser.add_argument('method', type=str, default='chroma', choices=['chroma', 'spectra', 'prettymidi'], help='Features to extract from the MIDI files')
    args = parser.parse_args()

    evaluation_data_path = args.evaluation_data_path
    method = args.method

    align_method = {
        'chroma': align_chroma,
        'spectra': align_spectra,
        'prettymidi': align_prettymidi
    }[method]

    metrics = run_baseline(evaluation_data_path, align_method)

    # Print the metrics
    for key, value in metrics.items():
        print(f'{key}: {value:.2f}')
    
    # create baseline_results directory if it doesn't exist
    os.makedirs('../baseline_results', exist_ok=True)

    # write the JSON string to a file in the "../results" directory
    metrics_json = json.dumps(metrics)
    with open(f'../baseline_results/{method}_metrics.json', 'w') as f:
        f.write(metrics_json)


if __name__ == '__main__':
    main()
import torch
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Literal, Dict, Callable
import argparse
import json
import os

import sys
sys.path.append('..')
from aligner.dataset import MaestroDataset
from aligner.utils.metrics import temporal_distance, binary_accuracy, score_coverage
from aligner.utils.constants import AUDIO_RESOLUTION, EVENT_RESOLUTION, N_FRAMES_PER_CLIP

from baseline.algos import align_chroma, align_spectra, align_prettymidi


def run_baseline(
    evaluation_data_path: torch.utils.data.Dataset, 
    align_method: Callable,
    alignment_type: Literal['clip-to-whole', 'clip-to-clip', 'whole-to-whole']
) -> Dict[str, float]:

    # Load the test set
    # TODO: Replace MaestroDataset with the appropriate dataset class
    evaluation_dataset = MaestroDataset(evaluation_data_path, 'test')
    dataloader = DataLoader(dataset=evaluation_dataset)

    total_distance = 0
    total_accuracy = 0
    total_coverage = 0

    num_clips = 0

    # loop through each (score, audio) pair in the test set
    loop = tqdm(dataloader)
    for item in tqdm(dataloader):
        Y = item.Y.transpose(-2, -1).squeeze(0) # (X, E) -> (E, X)
        total_score_events, total_audio_frames = Y.shape[-2:]

        score_fp = item.score_fp
        audio_fp = item.audio_fp

        # construct the score event timestamps
        # example: for a score of duration 100 ms and resolution 10 ms, we get
        #   [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
        score_event_timestamps = torch.Tensor([(event_index * EVENT_RESOLUTION) + (EVENT_RESOLUTION / 2) for event_index in range(start_score, end_score)])

        # get the number of audio frames and score events to align
        amount_audio = N_FRAMES_PER_CLIP if (alignment_type == 'clip-to-clip' or alignment_type == 'clip-to-whole') else total_audio_frames
        amount_score = int(N_FRAMES_PER_CLIP * AUDIO_RESOLUTION / EVENT_RESOLUTION) if (alignment_type == 'clip-to-clip') else total_score_events

        # loop through each clip in the audio
        for i in range(0, total_audio_frames, amount_audio):
            # get audio to align (in seconds)
            end_audio = min((i + amount_audio), total_audio_frames)
            duration = (end_audio - i) * AUDIO_RESOLUTION
            start_time = i * AUDIO_RESOLUTION

            # get score to align (in frames)
            start_score = 0 if alignment_type == 'clip-to-whole' else int(i * AUDIO_RESOLUTION / EVENT_RESOLUTION)
            end_score = min(start_score + amount_score, total_score_events)

            # Align the score and audio to get the warping path (wp)
            wp = align_method(score_fp, audio_fp, start_time, duration, start_score, end_score, alignment_type)

            # get relevant portion of the true alignment matrix
            Y_align = Y[start_score:end_score, i:end_audio]

            # construct the predicted alignment matrix
            Y_pred = torch.zeros(Y_align.shape)
            Y_pred[wp[:, 1], wp[:, 0]] = 1

            # compute metrics
            total_distance += temporal_distance(Y_pred, Y_align, score_event_timestamps[start_score:end_score], tolerance=AUDIO_RESOLUTION)
            total_accuracy += binary_accuracy(Y_pred, Y_align, score_event_timestamps[start_score:end_score], tolerance=AUDIO_RESOLUTION)
            total_coverage += score_coverage(Y_pred, Y_align)

            num_clips += 1

            loop.set_description(f'Distance: {total_distance / num_clips:.2f}, Acc: {total_accuracy / num_clips:.2f}, Coverage: {total_coverage / num_clips:.2f}')
        
        loop.update(1)
    
    metrics = {
        'Distance': total_distance / num_clips,
        'Accuracy': total_accuracy / num_clips,
        'Coverage': total_coverage / num_clips
    }

    return metrics


def main():
    # Parse arguments using argparse
    parser = argparse.ArgumentParser(description='Baseline script')
    parser.add_argument('evaluation_data_path', type=str, help='Path to the evaluation data')
    parser.add_argument('method', type=str, default='chroma', choices=['chroma', 'spectra', 'prettymidi'], help='Features to extract from the MIDI files')
    parser.add_argument('alignment_type', type=str, default='clip-to-whole', choices=['clip-to-whole', 'clip-to-clip', 'whole-to-whole'], help='Type of alignment to perform between the score and audio clips')
    args = parser.parse_args()

    evaluation_data_path = args.evaluation_data_path
    method = args.method
    alignment_type = args.alignment_type

    align_method = {
        'chroma': align_chroma,
        'spectra': align_spectra,
        'prettymidi': align_prettymidi
    }[method]

    metrics = run_baseline(evaluation_data_path, align_method, alignment_type)

    # Print the metrics
    print(f'Baseline Metrics for {alignment_type} {method} alignment:')
    for key, value in metrics.items():
        print(f'{key}: {value:.2f}')
    
    # create baseline_results directory if it doesn't exist
    os.makedirs('../baseline_results', exist_ok=True)

    # write the JSON string to a file in the "../results" directory
    metrics_json = json.dumps(metrics)
    with open(f'../baseline_results/{alignment_type}_{method}_metrics.json', 'w') as f:
        f.write(metrics_json)


if __name__ == '__main__':
    main()
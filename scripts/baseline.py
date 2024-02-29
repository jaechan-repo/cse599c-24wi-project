from fastdtw import fastdtw
import music21
import torch
import sys
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append('..')
from aligner.dataset import MaestroDataset


# Feature Extraction from MIDI
def extract_features(midi_file):
    # Convert MIDI file to a music21 stream
    midi_data = music21.converter.parse(midi_file)

    # Extract note pitches and durations as features
    features = [(note.pitch.midi, note.duration.quarterLength) for note in midi_data.notes]

    return features

# Dynamic Time Warping (DTW) Alignment using PyTorch
def dtw_alignment(features1, features2):
    # Convert features to numpy arrays
    array1 = np.array(features1)
    array2 = np.array(features2)

    # Compute DTW alignment using dtw-python
    distance, path = fastdtw(x=array1, y=array2, dist='euclidean')
    
    return path


# run baseline over test set
def run_baseline(evaluation_data_path: torch.utils.data.Dataset):

    # Load the test set
    # TODO: Replace MaestroDataset with the appropriate dataset class
    evaluation_dataset = MaestroDataset(evaluation_data_path, 'test')
    dataloader = DataLoader(dataset=evaluation_dataset)

    accuracies = []

    # Compute the DTW alignment for each test set
    loop = tqdm(dataloader)
    for item in tqdm(dataloader):

        reference_midi_file = item.reference_midi_file
        query_midi_file = item.query_midi_file

        # Extract features from the altered query MIDI file
        test_features = extract_features(query_midi_file)

        # Extract features from the reference MIDI file
        reference_features = extract_features(reference_midi_file)

        # Compute the DTW alignment
        path = dtw_alignment(test_features, reference_features)

        # calculate percentage of alignment path on diagonal,
        accuracy = sum([1 for (x, y) in path if x == y]) / len(path)
        accuracies.append(accuracy)

        loop.set_description(f'Accuracy: {accuracy}')
        loop.update(1)

    return accuracies


def main():
    # Parse arguments: evaluation_data_path
    evaluation_data_path = sys.argv[1]

    accuracies = run_baseline(evaluation_data_path)

    # Compute the average accuracy
    average_accuracy = sum(accuracies) / len(accuracies)

    print(f'Average accuracy: {average_accuracy}')


if __name__ == '__main__':
    main()
### Evaluation code for audio-score alignment

import torch
from your_model import YourModel
from your_dataset import YourDataset
import tqdm
import argparse
from ../utils/metrics import temporal_distance_vec, binary_accuracy_vec, monotonicity_vec, score_coverage_vec

def load_model(model_path):
    model = YourModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def evaluate(model, dataloader):
    """Evaluate the model on the evaluation dataset.
    
    Args:
        model (torch.nn.Module): Model to evaluate.
        dataloader (torch.utils.data.DataLoader): Evaluation dataset.
        
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
        audio_frames, score_events, Y, midi_event_timestamps = batch
        Y_pred = model(audio_frames, score_events)

        loss = compute_loss(Y_pred, Y)

        Y_pred_binary = torch.zeros(Y_pred.shape)
        Y_pred_binary[torch.arange(Y_pred.shape[0]), torch.argmax(Y_pred, dim=1)] = 1

        distance = temporal_distance_vec(Y_pred_binary, Y, midi_event_timestamps)
        accuracy = binary_accuracy_vec(Y_pred_binary, Y)
        monotonic = monotonicity_vec(Y_pred_binary, midi_event_timestamps)
        coverage = score_coverage_vec(Y_pred_binary)

        total_loss += torch.mean(loss)
        total_distance += torch.mean(distance)
        total_accuracy += torch.mean(accuracy)
        total_monotonicity += torch.mean(monotonic)
        total_coverage += torch.mean(coverage)

    return total_loss / len(dataloader), total_distance / len(dataloader), total_accuracy / len(dataloader), total_monotonicity / len(dataloader), total_coverage / len(dataloader)
        

def main():
    # Parse arguments: model_path, evaluation_data_path
    parser = argparse.ArgumentParser(description='Evaluation script')
    parser.add_argument('model_path', type=str, help='Path to the trained model')
    parser.add_argument('evaluation_data_path', type=str, help='Path to the evaluation data')
    args = parser.parse_args()

    model = load_model(args.model_path)
    evaluation_dataset = YourDataset(args.evaluation_data_path)
    dataloader = torch.utils.data.DataLoader(evaluation_dataset, batch_size=32)
    
    loss, distance, accuracy, monotonicity, coverage = evaluate(model, dataloader)
    print(f'Evaluation Metrics\nLoss: {loss:.2f}, Distance: {distance:.2f}, Accuracy: {accuracy:.2f}, Monotonicity: {monotonicity:.2f}, Coverage: {coverage:.2f}')

if __name__ == "__main__":
    main()
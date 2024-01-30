### Evaluation code for audio-score alignment

import torch
import tqdm
import argparse
from ../utils/metrics import temporal_distance_vec, binary_accuracy_vec, monotonicity_vec, score_coverage_vec

# TODO: Update the following imports with actual model, dataset, loss function, and decode function
from your_model import YourModel            
from your_dataset import YourDataset
from utils.loss import compute_loss
from utils.decode import decode_alignment


def load_model(model_path):
    model = YourModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def evaluate(model, dataloader, tau):
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

        # loss should use the non-decoded alignment matrix???
        loss = compute_loss(Y_pred, Y)

        Y_decoded = decode_alignment(Y_pred)

        #Y_pred_binary = torch.zeros_like(Y_pred)
        #Y_pred_binary[torch.arange(Y_pred.shape[0]), torch.argmax(Y_pred, dim=1)] = 1

        distance = temporal_distance_vec(Y_decoded, Y, midi_event_timestamps, tau)
        accuracy = binary_accuracy_vec(Y_decoded, Y, midi_event_timestamps, tau)
        monotonic = monotonicity_vec(Y_decoded, midi_event_timestamps)
        coverage = score_coverage_vec(Y_decoded)

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
    parser.add_argument('tau', type=float, help='Threshold for alignment distance. Default should be MIDI score event duration / 2')
    args = parser.parse_args()

    tau = args.tau
    model = load_model(args.model_path)
    evaluation_dataset = YourDataset(args.evaluation_data_path)
    dataloader = torch.utils.data.DataLoader(evaluation_dataset, batch_size=32)
    
    loss, distance, accuracy, monotonicity, coverage = evaluate(model, dataloader, tau)
    print(f'Evaluation Metrics with Tau={tau}\nLoss: {loss:.2f}, Distance: {distance:.2f}, Accuracy: {accuracy:.2f}, Monotonicity: {monotonicity:.2f}, Coverage: {coverage:.2f}')

if __name__ == "__main__":
    main()

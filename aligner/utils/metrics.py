import torch


def compute_loss(
        Y_pred: torch.Tensor, 
        Y: torch.Tensor, 
        midi_event_timestamps: torch.Tensor,
        reduction: str = 'none'
    ) -> float:

    """Compute the loss between the predicted cross-attention probability alignment matrix and the ground-truth matrices.
    Loss is composed of cross-entropy loss and temporal monotonicity constraint.

    Args:
        Y_pred (torch.Tensor): Predicted binary alignment matrix of shape (N, E, X), where E is the number of MIDI events in the score and X is the number of audio frames.
        Y (torch.Tensor): Ground truth binary alignment matrix of shape (E, X).
        midi_event_timestamps (torch.Tensor): Timestamps of MIDI events of shape (E,).
        reduction (str): Reduction method for the loss. Either \textbf{mean} or \textbf{none}.

    Returns:
        float or torch.Tensor: Loss between Y_pred and Y across the batch.
    """

    # Cross-entropy loss
    loss = torch.nn.functional.binary_cross_entropy(Y_pred, Y)

    # TODO: Monotonicity constraint @Jaechan.

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'none':
        return loss


def temporal_distance(
        Y_pred: torch.Tensor, 
        Y: torch.Tensor, 
        midi_event_timestamps: torch.Tensor, 
        tolerance: float = 0.
    ) -> float:
    """Compute the audio-frame-wise temporal alignment distance between the predicted and ground-truth binary alignment matrices.

    Args:
        Y_pred (torch.Tensor): Predicted binary alignment matrix of shape (E, X), where E is the number of MIDI events in the score and X is the number of audio frames.
        Y (torch.Tensor): Ground truth binary alignment matrix of shape (E, X).
        midi_event_timestamps (torch.Tensor): Timestamps of MIDI events of shape (E,).
        tolerance (float): Tolerance threshold for alignment distance.

    Returns:
        float: Average temporal alignment distance between Y_pred and Y.
    """

    pred_indices = torch.argmax(Y_pred, dim=0)
    true_indices = torch.argmax(Y, dim=0)

    pred_timestamps = midi_event_timestamps[pred_indices]
    true_timestamps = midi_event_timestamps[true_indices]

    L1_distances = torch.abs(pred_timestamps - true_timestamps) - tolerance
    threshold_distances = L1_distances * (L1_distances > 0).float()

    return torch.mean(threshold_distances)

'''
def temporal_distance_vec(
        Y_pred: torch.Tensor, 
        Y: torch.Tensor,
        midi_event_timestamps: torch.Tensor, 
        tolerance: float,
        reduction: str = 'none'
    ) -> float:
    """Compute the audio-frame-wise temporal alignment distance between the predicted and ground-truth binary alignment matrices.

    Args:
        Y_pred (torch.Tensor): Predicted binary alignment matrices of shape (N, E, X), where N is the batch size, E is the number of MIDI events in the score, and X is the number of audio frames.
        Y (torch.Tensor): Ground truth binary alignment matrices of shape (N, E, X).
        midi_event_timestamps (torch.Tensor): Timestamps of MIDI events of shape (E,).
        tolerance (float): Tolerance threshold for alignment distance.
        reduction (str): Reduction method for the temporal distance. Either \textbf{mean} or \textbf{none}.

    Returns:
        float or torch.Tensor: Average temporal alignment distance between Y_pred and Y across the batch.
    """
    pred_indices = torch.argmax(Y_pred, dim=1)
    true_indices = torch.argmax(Y, dim=1)

    # Expand midi_event_timestamps to match the batch size
    expanded_timestamps = midi_event_timestamps.unsqueeze(0).expand(Y_pred.shape[0], -1)

    pred_timestamps = torch.gather(expanded_timestamps, 1, pred_indices)
    true_timestamps = torch.gather(expanded_timestamps, 1, true_indices)

    L1_distances = torch.abs(pred_timestamps - true_timestamps) - tolerance
    threshold_distances = L1_distances * (L1_distances > 0).float()
    
    avg_distance_per_sample = torch.mean(threshold_distances, dim=1)
    
    if reduction == 'mean':
        return torch.mean(avg_distance_per_sample)
    elif reduction == 'none':
        return avg_distance_per_sample
'''

def binary_accuracy(
        Y_pred: torch.Tensor, 
        Y: torch.Tensor, 
        midi_event_timestamps: torch.Tensor, 
        tolerance: float
    ) -> float:
    """Compute the audio-frame-wise binary alignment accuracy between the predicted and ground-truth binary alignment matrices.

    Args:
        Y_pred (torch.Tensor): Predicted binary alignment matrix of shape (E, X), where E is the number of MIDI events in the score and X is the number of audio frames.
        Y (torch.Tensor): Ground truth binary alignment matrix of shape (E, X).
        midi_event_timestamps (torch.Tensor): Timestamps of MIDI events of shape (E,).
        tolerance (float): Tolerance threshold for alignment distance.

    Returns:
        float: Average binary accuracy of alignment predictions.
    """

    pred_indices = torch.argmax(Y_pred, dim=0)
    true_indices = torch.argmax(Y, dim=0)

    pred_timestamps = midi_event_timestamps[pred_indices]
    true_timestamps = midi_event_timestamps[true_indices]

    binary_accuracies = (torch.abs(pred_timestamps - true_timestamps) <= tolerance).float()
    return torch.mean(binary_accuracies)

'''
def binary_accuracy_vec(
        Y_pred: torch.Tensor, 
        Y: torch.Tensor, 
        midi_event_timestamps: torch.Tensor, 
        tolerance: float,
        reduction: str = 'none'
    ) -> float:
    """Compute the audio-frame-wise binary alignment accuracy between the predicted and ground-truth binary alignment matrices.

    Args:
        Y_pred (torch.Tensor): Predicted binary alignment matrices of shape (N, E, X), where N is the batch size, E is the number of MIDI events in the score, and X is the number of audio frames.
        Y (torch.Tensor): Ground truth binary alignment matrix of shape (N, E, X).
        midi_event_timestamps (torch.Tensor): Timestamps of MIDI events of shape (E,).
        tolerance (float): Tolerance threshold for alignment distance.
        reduction (str): Reduction method for the binary accuracy. Either \textbf{mean} or \textbf{none}.

    Returns:
        float or torch.Tensor: Average binary accuracy of alignment predictions across the batch.
    """

    pred_indices = torch.argmax(Y_pred, dim=1)
    true_indices = torch.argmax(Y, dim=1)

    # Expand midi_event_timestamps to match the batch size
    expanded_timestamps = midi_event_timestamps.unsqueeze(0).expand(Y_pred.shape[0], -1)

    pred_timestamps = torch.gather(expanded_timestamps, 1, pred_indices)
    true_timestamps = torch.gather(expanded_timestamps, 1, true_indices)

    binary_accuracies = (torch.abs(pred_timestamps - true_timestamps) <= tolerance).float()

    mean_acc_per_sample = torch.mean(binary_accuracies, dim=1)

    if reduction == 'mean':
        return torch.mean(mean_acc_per_sample)
    elif reduction == 'none':
        return mean_acc_per_sample
'''

def monotonicity(Y_pred: torch.Tensor) -> bool:
    """Compute the monotonicity of the predicted alignment matrix.

    Args:
        Y_pred (torch.Tensor): Predicted binary alignment matrix of shape (E, X), where E is the number of MIDI events in the score and X is the number of audio frames.

    Returns:
        float: Whether or not the predicted alignment adheres to monotonicity.
    """

    pred_indices = torch.argmax(Y_pred, dim=0)
    return torch.all(pred_indices[1:] >= pred_indices[:-1]).float()

'''
def monotonicity_vec(
        Y_pred: torch.Tensor, 
        reduction: str = 'none'
    ) -> float:
    """Compute the monotonicity of the predicted alignment matrices.

    Args:
        Y_pred (torch.Tensor): Predicted binary alignment matrix of shape (N, E, X), N is the batch size, E is the number of MIDI events in the score, and X is the number of audio frames.
        reduction (str): Reduction method for the monotonicity. Either \textbf{mean} or \textbf{none}.

    Returns:
        float or torch.Tensor: Whether or not the predicted alignment adheres to monotonicity across the batch."""
    pred_indices = torch.argmax(Y_pred, dim=1)

    # Check if timestamps are monotonically non-decreasing along each frame
    diffs = pred_indices[:, 1:] - pred_indices[:, :-1]
    monotonicity_per_sample = torch.all(diffs >= 0, dim=1).float()

    if reduction == 'mean':
        return torch.mean(monotonicity_per_sample)
    elif reduction == 'none':
        return monotonicity_per_sample
'''

def score_coverage(Y_pred: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute the score-wise alignment coverage of the predicted alignment matrix.
    This metric gives the percentage of ground-truth MIDI events that are aligned to at least one audio frame.

    Args:
        Y_pred (torch.Tensor): Predicted binary alignment matrix of shape (E, X), where E is the number of MIDI events in the score and X is the number of audio frames.
        Y (torch.Tensor): Ground truth binary alignment matrix of shape (E, X).

    Returns:
        float: Score-wise alignment coverage.
    """

    pred_indices = torch.argmax(Y_pred, dim=0)
    true_indices = torch.argmax(Y, dim=0)

    # get percentage of unique true indices that are covered by predicted indices
    events_covered = (torch.unique(pred_indices).unsqueeze(0) == torch.unique(true_indices).unsqueeze(1)).any(dim=0).float().mean()    
    return events_covered

'''
def score_coverage_vec(Y_pred: torch.Tensor, reduction: str = 'none') -> float:
    """Compute the score-wise alignment coverage of the predicted alignment matrix.

    Args:
        Y_pred (torch.Tensor): Predicted binary alignment matrices of shape (N, E, X), where N is the batch size, E is the number of MIDI events in the score, and X is the number of audio frames.
        reduction (str): Reduction method for the score coverage. Either \textbf{mean} or \textbf{none}.

    Returns:   
        float or torch.Tensor: Score-wise alignment coverage across the batch."""
        
    events_covered = (Y_pred.sum(dim=2) > 0).float()
    percent_covered_per_sample = torch.mean(events_covered, dim=1)

    if reduction == 'mean':
        return torch.mean(percent_covered_per_sample)
    elif reduction == 'none':
        return percent_covered_per_sample
'''
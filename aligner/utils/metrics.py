import torch


def compute_loss(
        Y_pred: torch.Tensor, 
        Y: torch.Tensor, 
        midi_event_timestamps: torch.Tensor
    ) -> torch.Tensor:

    """Compute the loss between the predicted cross-attention probability alignment matrix and the ground-truth matrices.
    Loss is composed of cross-entropy loss and temporal monotonicity constraint.

    Args:
        Y_pred (torch.Tensor): Predicted binary alignment matrix of shape (N, E, X), where E is the number of MIDI events in the score and X is the number of audio frames.
        Y (torch.Tensor): Ground truth binary alignment matrix of shape (E, X).
        midi_event_timestamps (torch.Tensor): Timestamps of MIDI events of shape (E,).

    Returns:
        float: Average loss between Y_pred and Y.
    """

    # Cross-entropy loss
    loss = torch.nn.functional.binary_cross_entropy(Y_pred, Y)

    # TODO: Monotonicity constraint @Jaechan.

    return torch.mean(loss)


def temporal_distance(
        Y_pred: torch.Tensor, 
        Y: torch.Tensor, 
        midi_event_timestamps: torch.Tensor, 
        tolerance: float = 0.
    ) -> torch.Tensor:
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


def temporal_distance_vec(
        Y_pred: torch.Tensor, 
        Y: torch.Tensor,
        midi_event_timestamps: torch.Tensor, 
        tolerance: float
    ) -> torch.Tensor:
    """Compute the audio-frame-wise temporal alignment distance between the predicted and ground-truth binary alignment matrices.

    Args:
        Y_pred (torch.Tensor): Predicted binary alignment matrices of shape (N, E, X), where N is the batch size, E is the number of MIDI events in the score, and X is the number of audio frames.
        Y (torch.Tensor): Ground truth binary alignment matrices of shape (N, E, X).
        midi_event_timestamps (torch.Tensor): Timestamps of MIDI events of shape (E,).
        tolerance (float): Tolerance threshold for alignment distance.

    Returns:
        torch.Tensor: Average temporal alignment distance between Y_pred and Y per sample.
    """
    pred_indices = torch.argmax(Y_pred, dim=1)
    true_indices = torch.argmax(Y, dim=1)

    # Expand midi_event_timestamps to match the batch size
    expanded_timestamps = midi_event_timestamps.unsqueeze(0).expand(Y_pred.shape[0], -1)

    pred_timestamps = torch.gather(expanded_timestamps, 1, pred_indices)
    true_timestamps = torch.gather(expanded_timestamps, 1, true_indices)

    L1_distances = torch.abs(pred_timestamps - true_timestamps) - tolerance
    threshold_distances = L1_distances * (L1_distances > 0).float()
    
    return torch.mean(threshold_distances, dim=1)


def binary_accuracy(
        Y_pred: torch.Tensor, 
        Y: torch.Tensor, 
        midi_event_timestamps: torch.Tensor, 
        tolerance: float
    ) -> torch.Tensor:
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


def binary_accuracy_vec(
        Y_pred: torch.Tensor, 
        Y: torch.Tensor, 
        midi_event_timestamps: torch.Tensor, 
        tolerance: float
    ) -> torch.Tensor:
    """Compute the audio-frame-wise binary alignment accuracy between the predicted and ground-truth binary alignment matrices.

    Args:
        Y_pred (torch.Tensor): Predicted binary alignment matrices of shape (N, E, X), where N is the batch size, E is the number of MIDI events in the score, and X is the number of audio frames.
        Y (torch.Tensor): Ground truth binary alignment matrix of shape (N, E, X).
        midi_event_timestamps (torch.Tensor): Timestamps of MIDI events of shape (E,).
        tolerance (float): Tolerance threshold for alignment distance.

    Returns:
        torch.Tensor: Average binary accuray of alignment predictions per sample.
    """

    pred_indices = torch.argmax(Y_pred, dim=1)
    true_indices = torch.argmax(Y, dim=1)

    # Expand midi_event_timestamps to match the batch size
    expanded_timestamps = midi_event_timestamps.unsqueeze(0).expand(Y_pred.shape[0], -1)

    pred_timestamps = torch.gather(expanded_timestamps, 1, pred_indices)
    true_timestamps = torch.gather(expanded_timestamps, 1, true_indices)

    binary_accuracies = (torch.abs(pred_timestamps - true_timestamps) <= tolerance).float()

    return torch.mean(binary_accuracies, dim=1)


def monotonicity(
        Y_pred: torch.Tensor, 
        midi_event_timestamps: torch.Tensor
    ) -> bool:
    """Compute the monotonicity of the predicted alignment matrix.

    Args:
        Y_pred (torch.Tensor): Predicted binary alignment matrix of shape (E, X), where E is the number of MIDI events in the score and X is the number of audio frames.
        midi_event_timestamps (torch.Tensor): Timestamps of MIDI events of shape (E,).

    Returns:
        bool: Whether or not the predicted alignment adheres to monotonicity.
    """

    pred_indices = torch.argmax(Y_pred, dim=0)
    pred_timestamps = midi_event_timestamps[pred_indices]
    return torch.all(pred_timestamps[1:] >= pred_timestamps[:-1]).bool()


def monotonicity_vec(
        Y_pred: torch.Tensor, 
        midi_event_timestamps: torch.Tensor
    ) -> torch.Tensor:
    """Compute the monotonicity of the predicted alignment matrices.

    Args:
        Y_pred (torch.Tensor): Predicted binary alignment matrix of shape (N, E, X), N is the batch size, E is the number of MIDI events in the score, and X is the number of audio frames.
        midi_event_timestamps (torch.Tensor): Timestamps of MIDI events of shape (E,).

    Returns:
        Tensor: Tensor of bool representing whether or not each predicted alignment matrix adheres to monotonicity.
    """
    pred_indices = torch.argmax(Y_pred, dim=1)

    # Expand midi_event_timestamps to match the batch size
    expanded_timestamps = midi_event_timestamps.unsqueeze(0).expand(Y_pred.shape[0], -1)

    pred_timestamps = torch.gather(expanded_timestamps, 1, pred_indices)

    # Check if timestamps are monotonically non-decreasing along each frame
    diffs = pred_timestamps[:, 1:] - pred_timestamps[:, :-1]
    monotonic = torch.all(diffs >= 0, dim=1).bool()
    return monotonic


def score_coverage(Y_pred: torch.Tensor) -> float:
    """Compute the score-wise alignment coverage of the predicted alignment matrix.

    Args:
        Y_pred (torch.Tensor): Predicted binary alignment matrix of shape (E, X), where E is the number of MIDI events in the score and X is the number of audio frames.

    Returns:
        float: Score-wise alignment coverage.
    """

    events_covered = (Y_pred.sum(dim=1) > 0).float()

    return torch.mean(events_covered)


def score_coverage_vec(Y_pred: torch.Tensor) -> torch.Tensor:
    """Compute the score-wise alignment coverage of the predicted alignment matrix.

    Args:
        Y_pred (torch.Tensor): Predicted binary alignment matrices of shape (N, E, X), where N is the batch size, E is the number of MIDI events in the score, and X is the number of audio frames.

    Returns:
        torch.Tensor: Tensor of floats indicating score-wise alignment coverage per sample.
    """
        
    events_covered = (Y_pred.sum(dim=2) > 0).float()
    return torch.mean(events_covered, dim=1)

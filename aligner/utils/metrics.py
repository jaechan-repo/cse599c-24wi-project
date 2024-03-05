import torch
from constants import AUDIO_RESOLUTION

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
        reduction (str): Reduction method for the loss. Either mean or none.

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


def temporal_distance_v2(
        Y_pred: torch.Tensor, 
        Y: torch.Tensor, 
    ) -> float:
    """Compute the audio-frame-wise temporal alignment distance between the predicted and ground-truth binary alignment matrices. This version allows an audio frame to be aligned to multiple MIDI events.

    Args:
        Y_pred (torch.Tensor)
            Predicted binary alignment matrix of shape (E, X), where E is the number of MIDI events in the score and X is the number of audio frames. 
            Y_pred is assumed to be decoded with dtw, such that each score event is aligned to at least one audio frame, with leading and trailing parts of the score not present in the audio being aligned to the first and last audio frames, respectively.
        Y (torch.Tensor):
            Ground truth binary alignment matrix of shape (E, X).

    Returns:
        float: Average element-wise temporal alignment distance between Y_pred and Y (in ms, according to audio-resolution).
    """

    # get the subscore of the gold alignment matrix with 1s
    Y_align = Y[Y.sum(dim=1).nonzero().squeeze()]
    Y_pred_align = Y_pred[Y.sum(dim=1).nonzero().squeeze()]

    # get audio frame alignment for each MIDI event in the gold subscore
    true_audio_indices = torch.argmax(Y_align, dim=1)
    pred_audio_indices = torch.argmax(Y_pred_align, dim=1)

    L1_distances = torch.abs(pred_audio_indices - true_audio_indices).float()

    return torch.mean(L1_distances) * AUDIO_RESOLUTION


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

def binary_accuracy_v2(
        Y_pred: torch.Tensor, 
        Y: torch.Tensor, 
    ) -> float:
    """Compute the audio-frame-wise binary alignment accuracy between the predicted and ground-truth binary alignment matrices.

    Args:
        Y_pred (torch.Tensor)
            Predicted binary alignment matrix of shape (E, X), where E is the number of MIDI events in the score and X is the number of audio frames. 
            Y_pred is assumed to be decoded with dtw, such that each score event is aligned to at least one audio frame, with leading and trailing parts of the score not present in the audio being aligned to the first and last audio frames, respectively.
        Y (torch.Tensor):
            Ground truth binary alignment matrix of shape (E, X).

    Returns:
        float: Average binary accuracy of alignment predictions.
    """

    # get the subscore of the gold alignment matrix with 1s
    Y_align = Y[Y.sum(dim=1).nonzero().squeeze()]
    Y_pred_align = Y_pred[Y.sum(dim=1).nonzero().squeeze()]

    # get audio frame alignment for each MIDI event in the gold subscore
    true_audio_indices = torch.argmax(Y_align, dim=1)
    pred_audio_indices = torch.argmax(Y_pred_align, dim=1)

    binary_accuracies = (pred_audio_indices == true_audio_indices).float()
    return torch.mean(binary_accuracies)


def monotonicity(Y_pred: torch.Tensor) -> bool:
    """Compute the monotonicity of the predicted alignment matrix.

    Args:
        Y_pred (torch.Tensor): Predicted binary alignment matrix of shape (E, X), where E is the number of MIDI events in the score and X is the number of audio frames.

    Returns:
        float: Whether or not the predicted alignment adheres to monotonicity.
    """

    pred_indices = torch.argmax(Y_pred, dim=0)
    return torch.all(pred_indices[1:] >= pred_indices[:-1]).float()


def monotonicity_v2(Y_pred: torch.Tensor) -> float:
    """Compute the motoricity of the predicted alignment matrix.

    Args:
        Y_pred (torch.Tensor): Predicted binary alignment matrix of shape (E, X), where E is the number of MIDI events in the score and X is the number of audio frames.

    Returns:
        float: Motoricity of the predicted alignment.
    """

    pred_audio_indices = torch.argmax(Y_pred, dim=1)
    return torch.all((pred_audio_indices[1:] >= pred_audio_indices[:-1]).float())


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

def coverage(Y_pred: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute alignment coverage as the intersection over union of the predicted and gold score ranges.

    Args:
        Y_pred (torch.Tensor): Predicted binary alignment matrix of shape (E, X), where E is the number of MIDI events in the score and X is the number of audio frames.
        Y (torch.Tensor): Ground truth binary alignment matrix of shape (E, X).

    Returns:
        float: Alignment coverage.
    """

    # get the indices of the subsection of the gold alignment matrix with 1s
    true_range = (Y.sum(dim=1) > 0).squeeze()
    pred_range = (Y_pred.sum(dim=1) > 0).squeeze()

    # get intersection and union of the two ranges
    intersection = (true_range & pred_range).float().sum()
    union = (true_range | pred_range).float().sum()

    return intersection / union
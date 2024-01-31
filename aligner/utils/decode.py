import torch

def max_decode(Y_pred: torch.Tensor) -> torch.Tensor:
    """Decode the predicted alignment matrix by taking the maximum value for each audio frame.

    Args:
        Y_pred (torch.Tensor): Predicted alignment matrix of shape (N, E, X), where N is the batch size, E is the number of MIDI events in the score and X is the number of audio frames.

    Returns:
        torch.Tensor: Decoded alignment matrix of shape (N, E, X).
    """
    N, E, X = Y_pred.shape

    max_idxs = torch.argmax(Y_pred, dim=1)
    Y_pred_binary = torch.zeros_like(Y_pred)
    Y_pred_binary[torch.arange(N).unsqueeze(-1), max_idxs, torch.arange(X)] = 1

    return Y_pred_binary


def DTW(Y_pred: torch.Tensor) -> torch.Tensor:
    """Decode the predicted alignment matrix using dynamic time warping.
    
    Args:
        Y_pred (torch.Tensor): Predicted alignment matrix of shape (N, E, X), where N is the batch size, E is the number of MIDI events in the score and X is the number of audio frames.
    
    Returns:
        torch.Tensor: Decoded alignment matrix of shape (N, E, X).
    """
    return NotImplementedError
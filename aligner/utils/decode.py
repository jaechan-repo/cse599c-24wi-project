import torch

def max_decode(Y_pred: torch.Tensor) -> torch.Tensor:
    """Decode the predicted alignment matrix by taking the maximum value for each audio frame.

    Args:
        Y_pred (torch.Tensor): Predicted alignment matrix of shape (N, E, X), where N is the batch size, E is the number of MIDI events in the score and X is the number of audio frames.

    Returns:
        torch.Tensor: Decoded alignment matrix of shape (N, E, X).
    """
    if Y_pred.dim() == 2:
        Y_pred = Y_pred.unsqueeze(0)

    N, E, X = Y_pred.shape

    max_idxs = torch.argmax(Y_pred, dim=-2)
    Y_pred_binary = torch.zeros_like(Y_pred)

    Y_pred_binary[torch.arange(N).unsqueeze(-1), max_idxs, torch.arange(X)] = 1

    return Y_pred_binary.squeeze(0)


def DTW(Y_pred: torch.Tensor) -> torch.Tensor:
    """Decode the predicted alignment matrix using a modified version of dynamic time warping.
    These modifications include:
    - Using a negative cost for alignment probabilities
    - Preventing downward motion in the warping path, such that each audio frame is aligned to only one score event
    - Allowing the warping path to start and end anywhere in the score, since the audio clip only contains a segment of the score
    
    Args:
        Y_pred (torch.Tensor): Predicted probability alignment matrix of shape (N, E, X), where N is the batch size, E is the number of MIDI events in the score and X is the number of audio frames. Each y_{n, e, x} is a probability value in [0, 1] representing the likelihood of audio frame x being aligned to MIDI event e being in sample n, such that \sum_{e=1}^{E} y_{n, e, x} = 1 for all n, x.
    
    Returns:
        torch.Tensor: Decoded alignment matrix of shape (N, E, X).
    """
    
    if Y_pred.dim() == 2:
        Y_pred = Y_pred.unsqueeze(0)

    N, E, X = Y_pred.shape
    device = Y_pred.device
    dtw_path = torch.zeros(N, E, X, device=device)
    Ds = torch.zeros(N, E, X, device=device)

    for n in range(N):
        # Initialize cumulative distance matrix
        D = torch.full((E + 1, X + 1), float('inf'), device=device)
        D[:, 0] = 0  # Allows starting at any row

        for i in range(1, E + 1):
            for j in range(1, X + 1):
                cost = -Y_pred[n, i - 1, j - 1]
                D[i, j] = cost + min(D[i - 1, j - 1], D[i, j - 1])

        Ds[n] = D[1:, 1:]
        # Find the minimum value in the last column
        min_value, min_row = torch.min(D[:, X], 0)

        # Backtrack from the position of minimum value
        i, j = min_row.item(), X
        while i > 0 and j > 0:
            dtw_path[n, i - 1, j - 1] = 1
            if i > 1 and D[i - 1, j - 1] <= D[i, j - 1]:
                i -= 1
                j -= 1
            else:
                j -= 1

    return dtw_path.squeeze(0)    


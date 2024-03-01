import torch
import torch.nn.functional as F
from torch import Tensor, LongTensor


def max_decode(Y_hat: Tensor
               ) -> LongTensor:
    """Decode the predicted alignment matrix by taking the maximum value for each audio frame.
    """
    return F.one_hot(Y_hat.argmax(dim=-1),
                     num_classes=Y_hat.shape[-1])


def dtw_decode(Y_hat: Tensor) -> LongTensor:
    unbatched = len(Y_hat.shape) != 3
    if unbatched:
        Y_hat = Y_hat.unsqueeze(0)

    prev = torch.empty(Y_hat.shape, dtype=torch.bool)
    DP = Y_hat.clone()
    batch_size, n_frames, n_events = Y_hat.shape

    for i in range(n_frames - 1):
        candidates = torch.stack([
            DP[:, i, :],
            F.pad(DP[:, i, :-1], (1, 0), value=float('-inf'))],
            dim=-1
        )
        values, indices = candidates.max(dim=-1)
        DP[:, i + 1, :] += values
        # from diagonal if 1, from sideway if 0
        prev[:, i + 1, :] = indices.bool()

    # Get event index for each frame
    path = torch.empty((batch_size, n_frames), dtype=torch.long)
    path[:, -1] = torch.argmax(DP[:, -1, :].clone(), dim=-1)
    del DP

    for i in range(n_frames - 1, 0, -1):
        prev_i = prev[torch.arange(batch_size), i, path[:, i]].int()    # (batch_size,)
        path[:, i - 1] = path[:, i] - prev_i

    Y_dtw = F.one_hot(path, num_classes=n_events)
    del path

    if unbatched:
        Y_dtw = Y_dtw.squeeze(0)
    return Y_dtw


def softmax(Y_hat: Tensor) -> Tensor:
    return Y_hat.softmax(dim=-1)

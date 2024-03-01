import torch
from torch import Tensor, LongTensor
from typing import Literal, Dict, Tuple
import torch.nn.functional as F
from .decode import dtw_decode


def cross_entropy(Y_hat: Tensor,
                  Y: Tensor,
                  reduction: Literal['mean', 'none'] = 'mean',
                  complement: bool = False,
                  temperature: Tensor | None = None,
                  eps=1e-9
                  ) -> Tensor:
    """Cross entropy loss.

    Args:
        Y_hat (Tensor): UNNORMALIZED input. Size: (bsz, n_frames, n_events)
        Y (Tensor): Target class probabilities. Size: (bsz, n_frames, n_events)
        weight (Tensor): Weight. Size: (bsz, n_frames)
    """
    if len(Y.shape) != 3:
        Y_hat, Y, weight = Y_hat.unsqueeze(0), Y.unsqueeze(0), weight.unsqueeze(0)
        reduction = 'mean'

    if temperature is not None:
        Y_hat /= temperature.clamp(min=eps)
    logits = Y_hat.softmax(dim=-1)   # (bsz, n_frames, n_events)

    if complement:
        logits = 1 - logits     # no longer a probability distribution

    nll = -torch.log(logits.clamp(min=eps))  # (bsz, n_frames, n_events)
    loss = torch.sum(Y * nll, dim=-1)   # (bsz, n_frames)
    assert not loss.isnan().any()

    if reduction == 'none':
        return loss         # (bsz, n_frames)
    elif reduction == 'mean':
        return loss.mean()  # ()
    raise ValueError


def rmse_loss(Y_hat: Tensor,
              Y: Tensor,
              eps: float = 1e-9,
              reduction: Literal['none', 'mean'] = 'mean'
              ) -> Tensor | float:
    """Computes the RMSE loss, considering each event index as a unit of distance
    and the softmax output of `Y_hat` (i.e. probabilities) as weights.

    Args:
        Y_hat (Tensor): UNNORMALIZED input. Size: (bsz, n_frames, n_events)
        Y (Tensor): Target class probabilities. Size: (bsz, n_frames, n_events)
    """
    if len(Y.shape) == 2:
        Y_hat, Y = Y_hat.unsqueeze(0), Y.unsqueeze(0)
        reduction = 'mean'

    y = Y.argmax(dim=-1).unsqueeze(-1).float()
    x = torch.arange(Y.shape[-1]).view(1,1,-1).type_as(y)

    dist = (x - y) ** 2
    logits = Y_hat.softmax(dim=-1)
    loss = torch.sum(logits * dist, dim=-1)     # (bsz, n_frames,)
    loss = torch.sqrt(loss + eps).mean(dim=-1)  # (bsz,)
    assert not loss.isnan().any()

    if reduction == 'none':
        return loss     # (bsz,)
    elif reduction == 'mean':
        return loss.mean()  # ()
    raise ValueError


def emd_loss(Y_hat: Tensor,
             Y: Tensor,
             reduction: Literal['mean'] = 'mean',
             temperature: Tensor | None = None,
             eps=1e-9):
    if len(Y.shape) == 2:
        Y_hat, Y = Y_hat.unsqueeze(0), Y.unsqueeze(0)
        reduction = 'mean'

    if temperature is not None:
        Y_hat /= temperature.clamp(min=eps)
    logits = Y_hat.softmax(dim=-1)
    cdf_pred = logits.cumsum(dim=-1).clamp(min=0, max=1)
    cdf_gold = Y.cumsum(dim=-1)
    # loss = (cdf_pred - cdf_gold) ** 2       # (bsz, n_frames, n_events)
    loss = F.binary_cross_entropy(cdf_pred, cdf_gold, reduction='none')

    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    raise ValueError


def structured_perceptron_loss(Y_hat: Tensor,
                               Y: Tensor,
                               reduction: Literal['none', 'mean'] = 'mean',
                               temperature: Tensor | None = None
                               ) -> Tensor | float:
    if len(Y.shape) == 2:
        Y_hat, Y = Y_hat.unsqueeze(0), Y.unsqueeze(0)
        reduction = 'mean'

    with torch.no_grad():
        Y_dtw = dtw_decode(Y_hat).type_as(Y)
        mask = Y.argmax(dim=-1) != Y_dtw.argmax(dim=-1)

    nll_gold = cross_entropy(Y_hat, Y, reduction='none', temperature=temperature)   # (bsz, n_frames)
    nll_dtw = cross_entropy(Y_hat, Y_dtw, reduction='none', temperature=temperature)
    loss = 2 * nll_gold - mask.float() * nll_dtw
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    raise ValueError


def compute_metrics(Y_hat: Tensor,
                    Y: Tensor,
                    n_events: LongTensor | None = None,
                    ) -> Tuple[float, Dict[str, Tensor]]:
    # temperature = 3 / torch.log(n_events.float() + 1).view(-1, 1, 1)
    temperature = None
    metrics = {
        'cross_entropy': cross_entropy(Y_hat, Y, temperature=temperature),
        'structured_perceptron_loss': structured_perceptron_loss(Y_hat, Y, temperature=temperature),
        'emd_loss': emd_loss(Y_hat, Y, temperature=temperature),
        'monotonicity': monotonicity(Y_hat)
    }
    metrics['custom_loss'] = (
        metrics['structured_perceptron_loss'] + 15 * metrics['emd_loss']
    )
    return metrics


def temporal_distance(Y_hat: LongTensor,
                      Y: LongTensor,
                      event_timestamps: Tensor, 
                      tolerance: float,
                      reduction: Literal['mean', 'none'] = 'mean'
                      ) -> Tensor | float:
    if len(Y.shape) == 2:
        Y_hat, Y, event_timestamps \
            = Y_hat.unsqueeze(0), Y.unsqueeze(0), event_timestamps.unsqueeze(0)
        reduction = 'mean'
    
    Y_hat, Y = Y_hat.float(), Y.float()

    pred_indices = torch.argmax(Y_hat, dim=-1)  # (batch_size, n_frames)
    true_indices = torch.argmax(Y, dim=-1)      # (batch_size, n_frames)

    pred_timestamps = torch.gather(event_timestamps, 1, pred_indices)
    true_timestamps = torch.gather(event_timestamps, 1, true_indices)

    L1_distances = torch.abs(pred_timestamps - true_timestamps) - tolerance
    thr_dist = L1_distances * (L1_distances > 0).float()

    avg_dist = torch.mean(thr_dist, dim=1)

    if reduction == 'mean':
        return torch.mean(avg_dist)
    elif reduction == 'none':
        return avg_dist
    raise ValueError


def binary_accuracy(Y_hat: LongTensor, 
                    Y: LongTensor, 
                    event_timestamps: Tensor, 
                    tolerance: float,
                    reduction: str = 'mean'
                    ) -> Tensor | float:
    if len(Y.shape) != 3:
        Y_hat, Y, event_timestamps \
            = Y_hat.unsqueeze(0), Y.unsqueeze(0), event_timestamps.unsqueeze(0)
        reduction = 'mean'

    Y_hat, Y = Y_hat.float(), Y.float()

    pred_indices = torch.argmax(Y_hat, dim=-1)
    true_indices = torch.argmax(Y, dim=-1)

    pred_timestamps = torch.gather(event_timestamps, 1, pred_indices)
    true_timestamps = torch.gather(event_timestamps, 1, true_indices)

    binary_acc = (torch.abs(pred_timestamps - true_timestamps) <= tolerance).float()
    mean_acc = torch.mean(binary_acc, dim=1)

    if reduction == 'mean':
        return torch.mean(mean_acc)
    elif reduction == 'none':
        return mean_acc
    raise ValueError


def monotonicity(Y_hat: Tensor,
                 reduction: Literal['mean', 'none'] = 'mean'
                 ) -> Tensor | bool:
    """Compute the monotonicity of the predicted alignment matrix.
    """
    if len(Y_hat.shape) != 3:
        Y_hat = Y_hat.unsqueeze(0)
        reduction = 'mean'

    pred_indices = torch.argmax(Y_hat, dim=-1)
    diffs = pred_indices[:, 1:] - pred_indices[:, :-1]
    mono = torch.all(diffs >= 0, dim=-1)

    if reduction == 'mean':
        return mono.float().mean()
    if reduction == 'none':
        return mono
    raise ValueError


def score_coverage_ratio_unbatched(Y_hat: Tensor, Y: Tensor) -> float:
    assert len(Y.shape) != 3, "Batched input detected."
    n_events_covered_hat = (Y_hat.sum(dim=0) > 0).float().sum()
    n_events_covered = (Y.sum(dim=0) > 0).float().sum()
    return n_events_covered_hat / n_events_covered

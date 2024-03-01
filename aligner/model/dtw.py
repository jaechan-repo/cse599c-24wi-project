import torch
import torch.nn as nn
from torch import Tensor
from ..utils.constants import *
import torch.nn.functional as F


def neg_inf_to_zero(input: Tensor) -> Tensor:
    return torch.where(input == float('-inf'),
                       torch.zeros_like(input),
                       input)


class HardDTW(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        out = input.clone()
        _, n_frames, _ = input.shape

        for i in range(n_frames - 1):
            candidates = torch.stack([
                out[:, i, :],
                F.pad(out[:, i, :-1], (1, 0), value=float('-inf'))],
                dim=-1
            )
            values, _ = candidates.max(dim=-1)
            out[:, i + 1, :] += values

        assert not out.isnan().any()
        return out


class BidirectionalHardDTW(nn.Module):

    def __init__(self):
        super().__init__()
        self.HardDTW = HardDTW()

    def forward(self, input: Tensor) -> Tensor:
        prefix = self.HardDTW(input)
        suffix = self.HardDTW(input.flip(dims=[1, 2])).flip(dims=[1, 2])
        out = prefix + suffix - neg_inf_to_zero(input)
        assert not out.isnan().any()
        return out


class SoftTruemax(nn.Module):

    def __init__(self, temperature=1.0):
        super().__init__()
        assert temperature <= 1.0
        self.temperature = temperature

    def forward(self, input: Tensor, dim: int) -> Tensor:
        contains_inf = (input == float('-inf')).any(dim=dim)
        input_not_inf = torch.where(contains_inf.unsqueeze(dim),
                                    torch.zeros_like(input),
                                    input)
        logits = torch.softmax(input_not_inf, dim=dim)
        soft_maximum = torch.sum(logits * input_not_inf, dim=dim)
        hard_maximum = input.max(dim=dim)[0]
        out = torch.where(contains_inf, hard_maximum, soft_maximum)
        assert not out.isnan().any()
        return out


class SoftDTW(nn.Module):

    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.SoftTruemax = SoftTruemax()


    def forward(self, input: Tensor
                ) -> Tensor:
        out = input.clone()
        _, n_frames, _ = out.shape

        for i in range(n_frames - 1):
            candidates = torch.stack([
                out[:, i, :],
                F.pad(out[:, i, :-1], (1, 0), value=float('-inf'))],
                dim=-1
            )
            values = self.SoftTruemax(candidates, dim=-1)
            out[:, i + 1, :] += values
        
        assert not out.isnan().any()
        return out


class BidirectionalSoftDTW(nn.Module):

    def __init__(self, temperature=1.0):
        super().__init__()
        self.SoftDTW = SoftDTW(temperature)

    def forward(self, input: Tensor) -> Tensor:
        prefix = self.SoftDTW(input)
        suffix = self.SoftDTW(input.flip(dims=[1, 2])).flip(dims=[1, 2])
        out = prefix + suffix - neg_inf_to_zero(input)
        assert not out.isnan().any()
        return out

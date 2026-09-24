"""Whether two backbones hold identical weights, for resume and EMA assertions."""

from __future__ import annotations

import torch
from torch import nn


def parameters_are_equal(left: nn.Module, right: nn.Module) -> bool:
    left_state, right_state = left.state_dict(), right.state_dict()
    if left_state.keys() != right_state.keys():
        return False
    return all(torch.equal(left_state[key], right_state[key]) for key in left_state)

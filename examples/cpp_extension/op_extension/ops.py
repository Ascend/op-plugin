import torch
from torch import Tensor


def custom_add(a: Tensor, b: Tensor) -> Tensor:
    return torch.ops.npu.ascendc_add(a, b)
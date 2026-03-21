import torch

def add_custom(self, other):
    return torch.ops.cpp_extension_base.add_custom(self, other)

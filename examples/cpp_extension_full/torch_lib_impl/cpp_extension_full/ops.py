import torch

def add_custom(self, other):
    return torch.ops.cpp_extension_full.add_custom(self, other)

import torch

def add_custom(self, other):
    return torch.ops.cpp_extension_full.add_custom(self, other)

def add_custom_backward(grad):
    return torch.ops.cpp_extension_full.add_custom_backward(grad)

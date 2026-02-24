import torch
import custom_ops_lib


def add_custom(self, other):
    return custom_ops_lib.add_custom(self, other)

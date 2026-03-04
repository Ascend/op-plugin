import os
import pathlib
import torch
from . import ops


# Load the custom operator library
def _load_opextension_so():
    so_dir = pathlib.Path(__file__).parents[0]
    so_files = list(so_dir.glob('custom_ops_lib*.so'))

    if not so_files:
        raise FileNotFoundError(f"can not find custom_ops_lib*.so in {so_dir}")

    custom_so_path = str(so_files[0])
    torch.ops.load_library(custom_so_path)

_load_opextension_so()
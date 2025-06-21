from .atb._atb_api_docs import _add_torch_npu_atb_api_docstr
from .atb._atb_ops import _patch_atb_and_loadso

_patch_atb_and_loadso()


_add_torch_npu_atb_api_docstr()

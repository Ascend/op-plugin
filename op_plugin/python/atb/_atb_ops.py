import os
import sys
import types
import pathlib
from itertools import chain
from functools import wraps, lru_cache
import torch
import torch_npu


API_LIST = [
    '_npu_matmul_add_fp32',
    '_npu_quant_rms_norm',
    '_npu_group_topk',
    '_npu_paged_attention',
    '_npu_paged_attention_mla',
    '_npu_paged_attention_quant',
    '_npu_quantize_per_tensor',
    '_npu_reshape_and_cache',
    '_npu_reshape_and_cache_siso',
    '_npu_rotary_embedding',
    '_npu_flash_attention',
    '_npu_flash_attention_unpad',
    '_npu_paged_attention_splitfuse',
    '_npu_flash_attention_qlens',
]


ATB_API_LIST = [
    'npu_paged_cache_load',
    'npu_multi_head_latent_attention',
    '_npu_paged_attention_v2',
    '_npu_flash_attention_v2',
    '_npu_flash_attention_prefix_v2',
    'npu_fused_add_topk_div',
    'npu_ring_mla',
    'npu_self_attention_prefix_encoder',
    'npu_mla_preprocess',
]

ATB_MODULE_NAME = 'atb'
ATB_MODULE = types.ModuleType(ATB_MODULE_NAME)


def _add_atb_module():
    
    setattr(torch_npu, ATB_MODULE_NAME, ATB_MODULE)
    sys.modules[f'torch_npu.{ATB_MODULE_NAME}'] = ATB_MODULE


_add_atb_module()


NNAL_EX = None
GLOBAL_E = None


try:
    npu_path = pathlib.Path(__file__).parents[2]
    atb_so_path = os.path.join(npu_path, 'lib', 'libop_plugin_atb.so')
    from torch_npu.utils._path_manager import PathManager
    PathManager.check_directory_path_readable(atb_so_path)
    torch.ops.load_library(atb_so_path)
    import torch_npu.op_plugin.atb._atb_meta_registrations
except OSError as e:
    nnal_strerror = ""
    if "libatb.so" in str(e):
        nnal_strerror = "Please check that the nnal package is installed. "\
                        "Please run 'source set_env.sh' in the NNAL installation path."
    if "undefined symbol" in str(e):
        nnal_strerror = "Please check the version of the NNAL package. "\
                        "An undefined symbol was found, "\
                        "which may be caused by a version mismatch between NNAL and torch_npu."
    NNAL_EX = OSError(e.errno, nnal_strerror)
    NNAL_EX.__traceback__ = e.__traceback__
    GLOBAL_E = e


@lru_cache(None)
def _register_atb_extensions():
    global NNAL_EX, GLOBAL_E
    if NNAL_EX is not None:
        raise NNAL_EX from GLOBAL_E
    _patch_atb_ops()
    from torch_npu.op_plugin.atb._atb_api_docs import _add_torch_npu_atb_api_docstr
    _add_torch_npu_atb_api_docstr()


def lazy_load_atb_so(api_func):
    @wraps(api_func)
    def wrapper(*args, **kwargs):
        _register_atb_extensions()
        return api_func(*args, **kwargs)
    
    return wrapper


def create_lazy_atb_function(api_name):
    @lazy_load_atb_so
    def generated_function(*args, **kwargs):
        return getattr(torch.ops.atb, api_name)(*args, **kwargs)
    generated_function.__name__ = api_name
    return generated_function


def generate_atb_lazy_function():
    for api_name in chain(API_LIST, ATB_API_LIST):
        globals()[api_name] = create_lazy_atb_function(api_name)


generate_atb_lazy_function()


def _patch_atb_ops():
    for api_name in API_LIST:
        setattr(torch_npu, api_name, getattr(torch.ops.atb, api_name))
    for api_name in ATB_API_LIST:
        setattr(ATB_MODULE, api_name, getattr(torch.ops.atb, api_name))


def _patch_atb_and_loadso():
    for api_name in API_LIST:
        func = globals().get(api_name)
        setattr(torch_npu, api_name, func)
    for api_name in ATB_API_LIST:
        func = globals().get(api_name)
        setattr(ATB_MODULE, api_name, func)

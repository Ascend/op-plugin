import math
import torch
import torch_npu
from torch.library import Library, impl
from torch.fx.node import has_side_effect
from torch_npu.utils._error_code import ErrCode, ops_error
from torch_npu.npu.utils import get_cann_version

'''
Registering Meta implementations for custom ops
'''
BIT_NUMBER = 128
UINT8_BIT_NUMBER = 8
NPU_TENSOR_DIM_LIMIT = 8
INPUTS_DIM_LIMIT_QUANTCONV2D = 4
ATTR_DIM_LIMIT_QUANTCONV2D = 2
#meta register implementation
m = Library("npu", "IMPL", "Meta")
m_aten = Library("aten", "IMPL", "Meta")

TORCH_DTYPE_MAP = {
    torch.float16: 5,
    torch.bfloat16: 15,
    torch.float32: 6,
    torch.float8_e5m2: 23,
    torch.float8_e4m3fn: 24,
    torch.bits8: 21,
    torch.int8: 1,
    torch.int32: 3,
}

TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP = {
    0: torch.uint8,
    1: torch.int8,
    2: torch.int16,
    3: torch.int32,
    4: torch.int64,
    5: torch.float16,
    6: torch.float32,
    7: torch.float64,
    8: torch.complex32,
    9: torch.complex64,
    10: torch.complex128,
    11: torch.bool,
    12: torch.qint8,
    13: torch.quint8,
    14: torch.qint32,
    15: torch.bfloat16,
    16: torch.quint4x2,
    21: torch.bits8,
    23: torch.float8_e5m2,
    24: torch.float8_e4m3fn,
    285: torch.uint8,  # torch_npu.int4 use torch.uint8
    290: torch.uint8,  # torch_npu.hifloat8 use torch.uint8
    291: torch.float8_e5m2,
    292: torch.float8_e4m3fn,
    296: torch.uint8,  # torch_npu.float4_e2m1fn_x2 use torch.uint8
    297: torch.uint8,  # torch_npu.float4_e1m2fn_x2 use torch.uint8
}


TORCH_NPU_DTYPE_TO_STRING_MAP = {
    290: "torch_npu.hifloat8",
    293: "torch_npu.float8_e8m0fnu",
    296: "torch_npu.float4_e2m1fn_x2",
    297: "torch_npu.float4_e1m2fn_x2",
}


def npu_dtype_to_str(dtype):
    torch_dtype = TORCH_NPU_DTYPE_TO_STRING_MAP.get(dtype)
    if torch_dtype is not None:
        return torch_dtype
    torch_dtype = TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP.get(dtype)
    if torch_dtype is None:
        return str(dtype)
    return str(torch_dtype)


def _is_pytorch_version_ge(min_version):
    def parse_version(v):
        parts = list(map(int, v.split('.')[:3]))
        return tuple(parts + [0] * (3 - len(parts)))

    current_version_str = torch.__version__.split('+')[0]
    current_version = parse_version(current_version_str)
    target_version = parse_version(min_version)
    return current_version >= target_version


@impl(m_aten, "matmul_backward")
def matmul_backward_meta(grad, self, other, mask):
    return (torch.empty_like(self), torch.empty_like(other))


@impl(m, "npu_incre_flash_attention")
def npu_incre_flash_attention_forward(query, key, value, *, padding_mask=None, atten_mask=None, pse_shift=None, actual_seq_lengths=None,
                                      antiquant_scale=None, antiquant_offset=None, block_table=None,
                                      dequant_scale1=None, quant_scale1=None, dequant_scale2=None, quant_scale2=None,
                                      quant_offset2=None, kv_padding_size=None, num_heads=1, scale_value=1.0, input_layout="BSH",
                                      num_key_value_heads=0, block_size=0, inner_precise=1):
    if quant_scale2 is not None:
        return torch.empty_like(query, dtype=torch.int8)
    elif query.dtype == torch.int8:
        return torch.empty_like(query, dtype=torch.half)
    else:
        return torch.empty_like(query)


@impl(m, "npu_sparse_flash_attention")
def npu_sparse_flash_attention_forward(query, key, value, sparse_indices, scale_value, *, block_table=None,
                                         actual_seq_lengths_query=None, actual_seq_lengths_kv=None, query_rope=None,
                                         key_rope=None, sparse_block_size=1, layout_query="BSND", layout_kv="BSND",
                                         sparse_mode=3, pre_tokens=(1 << 63) - 1, next_tokens=(1 << 63) - 1,
                                         attention_mode=0, return_softmax_lse=False):
    require_param = {"query": query, "key": key, "value": value, "sparse_indices": sparse_indices}

    for item_name, item in require_param.items():
        torch._check(
            item is not None,
            lambda: item_name + " should not be None, but the actual value is None" + ops_error(ErrCode.VALUE),
        )

    torch._check(
        query.numel() > 0,
        lambda: "Input query should not be empty." + ops_error(ErrCode.VALUE),
    )
    torch._check(
        key.numel() > 0,
        lambda: "Input key should not be empty." + ops_error(ErrCode.VALUE),
    )
    torch._check(
        value.numel() > 0,
        lambda: "Input value should not be empty." + ops_error(ErrCode.VALUE),
    )
    torch._check(
        sparse_indices.numel() > 0,
        lambda: "Input sparse_indices should not be empty." + ops_error(ErrCode.VALUE),
    )
    torch._check(
        not return_softmax_lse,
        lambda: "when return_softmax_lse is true, not support pytorch compile." + ops_error(ErrCode.VALUE),
    )

    if layout_query == "TND":
        torch._check(
            query.dim() == 3,
            lambda: "When the layout of query is TND, the query dimension must be 3, but got " + str(query.dim()) + ops_error(ErrCode.VALUE),
        )
        attention_out = torch.empty([query.size(0), query.size(1), query.size(2)], dtype=query.dtype, device='meta')
    elif layout_query == "BSND":
        torch._check(
            query.dim() == 4,
            lambda: "When the layout of query is BSND, the query dimension must be 4, but got " + str(query.dim()) + ops_error(ErrCode.VALUE),
        )
        attention_out = torch.empty([query.size(0), query.size(1), query.size(2), query.size(3)], dtype=query.dtype, device='meta')
    else:
        torch._check(
            False,
            lambda: "Not support layout of query: " + layout_query + ops_error(ErrCode.VALUE),
        )

    if return_softmax_lse:
        if layout_query == "TND":
            softmax_max = torch.empty([key.size(1), query.size(0), query.size(1) // key.size(1)], dtype=torch.float32, device='meta')
            softmax_sum = torch.empty([key.size(1), query.size(0), query.size(1) // key.size(1)], dtype=torch.float32, device='meta')
        if layout_query == "BSND":
            softmax_max = torch.empty([query.size(0), key.size(2), query.size(1), query.size(2) // key.size(2)], dtype=torch.float32, device='meta')
            softmax_sum = torch.empty([query.size(0), key.size(2), query.size(1), query.size(2) // key.size(2)], dtype=torch.float32, device='meta')
    else:
        softmax_max = torch.empty([0], dtype=torch.float32, device='meta')
        softmax_sum = torch.empty([0], dtype=torch.float32, device='meta')
    return (attention_out, softmax_max, softmax_sum)


@impl(m, "npu_sparse_flash_attention_grad")
def npu_sparse_flash_attention_grad_meta(query, key, value, sparse_indices, d_out, out, softmax_max, softmax_sum, scale_value, sparse_block_size, query_rope=None, key_rope=None, actual_seq_qlen=None, actual_seq_kvlen=None, layout="BSND", sparse_mode=3, pre_tokens=9223372036854775807, next_tokens=9223372036854775807, attention_mode=0):
    d_query = query.new_empty(query.shape, dtype=query.dtype, device='meta')
    d_key = key.new_empty(key.shape, dtype=key.dtype, device='meta')
    d_value = value.new_empty(value.shape, dtype=value.dtype, device='meta')
    d_query_rope = torch.empty([0], dtype=query.dtype, device='meta') if query_rope is None else query_rope.new_empty(query_rope.shape, dtype=query_rope.dtype, device='meta')
    d_key_rope = torch.empty([0], dtype=key.dtype, device='meta') if key_rope is None else key_rope.new_empty(key_rope.shape, dtype=key_rope.dtype, device='meta')
    return (d_query, d_key, d_value, d_query_rope, d_key_rope)


@impl(m, "npu_mla_prolog")
def npu_mla_prolog_forward(token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq, rmsnorm_gamma_ckv,
                   rope_sin, rope_cos, cache_index, kv_cache, kr_cache, *, dequant_scale_x=None, dequant_scale_w_dq=None, dequant_scale_w_uq_qr=None, dequant_scale_w_dkv_kr=None,
                   quant_scale_ckv=None, quant_scale_ckr=None, smooth_scales_cq=None,
                   rmsnorm_epsilon_cq=1e-5, rmsnorm_epsilon_ckv=1e-5, cache_mode="PA_BSND"):
    require_param = {"token_x": token_x, "weight_dq": weight_dq, "weight_uq_qr": weight_uq_qr, "weight_uk": weight_uk, "weight_dkv_kr": weight_dkv_kr, "rmsnorm_gamma_cq": rmsnorm_gamma_cq, "rmsnorm_gamma_ckv": rmsnorm_gamma_ckv, "rope_sin": rope_sin, "rope_cos": rope_cos, "cache_index": cache_index, "kv_cache": kv_cache, "kr_cache": kr_cache}

    for item_name, item in require_param.items():
        torch._check(
            item is not None,
            lambda: item_name + " should not be None, but the actual value is None" + ops_error(ErrCode.VALUE),
        )

    token_x_dim = token_x.dim()
    torch._check(
        token_x_dim == 2 or token_x_dim == 3,
        lambda: "token_x dim num should be 2 or 3, but the actual value is " + str(token_x_dim) + ops_error(ErrCode.VALUE),
    )

    weight_uk_dim = weight_uk.dim()
    torch._check(
        weight_uk_dim == 3,
        lambda: "weight_uk dim num should be 3, but the actual value is " + str(weight_uk_dim) + ops_error(ErrCode.VALUE),
    )

    rope_sin_dim = rope_sin.dim()
    torch._check(
        rope_sin_dim == 2 or rope_sin_dim == 3,
        lambda: "rope_sin dim num should be 2 or 3, but the actual value is " + str(rope_sin_dim) + ops_error(ErrCode.VALUE),
    )

    if token_x_dim == 3:
        query_shape = []
        query_shape.append(token_x.size(0))
        query_shape.append(token_x.size(1))
        query_shape.append(weight_uk.size(0))
        query_shape.append(weight_uk.size(2))

        query_rope_shape = []
        query_rope_shape.append(token_x.size(0))
        query_rope_shape.append(token_x.size(1))
        query_rope_shape.append(weight_uk.size(0))
        query_rope_shape.append(rope_sin.size(2))
    else:
        query_shape = []
        query_shape.append(token_x.size(0))
        query_shape.append(weight_uk.size(0))
        query_shape.append(weight_uk.size(2))

        query_rope_shape = []
        query_rope_shape.append(token_x.size(0))
        query_rope_shape.append(weight_uk.size(0))
        query_rope_shape.append(rope_sin.size(1))

    query = torch.empty(query_shape, dtype=rope_sin.dtype, device='meta')
    query_rope = torch.empty(query_rope_shape, dtype=rope_sin.dtype, device='meta')
    kv_cache_out = torch.empty_like(kv_cache, dtype=kv_cache.dtype, device='meta')
    kr_cache_out = torch.empty_like(kr_cache, dtype=kr_cache.dtype, device='meta')

    return (query, query_rope, kv_cache_out, kr_cache_out)


@impl(m, "npu_mla_prolog_v2")
def npu_mla_prolog_v2_forward(token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq, rmsnorm_gamma_ckv,
                    rope_sin, rope_cos, cache_index, kv_cache, kr_cache, *, dequant_scale_x=None, dequant_scale_w_dq=None, dequant_scale_w_uq_qr=None, dequant_scale_w_dkv_kr=None,
                    quant_scale_ckv=None, quant_scale_ckr=None, smooth_scales_cq=None,
                    rmsnorm_epsilon_cq=1e-5, rmsnorm_epsilon_ckv=1e-5, cache_mode="PA_BSND"):

    require_param = {"token_x": token_x, "weight_dq": weight_dq, "weight_uq_qr": weight_uq_qr, "weight_uk": weight_uk, "weight_dkv_kr": weight_dkv_kr, "rmsnorm_gamma_cq": rmsnorm_gamma_cq, "rmsnorm_gamma_ckv": rmsnorm_gamma_ckv, "rope_sin": rope_sin, "rope_cos": rope_cos, "cache_index": cache_index, "kv_cache": kv_cache, "kr_cache": kr_cache}

    for item_name, item in require_param.items():
        torch._check(
            item is not None,
            lambda: item_name + " should not be None, but the actual value is None" + ops_error(ErrCode.VALUE),
        )

    token_x_dim = token_x.dim()
    torch._check(
        token_x_dim == 2 or token_x_dim == 3,
        lambda: "token_x dim num should be 2 or 3, but the actual value is " + str(token_x_dim) + ops_error(ErrCode.VALUE),
    )

    weight_uk_dim = weight_uk.dim()
    torch._check(
        weight_uk_dim == 3,
        lambda: "weight_uk dim num should be 3, but the actual value is " + str(weight_uk_dim) + ops_error(ErrCode.VALUE),
    )

    rope_sin_dim = rope_sin.dim()
    torch._check(
        rope_sin_dim == 2 or rope_sin_dim == 3,
        lambda: "rope_sin dim num should be 2 or 3, but the actual value is " + str(rope_sin_dim) + ops_error(ErrCode.VALUE),
    )

    if token_x_dim == 3:
        query_shape = []
        query_shape.append(token_x.size(0))
        query_shape.append(token_x.size(1))
        query_shape.append(weight_uk.size(0))
        query_shape.append(weight_uk.size(2))

        query_rope_shape = []
        query_rope_shape.append(token_x.size(0))
        query_rope_shape.append(token_x.size(1))
        query_rope_shape.append(weight_uk.size(0))
        query_rope_shape.append(rope_sin.size(2))

        dequant_scale_q_nope_shape = []
        dequant_scale_q_nope_shape.append(token_x.size(0) * token_x.size(1))
        dequant_scale_q_nope_shape.append(weight_uk.size(0)) # support pertoken-head dynamic antiquant
        dequant_scale_q_nope_shape.append(1)

    else:
        query_shape = []
        query_shape.append(token_x.size(0))
        query_shape.append(weight_uk.size(0))
        query_shape.append(weight_uk.size(2))

        query_rope_shape = []
        query_rope_shape.append(token_x.size(0))
        query_rope_shape.append(weight_uk.size(0))
        query_rope_shape.append(rope_sin.size(1))

        dequant_scale_q_nope_shape = []
        dequant_scale_q_nope_shape.append(token_x.size(0))
        dequant_scale_q_nope_shape.append(weight_uk.size(0)) # support pertoken-head dynamic antiquant
        dequant_scale_q_nope_shape.append(1)

    # kvcache量化
    if token_x.dtype == torch.int8 and quant_scale_ckv is not None:
        query = torch.empty(query_shape, dtype=torch.int8, device='meta')
        dequant_scale_q_nope = torch.empty(dequant_scale_q_nope_shape, dtype=torch.float32, device='meta')
    else:
        query = torch.empty(query_shape, dtype=rope_sin.dtype, device='meta')
        dequant_scale_q_nope = torch.empty([1], dtype=torch.float32, device='meta')

    query_rope = torch.empty(query_rope_shape, dtype=torch.bfloat16, device='meta') # default dtype bfloat16
    kv_cache_out = torch.empty_like(kv_cache, dtype=kv_cache.dtype, device='meta')
    kr_cache_out = torch.empty_like(kr_cache, dtype=kr_cache.dtype, device='meta')

    return (query, query_rope, kv_cache_out, kr_cache_out, dequant_scale_q_nope)


@impl(m, "npu_mla_prolog_v3")
def npu_mla_prolog_v3_forward(token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq, rmsnorm_gamma_ckv,
                    rope_sin, rope_cos, kv_cache, kr_cache, *, cache_index=None, dequant_scale_x=None, dequant_scale_w_dq=None, dequant_scale_w_uq_qr=None, dequant_scale_w_dkv_kr=None,
                    quant_scale_ckv=None, quant_scale_ckr=None, smooth_scales_cq=None, actual_seq_len=None, k_nope_clip_alpha=None, rmsnorm_epsilon_cq=1e-5, rmsnorm_epsilon_ckv=1e-5,
                    cache_mode="PA_BSND", query_norm_flag=False, weight_quant_mode=0, kv_cache_quant_mode=0, query_quant_mode=0, ckvkr_repo_mode=0, quant_scale_repo_mode=0, tile_size=128, qc_qr_scale=1.0, kc_scale=1.0):

    require_param = {"token_x": token_x, "weight_dq": weight_dq, "weight_uq_qr": weight_uq_qr, "weight_uk": weight_uk, "weight_dkv_kr": weight_dkv_kr, "rmsnorm_gamma_cq": rmsnorm_gamma_cq, "rmsnorm_gamma_ckv": rmsnorm_gamma_ckv, "rope_sin": rope_sin, "rope_cos": rope_cos, "kv_cache": kv_cache, "kr_cache": kr_cache}

    for item_name, item in require_param.items():
        torch._check(
            item is not None,
            lambda: item_name + " should not be None, but the actual value is None" + ops_error(ErrCode.VALUE),
        )

    token_x_dim = token_x.dim()
    torch._check(
        token_x_dim == 2 or token_x_dim == 3,
        lambda: "token_x dim num should be 2 or 3, but the actual value is " + str(token_x_dim) + ops_error(ErrCode.VALUE),
    )

    weight_uk_dim = weight_uk.dim()
    torch._check(
        weight_uk_dim == 3,
        lambda: "weight_uk dim num should be 3, but the actual value is " + str(weight_uk_dim) + ops_error(ErrCode.VALUE),
    )

    rope_sin_dim = rope_sin.dim()
    if token_x_dim == 3:
        torch._check(
            rope_sin_dim == 3,
            lambda: "when token_x dim num is 3, rope_sin dim num should be 3, but the actual value is " + str(rope_sin_dim) + ops_error(ErrCode.VALUE),
        )

        query_shape = []
        query_shape.append(token_x.size(0))
        query_shape.append(token_x.size(1))
        query_shape.append(weight_uk.size(0))
        query_shape.append(weight_uk.size(2))

        query_rope_shape = []
        query_rope_shape.append(token_x.size(0))
        query_rope_shape.append(token_x.size(1))
        query_rope_shape.append(weight_uk.size(0))
        query_rope_shape.append(rope_sin.size(2))

        dequant_scale_q_nope_shape = []
        dequant_scale_q_nope_shape.append(token_x.size(0) * token_x.size(1))
        dequant_scale_q_nope_shape.append(weight_uk.size(0)) # support pertoken-head dynamic antiquant
        dequant_scale_q_nope_shape.append(1)

        query_norm_shape = []
        query_norm_shape.append(token_x.size(0))
        query_norm_shape.append(token_x.size(1))
        query_norm_shape.append(weight_dq.size(1))

        dequant_scale_q_norm_shape = []
        dequant_scale_q_norm_shape.append(token_x.size(0) * token_x.size(1))
        dequant_scale_q_norm_shape.append(1)

    else:
        torch._check(
            rope_sin_dim == 2,
            lambda: "when token_x dim num is 2, rope_sin dim num should be 2, but the actual value is " + str(rope_sin_dim) + ops_error(ErrCode.VALUE),
        )

        query_shape = []
        query_shape.append(token_x.size(0))
        query_shape.append(weight_uk.size(0))
        query_shape.append(weight_uk.size(2))

        query_rope_shape = []
        query_rope_shape.append(token_x.size(0))
        query_rope_shape.append(weight_uk.size(0))
        query_rope_shape.append(rope_sin.size(1))

        dequant_scale_q_nope_shape = []
        dequant_scale_q_nope_shape.append(token_x.size(0))
        dequant_scale_q_nope_shape.append(weight_uk.size(0)) # support pertoken-head dynamic antiquant
        dequant_scale_q_nope_shape.append(1)

        query_norm_shape = []
        query_norm_shape.append(token_x.size(0))
        query_norm_shape.append(weight_dq.size(1))

        dequant_scale_q_norm_shape = []
        dequant_scale_q_norm_shape.append(token_x.size(0))
        dequant_scale_q_norm_shape.append(1)

    is_cann_version_gte_required = torch_npu.npu.utils._is_gte_cann_version("8.5.0.alpha003", "CANN") # whether cann version >= 8.5.0.alpha003
    # kvcache量化
    if weight_quant_mode == 3 and kv_cache_quant_mode == 1:
        query = torch.empty(query_shape, dtype=torch.float8_e4m3fn, device='meta')
        dequant_scale_q_nope = torch.empty(dequant_scale_q_nope_shape, dtype=torch.float32, device='meta')
    elif weight_quant_mode == 2 and kv_cache_quant_mode == 1:
        query = torch.empty(query_shape, dtype=torch.int8, device='meta')
        dequant_scale_q_nope = torch.empty(dequant_scale_q_nope_shape, dtype=torch.float32, device='meta')
    else:
        query = torch.empty(query_shape, dtype=rope_sin.dtype, device='meta')
        if is_cann_version_gte_required:
            dequant_scale_q_nope = torch.empty([0], dtype=torch.float32, device='meta')
        else:
            dequant_scale_q_nope = torch.empty([1], dtype=torch.float32, device='meta')

    # 输出query_norm
    if query_norm_flag:
        query_norm = torch.empty(query_norm_shape, dtype=weight_uq_qr.dtype, device='meta')
        if weight_quant_mode == 1 or weight_quant_mode == 2:
            dequant_scale_q_norm = torch.empty(dequant_scale_q_norm_shape, dtype=torch.float32, device='meta')
        else:
            if is_cann_version_gte_required:
                dequant_scale_q_norm = torch.empty([0], dtype=torch.float32, device='meta')
            else:
                dequant_scale_q_norm = torch.empty([1], dtype=torch.float32, device='meta')
    else:
        if is_cann_version_gte_required:
            query_norm = torch.empty([0], dtype=weight_uq_qr.dtype, device='meta')
            dequant_scale_q_norm = torch.empty([0], dtype=torch.float32, device='meta')
        else:
            query_norm = torch.empty([1], dtype=weight_uq_qr.dtype, device='meta')
            dequant_scale_q_norm = torch.empty([1], dtype=torch.float32, device='meta')

    query_rope = torch.empty(query_rope_shape, dtype=torch.bfloat16, device='meta') # default dtype bfloat16

    return (query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm)


@impl(m, "npu_mla_prolog_v3_functional")
def npu_mla_prolog_v3_functional_forward(token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq, rmsnorm_gamma_ckv,
                    rope_sin, rope_cos, kv_cache, kr_cache, *, cache_index=None, dequant_scale_x=None, dequant_scale_w_dq=None, dequant_scale_w_uq_qr=None, dequant_scale_w_dkv_kr=None,
                    quant_scale_ckv=None, quant_scale_ckr=None, smooth_scales_cq=None, actual_seq_len=None, k_nope_clip_alpha=None, rmsnorm_epsilon_cq=1e-5, rmsnorm_epsilon_ckv=1e-5,
                    cache_mode="PA_BSND", query_norm_flag=False, weight_quant_mode=0, kv_cache_quant_mode=0, query_quant_mode=0, ckvkr_repo_mode=0, quant_scale_repo_mode=0, tile_size=128, qc_qr_scale=1.0, kc_scale=1.0):


    require_param = {"token_x": token_x, "weight_dq": weight_dq, "weight_uq_qr": weight_uq_qr, "weight_uk": weight_uk, "weight_dkv_kr": weight_dkv_kr, "rmsnorm_gamma_cq": rmsnorm_gamma_cq, "rmsnorm_gamma_ckv": rmsnorm_gamma_ckv, "rope_sin": rope_sin, "rope_cos": rope_cos, "kv_cache": kv_cache, "kr_cache": kr_cache}

    for item_name, item in require_param.items():
        torch._check(
            item is not None,
            lambda: item_name + " should not be None, but the actual value is None" + ops_error(ErrCode.VALUE),
        )

    token_x_dim = token_x.dim()
    torch._check(
        token_x_dim == 2 or token_x_dim == 3,
        lambda: "token_x dim num should be 2 or 3, but the actual value is " + str(token_x_dim) + ops_error(ErrCode.VALUE),
    )

    weight_uk_dim = weight_uk.dim()
    torch._check(
        weight_uk_dim == 3,
        lambda: "weight_uk dim num should be 3, but the actual value is " + str(weight_uk_dim) + ops_error(ErrCode.VALUE),
    )

    rope_sin_dim = rope_sin.dim()
    if token_x_dim == 3:
        torch._check(
            rope_sin_dim == 3,
            lambda: "when token_x dim num is 3, rope_sin dim num should be 3, but the actual value is " + str(rope_sin_dim) + ops_error(ErrCode.VALUE),
        )

        query_shape = []
        query_shape.append(token_x.size(0))
        query_shape.append(token_x.size(1))
        query_shape.append(weight_uk.size(0))
        query_shape.append(weight_uk.size(2))

        query_rope_shape = []
        query_rope_shape.append(token_x.size(0))
        query_rope_shape.append(token_x.size(1))
        query_rope_shape.append(weight_uk.size(0))
        query_rope_shape.append(rope_sin.size(2))

        dequant_scale_q_nope_shape = []
        dequant_scale_q_nope_shape.append(token_x.size(0) * token_x.size(1))
        dequant_scale_q_nope_shape.append(weight_uk.size(0)) # support pertoken-head dynamic antiquant
        dequant_scale_q_nope_shape.append(1)

        query_norm_shape = []
        query_norm_shape.append(token_x.size(0))
        query_norm_shape.append(token_x.size(1))
        query_norm_shape.append(weight_dq.size(1))

        dequant_scale_q_norm_shape = []
        dequant_scale_q_norm_shape.append(token_x.size(0) * token_x.size(1))
        dequant_scale_q_norm_shape.append(1)

    else:
        torch._check(
            rope_sin_dim == 2,
            lambda: "when token_x dim num is 2, rope_sin dim num should be 2, but the actual value is " + str(rope_sin_dim) + ops_error(ErrCode.VALUE),
        )

        query_shape = []
        query_shape.append(token_x.size(0))
        query_shape.append(weight_uk.size(0))
        query_shape.append(weight_uk.size(2))

        query_rope_shape = []
        query_rope_shape.append(token_x.size(0))
        query_rope_shape.append(weight_uk.size(0))
        query_rope_shape.append(rope_sin.size(1))

        dequant_scale_q_nope_shape = []
        dequant_scale_q_nope_shape.append(token_x.size(0))
        dequant_scale_q_nope_shape.append(weight_uk.size(0)) # support pertoken-head dynamic antiquant
        dequant_scale_q_nope_shape.append(1)

        query_norm_shape = []
        query_norm_shape.append(token_x.size(0))
        query_norm_shape.append(weight_dq.size(1))

        dequant_scale_q_norm_shape = []
        dequant_scale_q_norm_shape.append(token_x.size(0))
        dequant_scale_q_norm_shape.append(1)

    is_cann_version_gte_required = torch_npu.npu.utils._is_gte_cann_version("8.5.0.alpha003", "CANN") # whether cann version >= 8.5.0.alpha003
    # kvcache量化
    if weight_quant_mode == 3 and kv_cache_quant_mode == 1:
        query = torch.empty(query_shape, dtype=torch.float8_e4m3fn, device='meta')
        dequant_scale_q_nope = torch.empty(dequant_scale_q_nope_shape, dtype=torch.float32, device='meta')
    elif weight_quant_mode == 2 and kv_cache_quant_mode == 1:
        query = torch.empty(query_shape, dtype=torch.int8, device='meta')
        dequant_scale_q_nope = torch.empty(dequant_scale_q_nope_shape, dtype=torch.float32, device='meta')
    else:
        query = torch.empty(query_shape, dtype=rope_sin.dtype, device='meta')
        if is_cann_version_gte_required:
            dequant_scale_q_nope = torch.empty([0], dtype=torch.float32, device='meta')
        else:
            dequant_scale_q_nope = torch.empty([1], dtype=torch.float32, device='meta')

    query_rope = torch.empty(query_rope_shape, dtype=torch.bfloat16, device='meta') # default dtype bfloat16

    # 输出query_norm
    if query_norm_flag:
        query_norm = torch.empty(query_norm_shape, dtype=weight_uq_qr.dtype, device='meta')
        # 动态量化
        if weight_quant_mode == 1 or weight_quant_mode == 2:
            dequant_scale_q_norm = torch.empty(dequant_scale_q_norm_shape, dtype=torch.float32, device='meta')
        else:
            if is_cann_version_gte_required:
                dequant_scale_q_norm = torch.empty([0], dtype=torch.float32, device='meta')
            else:
                dequant_scale_q_norm = torch.empty([1], dtype=torch.float32, device='meta')
    else:
        if is_cann_version_gte_required:
            query_norm = torch.empty([0], dtype=weight_uq_qr.dtype, device='meta')
            dequant_scale_q_norm = torch.empty([0], dtype=torch.float32, device='meta')
        else:
            query_norm = torch.empty([1], dtype=weight_uq_qr.dtype, device='meta')
            dequant_scale_q_norm = torch.empty([1], dtype=torch.float32, device='meta')

    kv_cache_out = torch.empty_like(kv_cache, dtype=kv_cache.dtype, device='meta')
    kr_cache_out = torch.empty_like(kr_cache, dtype=kr_cache.dtype, device='meta')

    return (query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm, kv_cache_out, kr_cache_out)

if "2.1." in torch.__version__:
    @impl(m, "npu_prompt_flash_attention")
    def npu_prompt_flash_attention_forward(query, key, value, *, padding_mask=None, atten_mask=None, pse_shift=None, actual_seq_lengths=None, deq_scale1=None, quant_scale1=None, deq_scale2=None, quant_scale2=None, quant_offset2=None, num_heads=1, scale_value=1.0, pre_tokens=2147483647, next_tokens=0, input_layout="BSH", num_key_value_heads=0, actual_seq_lengths_kv=None, sparse_mode=0):
        tmp_out = torch.empty_like(query, dtype=query.dtype, device='meta')
        if input_layout == "BNSD_BSND":
            tmp_out = torch.empty([query.size(0), query.size(2), query.size(1), query.size(3)], dtype=query.dtype, device='meta')
        elif input_layout == "SH":
            tmp_out = torch.empty([query.size(0), query.size(1)], dtype=query.dtype, device='meta')
        elif input_layout == "BSH" or input_layout == "NSD":
            tmp_out = torch.empty([query.size(0), query.size(1), query.size(2)], dtype=query.dtype, device='meta')
        elif input_layout == "TND":
            tmp_out = torch.empty([query.size(0), query.size(1), value.size(2)], dtype=query.dtype, device='meta')
        elif input_layout == "BNSD":
            tmp_out = torch.empty([query.size(0), query.size(1), query.size(2), query.size(3)],
                dtype=query.dtype, device='meta')
        elif input_layout == "BSND":
            tmp_out = torch.empty([query.size(0), query.size(1), query.size(2), query.size(3)],
                dtype=query.dtype, device='meta')
        else:
            torch._check(
                False,
                lambda: "not support layout: " + str(input_layout) + ops_error(ErrCode.VALUE),
            )
        if quant_scale2 is not None:
            return torch.empty_like(tmp_out, dtype=torch.int8)
        elif query.dtype == torch.int8:
            return torch.empty_like(tmp_out, dtype=torch.half)
        else:
            return torch.empty_like(tmp_out, dtype=query.dtype)
else:
    @impl(m, "npu_prompt_flash_attention")
    def npu_prompt_flash_attention_forward(query, key, value, *, padding_mask=None, atten_mask=None, pse_shift=None, actual_seq_lengths=None, deq_scale1=None, quant_scale1=None, deq_scale2=None, quant_scale2=None, quant_offset2=None, num_heads=1, scale_value=1.0, pre_tokens=2147483647, next_tokens=0, input_layout="BSH", num_key_value_heads=0, actual_seq_lengths_kv=None, sparse_mode=0):
        tmp_out = torch.empty_like(query, dtype=query.dtype, device='meta')
        if input_layout == "TND":
            tmp_out = torch.empty([query.size(0), query.size(1), value.size(2)], dtype=query.dtype, device='meta')

        if quant_scale2 is not None:
            return torch.empty_like(tmp_out, dtype=torch.int8)
        elif query.dtype == torch.int8:
            return torch.empty_like(tmp_out, dtype=torch.half)
        else:
            return torch.empty_like(tmp_out, dtype=query.dtype)


@impl(m, "npu_mm_reduce_scatter_base")
def npu_mm_reduce_scatter_base_meta(self, x2, hcom, world_size, reduce_op='sum',
                                    bias=None, x1_scale=None, x2_scale=None, comm_turn=0,
                                    output_dtype=None, comm_mode=None):
    if world_size <= 0:
        world_size = 1
    out_m = math.floor(self.size(0) / world_size)
    dtype = self.dtype
    size = [out_m, x2.size(1)]
    if x2_scale is not None:
        if x2_scale.dtype == torch.int64:
            dtype = torch.float16
        elif output_dtype is not None:
            dtype = output_dtype
        else:
            dtype = torch.bfloat16

    return torch.empty(size, dtype=dtype, device='meta')


@impl(m, "npu_quant_mm_reduce_scatter")
def npu_quant_mm_reduce_scatter_meta(self, x2, hcom, world_size, reduce_op='sum',
                                     bias=None, x1_scale=None, x2_scale=None, quant_scale=None,
                                     block_size=0, comm_turn=0, group_sizes=None, amax_output=False, y_dtype=None,
                                     x1_dtype=None, x2_dtype=None, x1_scale_dtype=None, x2_scale_dtype=None):
    if world_size <= 0:
        raise RuntimeError("world_size must be bigger than zero")
    out_m = math.floor(self.size(0) / world_size)
    torch_dtype = self.dtype if y_dtype is None else TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP[y_dtype]
    return (self.new_empty(out_m, x2.size(1), dtype=torch_dtype), self.new_empty(0, dtype=torch.float32))


@impl(m, "npu_quant_reduce_scatter")
def npu_quant_reduce_scatter_meta(x, scales, hcom_name, world_size, reduce_op='sum',
                                  output_dtype=None, x_dtype=None, scales_dtype=None):
    world_size = 2
    out_m = x.size(0) // world_size
    size = [out_m, x.size(1)]

    dtype = x.dtype
    if output_dtype is not None:
        dtype = TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP[output_dtype]
    else:
        dtype = torch.bfloat16

    return torch.empty(size, dtype=dtype, device='meta')


@impl(m, "npu_quant_all_reduce")
def npu_quant_all_reduce_meta(x, scales, hcom_name, world_size, reduce_op='sum',
                              output_dtype=None, x_dtype=None, scales_dtype=None):
    world_size = 2
    size = x.size()
    dtype = x.dtype
    if output_dtype is not None:
        dtype = TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP[output_dtype]
    else:
        dtype = torch.bfloat16

    return torch.empty(size, dtype=dtype, device='meta')


@impl(m, "npu_gmm_alltoallv")
def npu_gmm_alltoallv_meta(gmm_x, gmm_weight, hcom, ep_world_size, send_counts,
                        recv_counts, *, send_counts_tensor=None,
                        recv_counts_tensor=None, mm_x=None,
                        mm_weight=None, trans_gmm_weight=False,
                        trans_mm_weight=False):
    if ep_world_size <= 0:
        ep_world_size = 1
    out_x = sum(recv_counts)
    out_y = gmm_weight.size(2)
    if trans_gmm_weight:
        out_y = gmm_weight.size(1)
    out_mm_x = 0
    out_mm_y = 0
    y = None
    mm_y = None
    if mm_x is not None:
        out_mm_x = mm_x.size(0)
        out_mm_y = mm_weight.size(1)
        if trans_mm_weight:
            out_mm_y = mm_weight.size(0)
        mm_y = torch.empty([out_mm_x, out_mm_y], dtype=mm_x.dtype, device='meta')
    y = torch.empty([out_x, out_y], dtype=gmm_x.dtype, device='meta')
    return (y, mm_y)


@impl(m, "npu_fused_matmul")
def npu_fused_matmul_meta(x, x2, *, bias=None, x3=None, fused_op_type=''):
    torch._check(
        x is not None,
        lambda: "x must not be None, please input some value" + ops_error(ErrCode.TYPE),
    )
    torch._check(
        x2 is not None,
        lambda: "x2 must not be None, please input some value" + ops_error(ErrCode.TYPE),
    )

    torch._check(
        x.dtype == x2.dtype,
        lambda: "x and x2 type not same. x.dtype is" + str(x.dtype) + "x2.dtype is" + str(x2.dtype) + ops_error(ErrCode.TYPE),
    )

    x_dim = x.dim()
    torch._check(
        x_dim >= 2,
        lambda: "x dim num cannot be less than 2,but the actual value is " + str(x_dim) + ops_error(ErrCode.VALUE),
    )
    x2_dim = x2.dim()
    torch._check(
        x2_dim >= 2,
        lambda: "x2 dim num cannot be less than 2,but the actual value is " + str(x2_dim) + ops_error(ErrCode.VALUE),
    )

    ma = x.size(x_dim - 2)
    ka = x.size(x_dim - 1)
    kb = x2.size(x2_dim - 2)
    nb = x2.size(x2_dim - 1)
    torch._check(
        ka == kb,
        lambda: "ka and kb should be the same" + ops_error(ErrCode.TYPE),
    )

    # infer output shape
    out_dim_num = max(x_dim, x2_dim)
    shape_long = x if x_dim > x2_dim else x2
    shape_short = x2 if x_dim > x2_dim else x
    vaild_offset = out_dim_num - min(x_dim, x2_dim)
    output_shape = []
    for i in range(0, out_dim_num - 2):
        short_dim = 1 if i < vaild_offset else shape_short.size(i - vaild_offset)
        long_dim = shape_long.size(i)
        torch._check(
            not (short_dim > 1 and long_dim > 1 and short_dim != long_dim),
            lambda: "the batch shape cannot be broadcast" + ops_error(ErrCode.VALUE),
        )
        cur_batch_val = max(short_dim, long_dim)
        output_shape.append(cur_batch_val)

    output_shape.append(ma)
    output_shape.append(nb)

    if fused_op_type == "gelu_erf" or fused_op_type == "gelu_tanh":
        torch._check(
            x3 is None,
            lambda: "there is no x3 for gelu_erf and gelu_tanh" + ops_error(ErrCode.TYPE),
    )
    if fused_op_type == "add" or fused_op_type == "mul":
        torch._check(
            x3 is not None,
            lambda: "there must have x3 for add and mul" + ops_error(ErrCode.TYPE),
        )
    result = torch.empty(output_shape, dtype=x.dtype, device='meta')
    return torch.empty_like(result, dtype=x.dtype)


@impl(m, "npu_alltoallv_gmm")
def npu_alltoallv_gmm_meta(gmm_x, gmm_weight, hcom, ep_world_size, send_counts,
                        recv_counts, *, send_counts_tensor=None,
                        recv_counts_tensor=None, mm_x=None,
                        mm_weight=None, trans_gmm_weight=False,
                        trans_mm_weight=False, permute_out_flag=False):
    if ep_world_size <= 0:
        ep_world_size = 1
    out_x = sum(recv_counts)
    out_y = gmm_weight.size(2)
    if trans_gmm_weight:
        out_y = gmm_weight.size(1)
    out_mm_x = 0
    out_mm_y = 0
    permute_out_x = 0
    permute_out_y = 0
    gmm_y = None
    mm_y = None
    permute_out = None
    if mm_x is not None:
        out_mm_x = mm_x.size(0)
        out_mm_y = mm_weight.size(1)
        if trans_mm_weight:
            out_mm_y = mm_weight.size(0)
        mm_y = torch.empty([out_mm_x, out_mm_y], dtype=mm_x.dtype, device='meta')
    if permute_out_flag:
        permute_out_x = out_x
        permute_out_y = gmm_x.size(1)
        permute_out = torch.empty([permute_out_x, permute_out_y], dtype=gmm_x.dtype, device='meta')
    gmm_y = torch.empty([out_x, out_y], dtype=gmm_x.dtype, device='meta')
    return (gmm_y, mm_y, permute_out)


@impl(m, "npu_all_gather_base_mm")
def npu_all_gather_base_mm_meta(self, x2, hcom, world_size, bias=None,
                                x1_scale=None, x2_scale=None,
                                gather_index=0, gather_output=True, comm_turn=0,
                                output_dtype=None, comm_mode=None):
    if world_size <= 0:
        world_size = 1
    # out_gather_mm
    out_x = self.size(0)
    if gather_index == 0:
        out_x = self.size(0) * world_size
    out_y = x2.size(1)
    # out_gather
    out_gather_x = x2.size(0) * world_size
    out_gather_y = x2.size(1)
    if gather_index == 0:
        out_gather_x = self.size(0) * world_size
        out_gather_y = self.size(1)
    out_size = (out_x, out_y)
    gather_output_size = 0
    if gather_output:
        gather_output_size = (out_gather_x, out_gather_y)
    dtype = self.dtype
    if x2_scale is not None:
        if x2_scale.dtype == torch.int64:
            dtype = torch.float16
        elif output_dtype is not None:
            dtype = output_dtype
        else:
            dtype = torch.bfloat16

    return (torch.empty(out_size, dtype=dtype, device='meta'),
            torch.empty(gather_output_size, dtype=self.dtype, device='meta'))


@impl(m, "npu_all_gather_quant_mm")
def npu_all_gather_quant_mm_meta(self, x2, hcom, world_size, bias=None, x1_scale=None, x2_scale=None,
                                 quant_scale=None, block_size=0, gather_index=0, gather_output=True,
                                 comm_turn=0, group_sizes=None, amax_output=False, y_dtype=None, x1_dtype=None,
                                 x2_dtype=None, x1_scale_dtype=None, x2_scale_dtype=None):
    if world_size <= 0:
        raise RuntimeError("world_size must be bigger than zero")
    # out_gather_mm
    out_x = self.size(0)
    if gather_index == 0:
        out_x = self.size(0) * world_size
    out_y = x2.size(1)
    # out_gather
    out_gather_x = x2.size(0) * world_size
    out_gather_y = x2.size(1)
    if gather_index == 0:
        out_gather_x = self.size(0) * world_size
        out_gather_y = self.size(1)
    torch_dtype = self.dtype if y_dtype is None else TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP[y_dtype]

    if gather_output:
        return (self.new_empty((out_x, out_y), dtype=torch_dtype), self.new_empty(out_gather_x, out_gather_y),
                self.new_empty(0, dtype=torch.float32))
    else:
        return (self.new_empty((out_x, out_y), dtype=torch_dtype), self.new_empty(0),
                self.new_empty(0, dtype=torch.float32))


@impl(m, "npu_moe_init_routing")
def npu_moe_init_routing_meta(x, row_idx, expert_idx, active_num=99):
    n = x.size(0)
    h = x.size(1)
    k = row_idx.size(1)
    active_num = min(n, active_num)
    expanded_x_dim_list = [active_num * k, h]
    expanded_row_idx_dim_list = [n * k]
    expanded_expert_idx_dim_list = [n * k]
    return (x.new_empty(tuple(expanded_x_dim_list)), row_idx.new_empty(tuple(expanded_row_idx_dim_list)), row_idx.new_empty(tuple(expanded_row_idx_dim_list)))


@impl(m, "npu_moe_init_routing_v2")
def npu_moe_init_routing_v2_meta(x, expert_idx, *, scale=None, offset=None, active_num=-1, expert_capacity=-1, expert_num=-1, drop_pad_mode=0, expert_tokens_num_type=0, expert_tokens_num_flag=False, quant_mode=-1, active_expert_range=[], row_idx_type=0):
    x_dim = x.dim()
    torch._check(
        x_dim == 2,
        lambda: "the x shape support only 2d" + ops_error(ErrCode.VALUE),
    )
    expert_idx_dim = expert_idx.dim()
    torch._check(
        expert_idx_dim == 2,
        lambda: "the expert_idx shape support only 2d" + ops_error(ErrCode.VALUE),
    )
    torch._check(
        x.size(0) == expert_idx.size(0),
        lambda: "the first dim of expert_idx and x should be the same" + ops_error(ErrCode.VALUE),
    )
    if active_expert_range:
        torch._check(
            active_expert_range is not None and isinstance(active_expert_range, list) and len(active_expert_range) == 2,
            lambda: "active_expert_range is None or invalid. must be int[2]"
        )
        torch._check(
            active_expert_range[1] > active_expert_range[0],
            lambda: "active_expert_range is invalid. must be increasing"
        )
        torch._check(
            active_expert_range[0] >= 0 and active_expert_range[1] <= 10240,
            lambda: "active_expert_range must be within [0, 10240]"
        )
        expert_range_length = active_expert_range[1] - active_expert_range[0]
    else:
        expert_range_length = expert_num

    torch._check(
        drop_pad_mode is not None and isinstance(drop_pad_mode, int) and drop_pad_mode in [0, 1],
        lambda: "drop_pad_mode is None or invalid. must be in [0, 1]"
    )
    torch._check(
        expert_tokens_num_type is not None and isinstance(expert_tokens_num_type, int) and expert_tokens_num_type in [0, 1, 2],
        lambda: "expert_tokens_num_type is None or invalid. must be in [0, 1, 2]"
    )
    torch._check(
        expert_tokens_num_flag is not None and isinstance(expert_tokens_num_flag, bool) and expert_tokens_num_flag in [True, False],
        lambda: "expert_tokens_num_flag is None or invalid. must be in [True, False]"
    )
    torch._check(
        quant_mode is not None and isinstance(quant_mode, int) and quant_mode in [-1, 0, 1, 2, 3],
        lambda: "quant_mode is None or invalid. must be in [-1, 0, 1, 2, 3]"
    )
    torch._check(
        row_idx_type is not None and isinstance(row_idx_type, int) and row_idx_type in [0, 1],
        lambda: "row_idx_type is None or invalid. must be in [0, 1]"
    )

    if scale is not None:
        scale_dim = scale.dim()
        if quant_mode == -1:
            torch._check(
                scale_dim == 1,
                lambda: "the scale shape support only 1D (bs,) in no quant mode" + ops_error(ErrCode.VALUE),
            )
            torch._check(
                x.size(0) == scale.size(0),
                lambda: "the first dim of scale and the first dim of x should be the same" + ops_error(ErrCode.VALUE),
            )
        elif quant_mode == 0:
            torch._check(
                scale_dim == 1,
                lambda: "the scale shape support only 1D in static quant mode" + ops_error(ErrCode.VALUE),
            )
            torch._check(
                scale.size(0) == 1,
                lambda: "the shape of scale should be 1" + ops_error(ErrCode.VALUE),
            )
            if offset is not None:
                offset_dim = offset.dim()
                torch._check(
                    offset_dim == 1,
                    lambda: "the offset shape support only 1D" + ops_error(ErrCode.VALUE),
                )
                torch._check(
                    scale.size(0) == offset.size(0),
                    lambda: "the 1st dim of offset and the 1st dim of scale should be the same" + ops_error(ErrCode.VALUE),
                )
        elif quant_mode == 1:
            torch._check(
                scale_dim == 2,
                lambda: "the scale shape support only 2D in dynamic quant mode" + ops_error(ErrCode.VALUE),
            )
            torch._check(
                scale.size(0) in [expert_range_length, 1],
                lambda: "the first dim of scale must be in [expert_range_length, 1]" + ops_error(ErrCode.VALUE),
            )
            torch._check(
                x.size(1) == scale.size(1),
                lambda: "the 2nd dim of scale should be the same with the 2nd dim of x" + ops_error(ErrCode.VALUE),
            )
        # else: quant_mode为2、3时不使用scale也不需要校验

    bs = x.size(0)
    h = x.size(1)
    k = expert_idx.size(1)

    expanded_x_dtype = x.dtype
    expanded_scale_dtype = torch.float32
    if quant_mode in [0, 1]:
        expanded_x_dtype = torch.int8
    elif quant_mode == 2:
        expanded_x_dtype = torch.float8_e5m2
        expanded_scale_dtype = torch.float8_e8m0fnu
    elif quant_mode == 3:
        expanded_x_dtype = torch.float8_e4m3fn
        expanded_scale_dtype = torch.float8_e8m0fnu

    if drop_pad_mode == 1:
        expanded_x_dim_list = [expert_num, expert_capacity, h]
        expanded_scale_dim_list = [expert_num * expert_capacity]
    else:
        num_expanded_rows = bs * k if active_num <= 0 else min(active_num, bs * k)
        expanded_x_dim_list = [num_expanded_rows, h]
        if quant_mode in [2, 3]:
            MXQUANT_BLOCK_SIZE = 32
            PAD_TO_EVEN_FACTOR = 2
            scale_cols = (h + MXQUANT_BLOCK_SIZE - 1) // MXQUANT_BLOCK_SIZE
            scale_cols = (scale_cols + PAD_TO_EVEN_FACTOR - 1) // PAD_TO_EVEN_FACTOR * PAD_TO_EVEN_FACTOR
            expanded_scale_dim_list = [num_expanded_rows, scale_cols]
        elif quant_mode in [-1, 1]: # quant_mode in [-1, 0, 1]
            expanded_scale_dim_list = [num_expanded_rows]
    if quant_mode == 0:
        expanded_scale_dim_list = []

    expanded_row_idx_dim_list = [bs * k]

    if not expert_tokens_num_flag:
        expert_token_cumsum_or_count_dim_list = []
    elif (expert_tokens_num_type in range(0, 2)):   # [0, 1]
        expert_token_cumsum_or_count_dim_list = [expert_range_length]
    elif (expert_tokens_num_type == 2): # 2: key_value
        expert_token_cumsum_or_count_dim_list = [expert_num, 2]

    return (x.new_empty(tuple(expanded_x_dim_list), dtype=expanded_x_dtype),
            x.new_empty(tuple(expanded_row_idx_dim_list), dtype=torch.int32),
            x.new_empty(tuple(expert_token_cumsum_or_count_dim_list), dtype=torch.int64),
            x.new_empty(tuple(expanded_scale_dim_list), dtype=expanded_scale_dtype))


@impl(m, "ffn_worker_scheduler_")
def ffn_worker_scheduler__meta(self, *, sync_group_size=1, execute_mode=0):
    return self


@impl(m, "attention_worker_scheduler_")
def attention_worker_scheduler__meta(self):
    return self


@impl(m, "ffn_worker_scheduler")
def ffn_worker_scheduler_meta(self, *, sync_group_size=1, execute_mode=0):
    return torch.empty_like(self)


@impl(m, "attention_worker_scheduler")
def attention_worker_scheduler_meta(self):
    return torch.empty_like(self)


@impl(m, "npu_ffn_worker_batching")
def npu_ffn_worker_batching(schedule_context, expert_num, max_out_shape, *, token_dtype=0, need_schedule=0, layer_num=0):
    Y_size = max_out_shape[0] * max_out_shape[1] * max_out_shape[2]
    H_size = max_out_shape[3]
    H_dtype = torch.float16
    if token_dtype == 1:
        H_dtype = torch.bfloat16
    if token_dtype == 2:
        H_dtype = torch.int8
    return (torch.empty(Y_size, H_size, dtype=H_dtype, device=schedule_context.device),
            torch.empty(expert_num, 2, dtype=torch.int64, device=schedule_context.device),
            torch.empty(Y_size, dtype=torch.int32, device=schedule_context.device),
            torch.empty(Y_size, dtype=torch.int32, device=schedule_context.device),
            torch.empty(Y_size, dtype=torch.int32, device=schedule_context.device),
            torch.empty(Y_size, dtype=torch.int32, device=schedule_context.device),
            torch.empty(Y_size, dtype=torch.float32, device=schedule_context.device),
            torch.empty(1, dtype=torch.int64, device=schedule_context.device)
            )


@impl(m, "npu_moe_gating_top_k_softmax")
def npu_moe_gating_top_k_softmax_meta(x, finished=None, k=1):
    x_dim = x.dim()
    torch._check(
        x_dim == 2 or x_dim == 3,
        lambda: "the x shape support only 2d and 3d)" + ops_error(ErrCode.VALUE),
    )
    if x_dim == 3:
        y_dim_list = [x.size(0), x.size(1), k]
        expert_idx_dim_list = [x.size(0), x.size(1), k]
        row_idx_dim_list = [x.size(0), x.size(1), k]
    else:
        y_dim_list = [x.size(0), k]
        expert_idx_dim_list = [x.size(0), k]
        row_idx_dim_list = [x.size(0), k]
    return (x.new_empty(tuple(y_dim_list), dtype=x.dtype),
            x.new_empty(tuple(expert_idx_dim_list), dtype=torch.int32),
            x.new_empty(tuple(row_idx_dim_list), dtype=torch.int32))


@impl(m, "npu_moe_gating_top_k_softmax_v2")
def npu_moe_gating_top_k_softmax_v2_meta(x, *, k=1, finished=None, renorm=0, output_softmax=False):
    x_dim = x.dim()
    torch._check(
        x_dim == 2 or x_dim == 3,
        lambda: "the x shape support only 2d and 3d)" + ops_error(ErrCode.VALUE),
    )
    if x_dim == 3:
        y_dim_list = [x.size(0), x.size(1), k]
        expert_idx_dim_list = [x.size(0), x.size(1), k]
    else:
        y_dim_list = [x.size(0), k]
        expert_idx_dim_list = [x.size(0), k]

    if renorm == 0 and output_softmax:
        if x.dim == 3:
            softmax_result_dim_list = [x.size(0), x.size(1), x.size(2)]
        else:
            softmax_result_dim_list = [x.size(0), x.size(1)]
    else:
        softmax_result_dim_list = [0, ]

    return (x.new_empty(tuple(y_dim_list), dtype=x.dtype),
            x.new_empty(tuple(expert_idx_dim_list), dtype=torch.int32),
            x.new_empty(tuple(softmax_result_dim_list), dtype=torch.float32))


@impl(m, "npu_moe_gating_top_k")
def npu_moe_gating_top_k_meta(x, k=1, bias=None, k_group=1, group_count=1, group_select_mode=0, renorm=0, norm_type=0, out_flag=False, routed_scaling_factor=1.0, eps=1e-20):
    x_dim = x.dim()
    torch._check(
        x_dim == 2,
        lambda: "the x shape support only 2d)" + ops_error(ErrCode.VALUE),
    )
    if bias is not None:
        bias_dim = bias.dim()
        torch._check(
            bias_dim == 1,
            lambda: "the bias shape support only 1d)" + ops_error(ErrCode.VALUE),
        )
    y_dim_list = [x.size(0), k]
    expert_idx_dim_list = [x.size(0), k]
    y2_dim_list = [x.size(0), x.size(1)]
    return (x.new_empty(tuple(y_dim_list), dtype=x.dtype),
            x.new_empty(tuple(expert_idx_dim_list), dtype=torch.int32),
            x.new_empty(tuple(y2_dim_list), dtype=torch.float32))


def get_query_and_attention_out_layout(query, input_layout):
    class ParserLayout:
        def __init__(self, qLayout: str, outLayout: str, qDim: int):
            self.qLayout = qLayout
            self.outLayout = outLayout
            self.qDim = qDim

    LAYOUT_MAP: Dict[str, ParserLayout] = {
        "BSH": ParserLayout("BSH", "BSH", 3),
        "BSND": ParserLayout("BSND", "BSND", 4),
        "BNSD": ParserLayout("BNSD", "BNSD", 4),
        "TND": ParserLayout("TND", "TND", 3),
        "NTD": ParserLayout("NTD", "NTD", 3),
        "BNSD_BSND": ParserLayout("BNSD", "BSND", 4),
        "BSH_BNSD": ParserLayout("BSH", "BNSD", 3),
        "BSND_BNSD": ParserLayout("BSND", "BNSD", 4),
        "NTD_TND": ParserLayout("NTD", "TND", 3),
        "BSH_NBSD": ParserLayout("BSH", "NBSD", 3),
        "BSND_NBSD": ParserLayout("BSND", "NBSD", 4),
        "BNSD_NBSD": ParserLayout("BNSD", "NBSD", 4),
        "TND_NTD": ParserLayout("TND", "NTD", 3),
        "NSD": ParserLayout("NSD", "NSD", 3)
    }

    if input_layout in LAYOUT_MAP:
        layout_entry = LAYOUT_MAP[input_layout]

        query_layout = layout_entry.qLayout
        attention_out_layout = layout_entry.outLayout
        query_dim = layout_entry.qDim

        torch._check(
            query.dim() == query_dim,
            lambda: (
                f"Layout {query_layout}, queryDims({query.dim()}) must be {query_dim}!" + ops_error(ErrCode.VALUE)
            ),
        )
    else:
        torch._check(
            False,
            lambda: (
                f"Layout {input_layout} is not supported!" + ops_error(ErrCode.VALUE)
            ),
        )
    return query_layout, attention_out_layout


def get_query_b_n_s(query, query_layout, num_heads):
    if query_layout == "BSH":
        b = query.size(0)
        s1 = query.size(1)
        n1 = num_heads
    elif query_layout == "BSND":
        b = query.size(0)
        s1 = query.size(1)
        n1 = query.size(2)
    elif query_layout == "BNSD":
        b = query.size(0)
        s1 = query.size(2)
        n1 = query.size(1)
    elif query_layout == "NSD":
        b = 1
        s1 = query.size(1)
        n1 = query.size(0)
    else:
        torch._check(
            False,
            lambda: (
                f"Layout {query_layout} is not supported in get_query_b_n_s function!" + ops_error(ErrCode.VALUE)
            ),
        )
    return b, s1, n1


def get_query_t_n(query, query_layout):
    if query_layout == "TND":
        t = query.size(0)
        n1 = query.size(1)
    elif query_layout == "NTD":
        t = query.size(1)
        n1 = query.size(0)
    else:
        torch._check(
            False,
            lambda: (
                f"Layout {query_layout} is not supported in get_query_t_n function!" + ops_error(ErrCode.VALUE)
            ),
        )
    return t, n1


def get_value_d(block_table, value, query, query_layout, num_key_value_heads):
    if block_table is not None:
        if value.dim() == 3:
            value_d = value.size(2) // num_key_value_heads
        elif value.dim() == 4:
            value_d = value.size(3)
        elif value.dim() == 5:
            value_d = value.size(2) * value.size(4)
        else:
            torch._check(
                False,
                lambda: "when Page Attention enabled, value's dim should be 3/4/5, but got " + str(value.dim()) +
                ops_error(ErrCode.VALUE),
            )
    else:
        torch._check(
            value.dim() == query.dim(),
            lambda: (
                f"when Page Attention not enabled, value'dim{value.dim()} should equal to query's dim{query.dim()}!" +
                ops_error(ErrCode.VALUE)
            ),
        )
        if query_layout == "BSH":
            value_d = value.size(2) // num_key_value_heads
        if query_layout == "BNSD" or query_layout == "BSND":
            value_d = value.size(3)
        if query_layout == "TND" or query_layout == "NTD" or query_layout == "NSD":
            value_d = value.size(2)
    return value_d


def get_change_d_scale(value):
    change_d_scale = 1
    #int4伪装int32
    if value is not None and value.dtype == torch.int32:
        change_d_scale = 8

    return change_d_scale


def get_change_d_scale_v2(value, value_dtype):
    change_d_scale = 1

    if value is None:
        return change_d_scale
    #int4伪装int32
    if value.dtype == torch.int32:
        change_d_scale = 8
    # value_dtype float4_e2m1fn_x2 伪装 uint8
    if (hasattr(torch, 'float4_e2m1fn_x2') and value.dtype == torch.float4_e2m1fn_x2) or value_dtype == torch_npu.float4_e2m1fn_x2:
        change_d_scale = 2
    # value_dtype float4_e1m2fn_x2 伪装 uint8
    if (hasattr(torch, 'float4_e1m2fn_x2') and value.dtype == torch.float4_e1m2fn_x2) or value_dtype == torch_npu.float4_e1m2fn_x2:
        change_d_scale = 2

    return change_d_scale


def infer_attention_out_shape(attention_out_layout, query, query_layout, num_heads, value_d):
    attention_out = torch.empty_like(query, dtype=query.dtype, device='meta')
    if attention_out_layout == "BSH":
        b, s1, n1 = get_query_b_n_s(query, query_layout, num_heads)
        attention_out = torch.empty([b, s1, n1 * value_d], dtype=query.dtype, device='meta')
    elif attention_out_layout == "BSND":
        b, s1, n1 = get_query_b_n_s(query, query_layout, num_heads)
        attention_out = torch.empty([b, s1, n1, value_d], dtype=query.dtype, device='meta')
    elif attention_out_layout == "BNSD":
        b, s1, n1 = get_query_b_n_s(query, query_layout, num_heads)
        attention_out = torch.empty([b, n1, s1, value_d], dtype=query.dtype, device='meta')
    elif attention_out_layout == "NBSD":
        b, s1, n1 = get_query_b_n_s(query, query_layout, num_heads)
        attention_out = torch.empty([n1, b, s1, value_d], dtype=query.dtype, device='meta')
    elif attention_out_layout == "TND":
        t, n1 = get_query_t_n(query, query_layout)
        attention_out = torch.empty([t, n1, value_d], dtype=query.dtype, device='meta')
    elif attention_out_layout == "NTD":
        t, n1 = get_query_t_n(query, query_layout)
        attention_out = torch.empty([n1, t, value_d], dtype=query.dtype, device='meta')
    elif attention_out_layout == "NSD":
        _, s1, n1 = get_query_b_n_s(query, query_layout, num_heads)
        attention_out = torch.empty([n1, s1, value_d], dtype=query.dtype, device='meta')
    return attention_out


def infer_lse_out_shape(query, input_layout, query_layout, num_heads):
    lse_out = torch.empty([0], dtype=torch.float32, device='meta')

    tnd_like_layouts = {"TND", "NTD", "TND_NTD", "NTD_TND"}
    if input_layout in tnd_like_layouts:
        t, n1 = get_query_t_n(query, query_layout)
        lse_out = torch.empty([t, n1, 1], dtype=torch.float32, device='meta')
    else:
        b, s1, n1 = get_query_b_n_s(query, query_layout, num_heads)
        lse_out = torch.empty([b, n1, s1, 1], dtype=torch.float32, device='meta')
    return lse_out


@impl(m, "npu_fused_infer_attention_score")
def npu_fused_infer_attention_score_forward(query, key, value, *, pse_shift=None, atten_mask=None, actual_seq_lengths=None, actual_seq_lengths_kv=None,
                                    dequant_scale1=None, quant_scale1=None, dequant_scale2=None, quant_scale2=None,
                                    quant_offset2=None, antiquant_scale=None, antiquant_offset=None, block_table=None,
                                    query_padding_size=None, kv_padding_size=None, key_antiquant_scale=None, key_antiquant_offset=None,
                                    value_antiquant_scale=None, value_antiquant_offset=None, key_shared_prefix=None, value_shared_prefix=None,
                                    actual_shared_prefix_len=None, query_rope=None, key_rope=None, key_rope_antiquant_scale=None, num_heads=1, scale=1.0, pre_tokens=2147483647, next_tokens=2147483647,
                                    input_layout="BSH", num_key_value_heads=0, sparse_mode=0, inner_precise=0, block_size=0, antiquant_mode=0,
                                    softmax_lse_flag=False, key_antiquant_mode=0, value_antiquant_mode=0):
    torch._check(
        num_heads > 0,
        lambda: "numHeads should be greater than 0, but got " + str(num_heads) +
            ops_error(ErrCode.VALUE),
    )
    num_key_value_heads = num_heads if num_key_value_heads == 0 else num_key_value_heads

    # get query_layout, attention_out_layout
    query_layout, attention_out_layout = get_query_and_attention_out_layout(query, input_layout)

    # get value_d
    value_d = get_value_d(block_table, value, query, query_layout, num_key_value_heads)
    # 获取change_d_scale
    change_d_scale = get_change_d_scale(value)
    value_d = value_d * change_d_scale

    # infer attention out shape
    tmp_out = infer_attention_out_shape(attention_out_layout, query, query_layout, num_heads, value_d)

    # handle quant
    if quant_scale2 is not None:
        attention_out = torch.empty_like(tmp_out, dtype=torch.int8)
    elif query.dtype == torch.int8:
        if query_rope is not None:
            attention_out = torch.empty_like(tmp_out, dtype=query_rope.dtype)
        else:
            attention_out = torch.empty_like(tmp_out, dtype=torch.half)
    else:
        attention_out = torch.empty_like(tmp_out, dtype=query.dtype)

    # infer lse out shape
    tmp_lse_out = infer_lse_out_shape(query, input_layout, query_layout, num_heads)

    if softmax_lse_flag:
        lse_out = torch.empty_like(tmp_lse_out, dtype=torch.float32)
    else:
        lse_out = torch.empty([0], dtype=torch.float32, device='meta')

    return attention_out, lse_out


@impl(m, "npu_fused_infer_attention_score_v2")
def npu_fused_infer_attention_score_v2_forward(query, key, value, *, query_rope=None, key_rope=None, pse_shift=None, atten_mask=None, actual_seq_qlen=None, actual_seq_kvlen=None,
                                         block_table=None, dequant_scale_query=None, dequant_scale_key=None, dequant_offset_key=None, dequant_scale_value=None,
                                         dequant_offset_value=None, dequant_scale_key_rope=None, quant_scale_out=None, quant_offset_out=None, learnable_sink=None,
                                         num_query_heads=1, num_key_value_heads=0, softmax_scale=1.0, pre_tokens=2147483647, next_tokens=2147483647,
                                         input_layout="BSH", sparse_mode=0, block_size=0, query_quant_mode=0, key_quant_mode=0, value_quant_mode=0, inner_precise=0,
                                         return_softmax_lse=False, query_dtype=None, key_dtype=None, value_dtype=None, query_rope_dtype=None, key_rope_dtype=None,
                                         key_shared_prefix_dtype=None, value_shared_prefix_dtype=None, dequant_scale_query_dtype=None,
                                         dequant_scale_key_dtype=None, dequant_scale_value_dtype=None, dequant_scale_key_rope_dtype=None, out_dtype=None):
    torch._check(
        num_query_heads > 0,
        lambda: "numHeads should be greater than 0, but got " + str(num_query_heads) +
            ops_error(ErrCode.VALUE),
    )
    num_key_value_heads = num_query_heads if num_key_value_heads == 0 else num_key_value_heads

    # get query_layout, attention_out_layout
    query_layout, attention_out_layout = get_query_and_attention_out_layout(query, input_layout)

    # get value_d
    value_d = get_value_d(block_table, value, query, query_layout, num_key_value_heads)
    # 获取change_d_scale
    change_d_scale = get_change_d_scale_v2(value, value_dtype)
    value_d = value_d * change_d_scale

    # infer attention out shape
    tmp_out = infer_attention_out_shape(attention_out_layout, query, query_layout, num_query_heads, value_d)

    # input is hifloat8
    is_hifloat8_input = query.dtype == torch.uint8 and query_dtype is not None and query_dtype == torch_npu.hifloat8

    # handle quant
    if quant_scale_out is not None:
        output_type = torch.int8
        if out_dtype is not None:
            output_type = TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP[out_dtype]
        attention_out = torch.empty_like(tmp_out, dtype=output_type)
    elif query.dtype == torch.int8 or query.dtype == torch.float8_e4m3fn or is_hifloat8_input:
        if out_dtype is not None:
            output_type = TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP[out_dtype]
            attention_out = torch.empty_like(tmp_out, dtype=output_type)
        elif query_rope is not None:
            attention_out = torch.empty_like(tmp_out, dtype=query_rope.dtype)
        else:
            attention_out = torch.empty_like(tmp_out, dtype=torch.half)
    else:
        attention_out = torch.empty_like(tmp_out, dtype=query.dtype)

    # infer lse out shape
    tmp_lse_out = infer_lse_out_shape(query, input_layout, query_layout, num_query_heads)

    if return_softmax_lse:
        lse_out = torch.empty_like(tmp_lse_out, dtype=torch.float32)
    else:
        lse_out = torch.empty([0], dtype=torch.float32, device='meta')

    return attention_out, lse_out


@impl(m, "npu_quant_lightning_indexer")
def npu_quant_lightning_indexer_forward(query, key, weights, query_dequant_scale, key_dequant_scale, query_quant_mode, key_quant_mode, *, actual_seq_lengths_query=None,
                                        actual_seq_lengths_key=None, block_table=None, layout_query="BSND",
                                        layout_key="BSND", sparse_count=2048, sparse_mode=3, pre_tokens=9223372036854775807, next_tokens=9223372036854775807):
    require_param = {"query": query, "key": key, "weights": weights, "query_dequant_scale": query_dequant_scale, "key_dequant_scale": key_dequant_scale}

    for item_name, item in require_param.items():
        torch._check(
            item is not None,
            lambda: item_name + " should not be None, but the actual value is None" + ops_error(ErrCode.VALUE),
        )

    torch._check(
        query.numel() > 0,
        lambda: "Input query should not be empty." + ops_error(ErrCode.VALUE),
    )
    torch._check(
        key.numel() > 0,
        lambda: "Input key should not be empty." + ops_error(ErrCode.VALUE),
    )
    torch._check(
        sparse_count > 0,
        lambda: "sparse_count should be greater than 0, but got " + str(sparse_count) +
            ops_error(ErrCode.VALUE),
    )

    if layout_key == "TND":
        keyHeadNum = key.size(1)
    else:
        keyHeadNum = key.size(2)
    if layout_query == "BSND":
        out = torch.empty([query.size(0), query.size(1), keyHeadNum, sparse_count], dtype=torch.int32, device='meta')
    elif layout_query == "TND":
        out = torch.empty([query.size(0), keyHeadNum, sparse_count], dtype=torch.int32, device='meta')
    else:
        torch._check(
            False,
            lambda: "No support of query: " + str(layout_query) + ops_error(ErrCode.VALUE),
        )

    return out


@impl(m, "npu_kv_quant_sparse_flash_attention")
def npu_kv_quant_sparse_flash_attention_forward(query, key, value, sparse_indices, scale_value, key_quant_mode,
                                                value_quant_mode, *, key_dequant_scale=None, value_dequant_scale=None, block_table=None,
                                                actual_seq_lengths_query=None, actual_seq_lengths_kv=None, sparse_block_size=1, layout_query="BSND",
                                                layout_kv="BSND", sparse_mode=3, pre_tokens=9223372036854775807, next_tokens=9223372036854775807, attention_mode=0,
                                                quant_scale_repo_mode=1, tile_size=128, rope_head_dim=64):
    require_param = {"query": query, "key": key, "value": value, "sparse_indices": sparse_indices}

    for item_name, item in require_param.items():
        torch._check(
            item is not None,
            lambda: item_name + " should not be None, but the actual value is None" + ops_error(ErrCode.VALUE),
        )

    torch._check(
        query.numel() > 0,
        lambda: "Input query should not be empty." + ops_error(ErrCode.VALUE),
    )
    torch._check(
        key.numel() > 0,
        lambda: "Input key should not be empty." + ops_error(ErrCode.VALUE),
    )
    torch._check(
        value.numel() > 0,
        lambda: "Input value should not be empty." + ops_error(ErrCode.VALUE),
    )
    torch._check(
        sparse_indices.numel() > 0,
        lambda: "Input sparse_indices should not be empty." + ops_error(ErrCode.VALUE),
    )

    if layout_query == "BSND":
        torch._check(
            query.dim() == 4,
            lambda: "When the layout of query is BSND, the query dimension must be 4, but got " + str(query.dim()) + ops_error(ErrCode.VALUE),
        )
        out = torch.empty([query.size(0), query.size(1), query.size(2), query.size(3) - rope_head_dim], dtype=query.dtype, device='meta')
    elif layout_query == "TND":
        torch._check(
            query.dim() == 3,
            lambda: "When the layout of query is TND, the query dimension must be 3, but got " + str(query.dim()) + ops_error(ErrCode.VALUE),
        )
        out = torch.empty([query.size(0), query.size(1), query.size(2) - rope_head_dim], dtype=query.dtype, device='meta')
    else:
        torch._check(
            False,
            lambda: "Not support layout of query:" + layout_query + ops_error(ErrCode.VALUE),
        )
    return out


@impl(m, "npu_fusion_attention")
def npu_fusion_attention_forward(query, key, value, head_num, input_layout, pse=None, padding_mask=None,
                                atten_mask=None, scale=1.0, keep_prob=1.0, pre_tockens=2147483647, next_tockens=2147483647,
                                inner_precise=0, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0,
                                 gen_mask_parallel=True, sync=False, softmax_layout="", sink=None):
    B = query.size(0)
    N = head_num
    S1 = query.size(2)
    S2 = key.size(2)

    if input_layout == "BSH":
        B = query.size(0)
        S1 = query.size(1)
        S2 = key.size(1)

    if input_layout == "SBH":
        B = query.size(1)
        S1 = query.size(0)
        S2 = key.size(0)

    if input_layout == "BSND":
        S1 = query.size(1)
        S2 = key.size(1)

    seed = 0
    offset = 0
    numels = 0
    attention_score = query.new_empty(query.shape, dtype=query.dtype, device='meta')
    softmax_max = torch.empty([B, head_num, S1, 8], dtype=torch.float32, device='meta')
    softmax_sum = torch.empty([B, head_num, S1, 8], dtype=torch.float32, device='meta')
    softmax_out = torch.empty([0], dtype=query.dtype, device='meta')
    return (attention_score, softmax_max, softmax_sum, softmax_out, seed, offset, numels)


@impl(m, "npu_fusion_attention_grad")
def npu_fusion_attention_backward(query, key, value, dy, head_num, input_layout, *, pse=None, padding_mask=None, atten_mask=None,
                                  softmax_max=None, softmax_sum=None, softmax_in=None, attention_in=None, scale_value=1.0,
                                  keep_prob=1.0, pre_tockens=2147483647, next_tockens=2147483647, inner_precise=0, seed=0, offset=0,
                                  numels=0, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0,
                                  gen_mask_parallel=True, sync=False, softmax_layout="", sink=None):
    dq = query.new_empty(query.shape, dtype=query.dtype, device='meta')
    dk = key.new_empty(key.shape, dtype=query.dtype, device='meta')
    dv = value.new_empty(value.shape, dtype=query.dtype, device='meta')
    dpse = torch.empty([0], dtype=query.dtype, device='meta')
    dsink = torch.empty([], device='meta') if sink is None else torch.empty(sink.shape, dtype=sink.dtype, device='meta')
    return (dq, dk, dv, dpse, dsink)


@impl(m, "npu_quant_fusion_attention")
def npu_quant_fusion_attention_forward(query, key, value, head_num, input_layout, *, pse=None, padding_mask=None, atten_mask=None, query_rope=None,
                                key_rope=None, d_scale_q=None, d_scale_k=None, d_scale_v=None, scale=1.0, keep_prob=1.0,
                                pre_tokens=2147483647, next_tokens=2147483647, inner_precise=0, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None,
                                sparse_mode=0, out_dtype=None, gen_mask_parallel=True, sync=False, pse_type=1, q_start_idx=None, kv_start_idx=None,
                                softmax_layout="", sink=None, query_quant_mode=0, query_dtype=None):
    B = query.size(0)
    N = head_num
    S1 = query.size(2)
    S2 = key.size(2)

    if input_layout == "BSH":
        B = query.size(0)
        S1 = query.size(1)
        S2 = key.size(1)

    if input_layout == "SBH":
        B = query.size(1)
        S1 = query.size(0)
        S2 = key.size(0)

    seed = 0
    offset = 0
    numels = 0
    if out_dtype is not None and out_dtype == 1:
        attention_score = torch.empty_like(query, dtype=torch.bfloat16, device='meta')
        softmax_out = torch.empty([0], dtype=torch.bfloat16, device='meta')
    else:
        attention_score = torch.empty_like(query, dtype=torch.half, device='meta')
        softmax_out = torch.empty([0], dtype=torch.half, device='meta')
    softmax_max = torch.empty([B, head_num, S1, 8], dtype=torch.float32, device='meta')
    softmax_sum = torch.empty([B, head_num, S1, 8], dtype=torch.float32, device='meta')
    return (torch.empty_like(attention_score),
            torch.empty_like(softmax_max),
            torch.empty_like(softmax_sum),
            torch.empty_like(softmax_out),
            seed,
            offset,
            numels)


@impl(m, "npu_fusion_attention_v2")
def npu_fusion_attention_forward_v2(query, key, value, head_num, input_layout, *, pse=None, padding_mask=None, atten_mask=None, query_rope=None,
                                key_rope=None, scale=1.0, keep_prob=1.0,
                                pre_tokens=2147483647, next_tokens=2147483647, inner_precise=0, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None,
                                sparse_mode=0, gen_mask_parallel=True, sync=False, pse_type=1, q_start_idx=None, kv_start_idx=None,
                                softmax_layout="", sink=None):
    B = query.size(0)
    N = head_num
    S1 = query.size(2)
    S2 = key.size(2)

    if input_layout == "BSH":
        B = query.size(0)
        S1 = query.size(1)
        S2 = key.size(1)

    if input_layout == "SBH":
        B = query.size(1)
        S1 = query.size(0)
        S2 = key.size(0)

    seed = 0
    offset = 0
    numels = 0
    attention_score = torch.empty_like(query, dtype=query.dtype, device='meta')
    softmax_max = torch.empty([B, head_num, S1, 8], dtype=torch.float32, device='meta')
    softmax_sum = torch.empty([B, head_num, S1, 8], dtype=torch.float32, device='meta')
    softmax_out = torch.empty([0], dtype=query.dtype, device='meta')
    return (torch.empty_like(attention_score),
            torch.empty_like(softmax_max),
            torch.empty_like(softmax_sum),
            torch.empty_like(softmax_out),
            seed,
            offset,
            numels)


@impl(m, "npu_fused_floyd_attention")
def npu_fused_floyd_attention(query_ik, key_ij, value_ij, key_jk, value_jk, *, atten_mask=None, scale_value=1.):
    out0_out1_shape = (query_ik.shape[0], query_ik.shape[1], query_ik.shape[2], query_ik.shape[3], 8)
    out0 = torch.empty(out0_out1_shape, dtype=torch.float32, device='meta')
    out1 = torch.empty_like(out0, device='meta')
    out2 = torch.empty_like(query_ik, device='meta')
    return (out0, out1, out2)


@impl(m, "npu_fused_floyd_attention_backward")
def npu_fused_floyd_attention_backward(grad_output, query_ik, key_ij, value_ij, key_jk, value_jk, attention_out, softmax_max, softmax_sum, *, atten_mask=None, scale_value=1.):
    dquery = torch.empty_like(query_ik, device='meta')
    dkey_0 = torch.empty_like(key_ij, device='meta')
    dvalue_0 = torch.empty_like(value_ij, device='meta')
    dkey_1 = torch.empty_like(key_jk, device='meta')
    dvalue_1 = torch.empty_like(value_jk, device='meta')
    return (dquery, dkey_0, dvalue_0, dkey_1, dvalue_1)


@impl(m, "npu_lightning_indexer")
def npu_lightning_indexer_forward(query, key, weights, *, actual_seq_lengths_query=None,
    actual_seq_lengths_key=None, block_table=None, layout_query="BSND", layout_key="BSND", sparse_count=2048, sparse_mode=3,
    pre_tokens=9223372036854775807, next_tokens=9223372036854775807, return_value=False):
    require_param = {"query": query, "key": key, "weights": weights}

    for item_name, item in require_param.items():
        torch._check(
            item is not None,
            lambda: item_name + " should not be None, but the actual value is None" + ops_error(ErrCode.VALUE),
        )

    torch._check(
        query.numel() > 0,
        lambda: "Input query should not be empty." + ops_error(ErrCode.VALUE),
    )
    torch._check(
        key.numel() > 0,
        lambda: "Input key should not be empty." + ops_error(ErrCode.VALUE),
    )
    torch._check(
        sparse_count > 0,
        lambda: "sparse_count should be greater than 0, but got " + str(sparse_count) +
            ops_error(ErrCode.VALUE),
    )
    torch._check(
        not return_value,
        lambda: "when return_value is true, not support pytorch compile." + ops_error(ErrCode.VALUE),
    )

    if layout_query == "BSND":
        sparse_indices_out = torch.empty([query.size(0), query.size(1), key.size(2), sparse_count], dtype=torch.int32, device='meta')
    else:
        if layout_key == "TND":
            n_dim_idx = 1
        else:
            n_dim_idx = 2
        sparse_indices_out = torch.empty([query.size(0), key.size(n_dim_idx), sparse_count], dtype=torch.int32, device='meta')
    if return_value:
        if layout_query == "BSND":
            sparse_values_out = torch.empty([query.size(0), query.size(1), key.size(2), sparse_count], dtype=query.dtype, device='meta')
        else:
            if layout_key == "TND":
                n_dim_idx = 1
            else:
                n_dim_idx = 2
            sparse_values_out = torch.empty([query.size(0), key.size(n_dim_idx), sparse_count], dtype=query.dtype, device='meta')
    else:
        sparse_values_out = torch.empty([0], dtype=query.dtype, device='meta')
    return (sparse_indices_out, sparse_values_out)


@impl(m, "npu_lightning_indexer_grad")
def npu_lightning_indexer_grad_meta(query, key, dy, sparse_indices, weights, actual_seq_lengths_query=None, actual_seq_lengths_key=None, layout="BSND", sparse_mode=3, pre_tokens=9223372036854775807, next_tokens=9223372036854775807):
    d_query = query.new_empty(query.shape, dtype=query.dtype, device='meta')
    d_key = key.new_empty(key.shape, dtype=key.dtype, device='meta')
    d_weights = weights.new_empty(weights.shape, dtype=weights.dtype, device='meta')
    return (d_query, d_key, d_weights)


@impl(m, "npu_sparse_lightning_indexer_grad_kl_loss")
def npu_sparse_lightning_indexer_grad_kl_loss_meta(query, key, query_index, key_index, weights, sparse_indices, softmax_max, softmax_sum, scale_value, *, query_rope=None, key_rope=None, actual_seq_qlen=None, actual_seq_klen=None, layout='BSND', sparse_mode=3, pre_tokens=9223372036854775807, next_tokens=9223372036854775807):
    d_query_index = query_index.new_empty(query_index.shape, dtype=query_index.dtype, device='meta')
    d_key_index = key_index.new_empty(key_index.shape, dtype=key_index.dtype, device='meta')
    d_weights = weights.new_empty(weights.shape, dtype=weights.dtype, device='meta')
    loss = torch.empty([1], dtype=torch.float32, device='meta')
    return (d_query_index, d_key_index, d_weights, loss)


@impl(m, "npu_quant_fusion_attention_grad")
def npu_quant_fusion_attention_backward(query, key, value, dy, head_num, input_layout, *, pse=None, padding_mask=None, atten_mask=None,
                                  d_scale_q=None, d_scale_k=None, d_scale_v=None, softmax_max=None,
                                  softmax_sum=None, softmax_in=None, attention_in=None, query_rope=None, key_rope=None, scale_value=1.0,
                                  keep_prob=1.0, pre_tokens=2147483647, next_tokens=2147483647, inner_precise=0, seed=0, offset=0,
                                  numels=0, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0, out_dtype=None,
                                  gen_mask_parallel=True, sync=False, pse_type=1, q_start_idx=None, kv_start_idx=None,
                                  softmax_layout="", sink=None, query_quant_mode=0, query_dtype=None):
    if out_dtype is not None and out_dtype == 1:
        dq = torch.empty_like(query, dtype=torch.bfloat16, device='meta')
        dq_rope = torch.empty_like([0], dtype=torch.bfloat16, device='meta')
        dk = torch.empty_like(key, dtype=torch.bfloat16, device='meta')
        dk_rope = torch.empty_like([0], dtype=torch.bfloat16, device='meta')
        dv = torch.empty_like(value, dtype=torch.bfloat16, device='meta')
        dpse = torch.empty_like([0], dtype=torch.bfloat16, device='meta')
    else:
        dq = torch.empty_like(query, dtype=torch.half, device='meta')
        dq_rope = torch.empty_like([0], dtype=torch.half, device='meta')
        dk = torch.empty_like(key, dtype=torch.half, device='meta')
        dk_rope = torch.empty_like([0], dtype=torch.half, device='meta')
        dv = torch.empty_like(value, dtype=torch.half, device='meta')
        dpse = torch.empty_like([0], dtype=torch.half, device='meta')
    dsink = None if sink is None else torch.empty_like(sink, dtype=sink.dtype, device='meta')
    return (dq, dk, dv, dpse, dq_rope, dk_rope, dsink)


@impl(m, "npu_fusion_attention_grad_v2")
def npu_fusion_attention_backward_v2(query, key, value, dy, head_num, input_layout, *, pse=None, padding_mask=None, atten_mask=None,
                                  softmax_max=None,
                                  softmax_sum=None, softmax_in=None, attention_in=None, query_rope=None, key_rope=None, scale_value=1.0,
                                  keep_prob=1.0, pre_tokens=2147483647, next_tokens=2147483647, inner_precise=0, seed=0, offset=0,
                                  numels=0, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0,
                                  gen_mask_parallel=True, sync=False, pse_type=1, q_start_idx=None, kv_start_idx=None,
                                  softmax_layout="", sink=None):
    dq = torch.empty_like(query, dtype=query.dtype, device='meta')
    dq_rope = torch.empty_like([0], dtype=query.dtype, device='meta')
    dk = torch.empty_like(key, dtype=query.dtype, device='meta')
    dk_rope = torch.empty_like([0], dtype=query.dtype, device='meta')
    dv = torch.empty_like(value, dtype=query.dtype, device='meta')
    dpse = torch.empty_like([0], dtype=query.dtype, device='meta')
    dsink = None if sink is None else torch.empty_like(sink, dtype=sink.dtype, device='meta')
    return (dq, dk, dv, dpse, dq_rope, dk_rope, dsink)


@impl(m, "npu_rotary_mul")
def npu_rotary_mul_meta(embedding, cosine, sine, mode='half', rotate=None):
    return torch.empty_like(embedding)


@impl(m, "npu_rotary_mul_backward")
def npu_rotary_mul_backward(grad, embedding, cosine, sine, mode=0):
    dx = torch.empty_like(embedding, dtype=embedding.dtype, device='meta')
    dr1 = torch.empty_like(cosine, dtype=embedding.dtype, device='meta')
    dr2 = torch.empty_like(sine, dtype=embedding.dtype, device='meta')
    return (dx, dr1, dr2)


@impl(m, "fast_gelu")
def fast_gelu_meta(self):
    return torch.empty_like(self)


@impl(m, "npu_fast_gelu_backward")
def npu_fast_gelu_backward_meta(grad, self):
    return torch.empty_like(self)


@impl(m, "npu_fast_gelu")
def npu_fast_gelu_meta(self):
    return torch.empty_like(self)


@impl(m, "npu_gelu")
def npu_gelu_meta(self, *, approximate="none"):
    return torch.empty_like(self)


@impl(m, "npu_gelu_backward")
def npu_gelu_backward_meta(grad, self, *, approximate="none"):
    return torch.empty_like(self)


if _is_pytorch_version_ge("2.6.0"):
    @impl(m, "npu_gelu_mul")
    def npu_gelu_mul_meta(input_tensor, *, approximate="none"):
        output_shape = list(input_tensor.shape)
        last_dim = input_tensor.shape[-1]
        output_shape[-1] = last_dim // 2
        output_shape = tuple(output_shape)
        output_dtype = input_tensor.dtype
        return torch.empty(size=output_shape, dtype=output_dtype, device=torch.device("meta"))


@impl(m, "npu_gelu_quant")
def npu_gelu_quant_meta(self, *, input_scale=None, input_offset=None,
                        approximate="none", quant_mode="dynamic", dst_type=1, round_mode='rint'):
    if not (quant_mode == "dynamic" or quant_mode == "static"):
        raise RuntimeError("Parameter(quant_mode) must be 'dynamic' or 'static', got " + quant_mode + ops_error(ErrCode.VALUE))

    out_scale = None
    if quant_mode == "static":
        if input_scale is None:
            raise RuntimeError("input_scale cannot be None when quant_mode is 'static'.")
    else:
        # infer out_scale shape
        out_scale_shape = self.shape[:-1]
        out_scale = self.new_empty(out_scale_shape, dtype=torch.float32)

    y_dst_dtype = TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP.get(dst_type)
    if dst_type is not None:
        y = torch.empty_like(self, dtype=y_dst_dtype)
    else:
        raise RuntimeError("Parameter(dst_type) enum value:{} not found in " \
                            "TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP, please check.".format(dst_type) +
                            ops_error(ErrCode.PARAM))
    return (y, out_scale)


@impl(m, "npu_dtype_cast")
def npu_dtype_cast_meta(self, dtype, input_dtype=None):
    dim_num = self.dim()
    input_shape = []
    for dim in range(dim_num):
        input_shape.append(self.size(dim))

    if input_dtype == 296 or input_dtype == 297:
        if dim_num != 0:
            input_shape[-1] *= 2
        else:
            raise RuntimeError("Scalar input cannot be float4_e2m1fn_x2 or float4_e1m2fn_x2" +
                               ops_error(ErrCode.PARAM))

    if dtype == 285 or dtype == 296 or dtype == 297:
        if dim_num == 0 or input_shape[-1] % 2:
            raise RuntimeError("If output dtype is float4_e2m1fn_x2, float4_e1m2fn_x2 or int4, " \
                                "the last dim of input must be divisible by 2" +
                               ops_error(ErrCode.PARAM))
        input_shape[-1] //= 2
    # torch_npu.hifloat8, torch_npu.float4_e2m1fn_x2, torch_npu.float4_e1m2fn_x2, torch_npu.int4
    if dtype in [285, 290, 296, 297]:
        output = self.new_empty(input_shape, dtype=torch.uint8)
    else:
        output_dst_dtype = TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP.get(dtype)
        if output_dst_dtype is not None:
            output = self.new_empty(input_shape, dtype=output_dst_dtype)
        else:
            raise RuntimeError("Parameter(dtype) enum value:{} not found in " \
                "TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP, please check.".format(dtype) +
                ops_error(ErrCode.PARAM))
    return output


@impl(m, "_npu_dtype_cast")
def _npu_dtype_cast_meta(self, dtype):
    return torch.empty_like(self, dtype=dtype)


@impl(m, "_npu_dtype_cast_backward")
def _npu_dtype_cast_backward_meta(self, dtype):
    return torch.empty_like(self, dtype=dtype)


@impl(m, "npu_dtype_cast_backward")
def npu_dtype_cast_backward_meta(self, dtype, grad_dtype=None, input_dtype=None):
    dim_num = self.dim()
    input_shape = []
    for dim in range(dim_num):
        input_shape.append(self.size(dim))

    if grad_dtype == 296 or grad_dtype == 297:
        if dim_num != 0:
            input_shape[-1] *= 2
        else:
            raise RuntimeError("Scalar input cannot be float4_e2m1 or float4_e1m2" +
                               ops_error(ErrCode.PARAM))

    if input_dtype == 296 or input_dtype == 297:
        if dim_num == 0 or input_shape[-1] % 2:
            raise RuntimeError("If output dtype is float4_e2m1 or float4_e1m2, " \
                                "the last dim of input must be divisible by 2" +
                               ops_error(ErrCode.PARAM))
        input_shape[-1] //= 2
    # torch_npu.hifloat8, torch_npu.float4_e2m1, torch_npu.float4_e1m2
    if input_dtype in [290, 296, 297]:
        output = self.new_empty(input_shape, dtype=torch.uint8)
    else:
        output = self.new_empty(input_shape, dtype=dtype)
    return output


@impl(m, "npu_bmmV2")
def npu_bmmV2_meta(self, mat2, output_sizes):
    dim1 = self.size(0)
    dim2 = self.size(1)
    dim3 = mat2.size(2)
    return self.new_empty((dim1, dim2, dim3))


@impl(m, "npu_transpose")
def npu_transpose_meta(self, perm, require_contiguous=True):
    output = self.permute(perm)
    return torch.empty_like(output, dtype=self.dtype)


@impl(m, "npu_deep_norm")
def npu_deep_norm_meta(self, gx, beta, gamma, alpha=0.3, epsilon=1e-6):
    rstd_dim = self.dim() - gamma.dim()
    ret = []
    for i in range(self.dim()):
        if i < rstd_dim:
            ret.append(self.size(i))
        else:
            ret.append(1)
    rstd = torch.empty(ret, dtype=torch.float32, device='meta')
    return (torch.empty_like(rstd), torch.empty_like(rstd), torch.empty_like(self, dtype=self.dtype))


@impl(m, "npu_rms_norm")
def npu_rms_norm_meta(self, gamma, epsilon=1e-6):
    rstd_dim = self.dim() - gamma.dim()
    ret = []
    for i in range(self.dim()):
        if i < rstd_dim:
            ret.append(self.size(i))
        else:
            ret.append(1)
    rstd = torch.empty(ret, dtype=torch.float32, device='meta')
    return (torch.empty_like(self, dtype=self.dtype), torch.empty_like(rstd))


@impl(m, "npu_gemma_rms_norm")
def npu_gemma_rms_norm_meta(self, gamma, epsilon=1e-6):
    rstd_dim = self.dim() - gamma.dim()
    ret = []
    for i in range(self.dim()):
        if i < rstd_dim:
            ret.append(self.size(i))
        else:
            ret.append(1)
    rstd = torch.empty(ret, dtype=torch.float32, device='meta')
    return (torch.empty_like(self, dtype=self.dtype), torch.empty_like(rstd))


@impl(m, "npu_add_rms_norm")
def npu_add_rms_norm_meta(x1, x2, gamma, epsilon=1e-6):
    rstd_dim = x1.dim() - gamma.dim()
    ret = []
    for i in range(x1.dim()):
        if i < rstd_dim:
            ret.append(x1.size(i))
        else:
            ret.append(1)
    rstd = torch.empty(ret, dtype=torch.float32, device='meta')
    return (torch.empty_like(x1, dtype=x1.dtype), torch.empty_like(rstd), torch.empty_like(x1, dtype=x1.dtype))


@impl(m, "npu_add_rms_norm_v2")
def npu_add_rms_norm_v2_meta(x1, x2, gamma, epsilon=1e-6):
    rstd_dim = x1.dim() - gamma.dim()
    ret = []
    for i in range(x1.dim()):
        if i < rstd_dim:
            ret.append(x1.size(i))
        else:
            ret.append(1)
    rstd = torch.empty(ret, dtype=torch.float32, device='meta')
    return torch.empty_like(rstd)


@impl(m, "npu_add_rms_norm_v2_functional")
def npu_add_rms_norm_v2_functional_meta(x1, x2, gamma, epsilon=1e-6):
    rstd_dim = x1.dim() - gamma.dim()
    ret = []
    for i in range(x1.dim()):
        if i < rstd_dim:
            ret.append(x1.size(i))
        else:
            ret.append(1)
    rstd = torch.empty(ret, dtype=torch.float32, device='meta')
    return (torch.empty_like(rstd), torch.empty_like(x1), torch.empty_like(x2))


@impl(m, "npu_rms_norm_quant")
def npu_rms_norm_quant_meta(x, gamma, beta, scale, offset, epsilon=1e-06):
    return torch.empty(x.size(), dtype=torch.int8, device=x.device)


@impl(m, "npu_add_rms_norm_cast")
def npu_add_rms_norm_cast_meta(x1, x2, gamma, epsilon=1e-6):
    rstd_dim = x1.dim() - gamma.dim()
    ret = []
    for i in range(x1.dim()):
        if i < rstd_dim:
            ret.append(x1.size(i))
        else:
            ret.append(1)
    rstd = torch.empty(ret, dtype=torch.float32, device='meta')
    return (torch.empty_like(x1, dtype=torch.float32), torch.empty_like(x1, dtype=x1.dtype), torch.empty_like(rstd), torch.empty_like(x1, dtype=x1.dtype))


@impl(m, "npu_add_rms_norm_dynamic_quant")
def npu_add_rms_norm_dynamic_quant_meta(x1, x2, gamma, *, smooth_scale1=None, smooth_scale2=None, beta=None, epsilon=1e-6, output_mask=None):
    return (torch.empty(x1.size(), dtype=torch.int8, device=x1.device),
            torch.empty(x1.size(), dtype=torch.int8, device=x1.device),
            torch.empty(x1.size(), dtype=x1.dtype, device=x1.device),
            torch.empty(x1.size()[:-1], dtype=torch.float32, device=x1.device),
            torch.empty(x1.size()[:-1], dtype=torch.float32, device=x1.device))


@impl(m, "npu_rms_norm_backward")
def npu_rms_norm_backward_meta(dy, self, gamma, rstd):
    return (torch.empty_like(self, dtype=self.dtype), torch.empty_like(gamma, dtype=gamma.dtype))


@impl(m, "scatter_update")
def scatter_update_meta(self, indices, updates, axis):
    return torch.empty_like(self)


@impl(m, "scatter_update_")
def scatter_update__meta(self, indices, updates, axis):
    return self


@impl(m, "_npu_dropout")
def _npu_dropout_meta(self, p):
    mask = math.floor(math.floor((self.numel() + BIT_NUMBER - 1) / BIT_NUMBER) * BIT_NUMBER / UINT8_BIT_NUMBER)
    return (torch.empty_like(self, dtype=self.dtype), torch.empty(mask, dtype=torch.uint8, device='meta'))


@impl(m, "npu_quant_scatter")
def npu_quant_scatter_meta(self, indices, updates, quant_scales, quant_zero_points=None, axis=-2, quant_axis=-1,
                           reduce='update', dst_type=1, round_mode='rint'):
    return torch.empty_like(self)


@impl(m, "npu_quant_scatter_")
def npu_quant_scatter__meta(self, indices, updates, quant_scales, quant_zero_points=None, axis=-2, quant_axis=-1,
                            reduce='update', dst_type=1, round_mode='rint'):
    return self


@impl(m, "npu_scatter_list_")
def scatter_list__meta(self, indices, updates, mask, reduce='update', axis=-2):
    return self


@impl(m, "npu_scatter_list")
def scatter_list_meta(self, indices, updates, mask, reduce='update', axis=-2):
    var_list = []
    for item in self:
        var_list.append(torch.empty_like(item))
    return var_list


@impl(m, "npu_scatter_nd_update")
def scatter_nd_update_meta(self, indices, updates):
    return torch.empty_like(self, dtype=self.dtype)


@impl(m, "npu_scatter_nd_update_")
def scatter_nd_update__meta(self, indices, updates):
    return self


@impl(m, "npu_scatter_pa_kv_cache_functional")
def npu_scatter_pa_kv_cache_functional_meta(key, value, key_cache, value_cache, slot_mapping, *, compress_lens=None,
    compress_seq_offsets=None, seq_lens=None):
    return (torch.empty_like(key_cache, dtype=key_cache.dtype), torch.empty_like(value_cache, dtype=value_cache.dtype))


@impl(m, "npu_scatter_pa_kv_cache")
def npu_scatter_pa_kv_cache_meta(key, value, key_cache, value_cache, slot_mapping, *, compress_lens=None,
    compress_seq_offsets=None, seq_lens=None):
    return


@impl(m, "npu_geglu")
def npu_geglu_meta(self, dim=-1, approximate=1, activate_left=False):
    dim_num = self.dim()
    input_shape = list(self.shape)

    if dim_num < 1 or dim_num > 8:
        raise RuntimeError("dim num out of range [1, 8]" + ops_error(ErrCode.PARAM))

    if dim >= dim_num or dim < -dim_num:
        raise RuntimeError("attribute [dim] out of range [-" + str(dim_num) + ", " + str(dim_num - 1) + "]" + ops_error(ErrCode.VALUE))

    if input_shape[dim] % 2 == 1:
        raise RuntimeError("x shape: " + str(input_shape) + ". Dim [" + str(dim) + "] of x should be divisible by 2, but get [" + str(input_shape[dim]) + "]" + ops_error(ErrCode.PARAM))

    input_shape[dim] //= 2
    return (self.new_empty(input_shape, dtype=self.dtype), self.new_empty(input_shape, dtype=self.dtype))


@impl(m, "npu_geglu_grad")
def npu_geglu_backward_meta(grad_output, self, gelu, dim, approximate, activate_left=False):
    return (torch.empty_like(self, dtype=self.dtype), torch.empty_like(self, dtype=self.dtype))


@impl(m, "npu_dropout_backward")
def npu_dropout_backward_meta(grad_output, mask, p):
    return torch.empty_like(grad_output, dtype=grad_output.dtype)


@impl(m, "npu_masked_softmax_with_rel_pos_bias")
def npu_masked_softmax_with_rel_pos_bias_meta(x, atten_mask, relative_pos_bias, scale_value=1.0, inner_precision_mode=0):
    return torch.empty_like(x, dtype=x.dtype)


@impl(m, "npu_moe_distribute_dispatch")
def npu_moe_distribute_dispatch_meta(x, expert_ids, group_ep, ep_world_size, ep_rank_id, moe_expert_num, scales=None, x_active_mask=None, expert_scales=None, group_tp="", tp_world_size=0,
                                     tp_rank_id=0, expert_shard_type=0, shared_expert_num=1, shared_expert_rank_num=0, quant_mode=0, global_bs=0, expert_token_nums_type=1):
    n = x.size(0)
    h = x.size(1)
    k = expert_ids.size(1)

    shared_front = 0
    outDtype = x.dtype
    if expert_shard_type == 0:
        shared_front = 1

    local_moe_expert_num = 0
    global_bs_real = 0
    if global_bs == 0:
        global_bs_real = n * ep_world_size
    else:
        global_bs_real = global_bs
    a = 0
    if shared_front == 1:
        if ep_rank_id < shared_expert_rank_num:
            local_moe_expert_num = 1
            a = global_bs_real // shared_expert_rank_num
        else:
            local_moe_expert_num = moe_expert_num // (ep_world_size - shared_expert_rank_num)
            a = global_bs_real * min(local_moe_expert_num, k)
    else:
        if ep_rank_id >= ep_world_size - shared_expert_rank_num:
            local_moe_expert_num = 1
            a = global_bs_real // shared_expert_rank_num
        else:
            local_moe_expert_num = moe_expert_num // (ep_world_size - shared_expert_rank_num)
            a = global_bs_real * min(local_moe_expert_num, k)
    ep_recv_cnt_num = 0
    if tp_world_size == 2:
        ep_recv_cnt_num = ep_world_size * local_moe_expert_num * tp_world_size
    else:
        ep_recv_cnt_num = ep_world_size * local_moe_expert_num

    if scales is not None or quant_mode != 0:
        outDtype = torch.int8
    local_moe_expert_num = int(local_moe_expert_num)

    expand_idx = x.new_empty(tuple([n * k]), dtype=torch.int32)
    if tp_world_size == 0:
        expand_x = x.new_empty(tuple([a, h]), dtype=outDtype)
        dynamic_scales = x.new_empty(tuple([a]), dtype=torch.float32)
    else:
        expand_x = x.new_empty(tuple([a * tp_world_size, h]), dtype=outDtype)
        dynamic_scales = x.new_empty(tuple([a * tp_world_size]), dtype=torch.float32)
    expert_token_nums = x.new_empty(tuple([local_moe_expert_num]), dtype=torch.int64)
    ep_recv_counts = x.new_empty(tuple([ep_recv_cnt_num]), dtype=torch.int32)
    tp_recv_counts = x.new_empty(tuple([tp_world_size]), dtype=torch.int32)
    expand_scales = x.new_empty(tuple([0]), dtype=torch.float32)
    if expert_scales is not None:
        ep_recv_cnt_num = ep_world_size * local_moe_expert_num + global_bs_real * 2 * k * (ep_world_size // 8)
        ep_recv_counts = x.new_empty(tuple([ep_recv_cnt_num]), dtype=torch.int32)
        expand_scales = x.new_empty(tuple([a]), dtype=torch.float32)
    return (expand_x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts, expand_scales)


def get_dispatch_dynamic_scales_dtype(x, scales, quant_mode):
    dynamic_scales_dtype = torch.float32
    if quant_mode == 0:
        if x.dtype != torch.bfloat16 and x.dtype != torch.float16 and scales is not None:
            dynamic_scales_dtype = scales.dtype
    elif quant_mode == 4:
        dynamic_scales_dtype = torch.uint8  # float8_e8m0
    return dynamic_scales_dtype


def get_dispatch_dynamic_shape(scales, quant_mode, a, h):
    shape = tuple([a])
    if quant_mode == 0 and scales is not None:
        if scales.dim() < 2:
            raise RuntimeError(f"Expected scales to be at least 2-d, but got {scales.dim()}-d.")
        shape = tuple([a * scales.shape[1]])
    elif quant_mode == 2:
        shape = tuple([a])
    elif quant_mode == 3:
        shape = tuple([a, math.ceil(h / 128)])
    elif quant_mode == 4:
        shape = tuple([a, (math.ceil(h / 32) + 1) // 2 * 2])
    return shape


@impl(m, "npu_moe_distribute_dispatch_v2")
def npu_moe_distribute_dispatch_v2_meta(x, expert_ids, group_ep, ep_world_size, ep_rank_id, moe_expert_num, scales=None, x_active_mask=None, expert_scales=None, elastic_info=None, performance_info=None, group_tp="", tp_world_size=0,
                                        tp_rank_id=0, expert_shard_type=0, shared_expert_num=1, shared_expert_rank_num=0, quant_mode=0, global_bs=0, expert_token_nums_type=1, comm_alg="",
                                        zero_expert_num=0, copy_expert_num=0, const_expert_num=0, y_dtype=None, x_dtype=None, scales_dtype=None):
    torch._check(
        (ep_rank_id >= 0) and (ep_rank_id < ep_world_size),
        lambda: (
            f"ep_rank_id should be in [0, ep_world_size), "
            f"but got {ep_world_size=}, {ep_rank_id=}."
            f"{ops_error(ErrCode.VALUE)}."
        ),
    )
    torch._check(
        (shared_expert_rank_num >= 0) and (shared_expert_rank_num < ep_world_size),
        lambda: (
            f"shared_expert_rank_num should be in [0, ep_world_size), "
            f"but got {ep_world_size=}, {shared_expert_rank_num=}."
            f"{ops_error(ErrCode.VALUE)}."
        ),
    )
    is_shared_default = ((shared_expert_num == 1) and (shared_expert_rank_num == 0))
    is_no_shared = ((shared_expert_num == 0) and (shared_expert_rank_num == 0))
    is_valid_shared = (
        (shared_expert_num > 0)
        and ((shared_expert_rank_num // shared_expert_num) > 0)
        and ((shared_expert_rank_num % shared_expert_num) == 0)
    )
    torch._check(
        is_shared_default or is_no_shared or is_valid_shared,
        lambda: (
            f"shared expert setting invalid, "
            f"got {shared_expert_num=}, {shared_expert_rank_num=}."
            f"{ops_error(ErrCode.VALUE)}."
        ),
    )
    torch._check(
        expert_token_nums_type in [0, 1],
        lambda: "the expert_token_nums_type should be 0 or 1" + ops_error(ErrCode.VALUE)
    )

    bs = x.size(0)
    h = x.size(1)
    k = expert_ids.size(1)

    shared_front = (expert_shard_type == 0)
    outDtype = torch.int8

    local_moe_expert_num = 1
    global_bs_real = 0
    if global_bs == 0:
        global_bs_real = bs * ep_world_size
    else:
        global_bs_real = global_bs
    a = 0
    if shared_front:
        if ep_rank_id < shared_expert_rank_num:
            local_moe_expert_num = 1
            max_bs = global_bs_real // ep_world_size
            rank_num_per_shared_expert = shared_expert_rank_num // shared_expert_num
            max_shared_group_num = (ep_world_size + rank_num_per_shared_expert - 1) // rank_num_per_shared_expert
            a = max_bs * max_shared_group_num
        else:
            local_moe_expert_num = moe_expert_num // (ep_world_size - shared_expert_rank_num)
            a = global_bs_real * min(local_moe_expert_num, k)
        if elastic_info is not None:
            if ((is_shared_default) or (is_no_shared)):
                local_moe_expert_num = max(local_moe_expert_num, moe_expert_num // (ep_world_size - shared_expert_rank_num))
                a = global_bs_real * min(local_moe_expert_num, k)
            else:
                max_bs = global_bs_real // ep_world_size
                rank_num_per_shared_expert = shared_expert_rank_num // shared_expert_num
                max_shared_group_num = (ep_world_size + rank_num_per_shared_expert - 1) // rank_num_per_shared_expert
                a = max(max_bs * max_shared_group_num, global_bs_real * min(moe_expert_num // (ep_world_size - shared_expert_rank_num), k))
                local_moe_expert_num = max(local_moe_expert_num, moe_expert_num // (ep_world_size - shared_expert_rank_num))

    ep_recv_cnt_num = 0
    if tp_world_size == 2:
        ep_recv_cnt_num = ep_world_size * local_moe_expert_num * tp_world_size
    else:
        ep_recv_cnt_num = ep_world_size * local_moe_expert_num

    if quant_mode == 0:
        outDtype = x.dtype
    elif y_dtype is not None:
        outDtype = TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP[y_dtype]

    expand_idx = x.new_empty((max(bs * k, a * 128)), dtype=torch.int32)
    expand_x = x.new_empty(tuple([max(a, a * tp_world_size), h]), dtype=outDtype)
    dynamic_scales_dtype = get_dispatch_dynamic_scales_dtype(x, scales, quant_mode)
    if tp_world_size == 0:
        dynamic_scales = x.new_empty((a), dtype=dynamic_scales_dtype)
    elif tp_world_size == 1:
        dynamic_scales_shape = get_dispatch_dynamic_shape(scales, quant_mode, a, h)
        dynamic_scales = x.new_empty(dynamic_scales_shape, dtype=dynamic_scales_dtype)
    else:
        dynamic_scales = x.new_empty((a * tp_world_size), dtype=dynamic_scales_dtype)
    expert_token_nums = x.new_empty((local_moe_expert_num), dtype=torch.int64)
    ep_recv_counts = x.new_empty((ep_recv_cnt_num), dtype=torch.int32)
    tp_recv_counts = x.new_empty((tp_world_size), dtype=torch.int32)
    expand_scales = x.new_empty((0), dtype=torch.float32)
    if expert_scales is not None:
        ep_recv_cnt_num = ep_world_size * local_moe_expert_num + global_bs_real * 2 * k * (ep_world_size // 8)
        ep_recv_counts = x.new_empty((ep_recv_cnt_num), dtype=torch.int32)
        expand_scales = x.new_empty((a), dtype=torch.float32)
    return (expand_x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts, expand_scales)


@impl(m, "npu_moe_distribute_combine")
def npu_moe_distribute_combine_meta(expand_x, expert_ids, expand_idx, ep_send_counts, expert_scales, group_ep, ep_world_size, ep_rank_id, moe_expert_num,
                                    tp_send_counts=None, x_active_mask=None, activation_scale=None, weight_scale=None, group_list=None, expand_scales=None, group_tp="", tp_world_size=0,
                                    tp_rank_id=0, expert_shard_type=0, shared_expert_num=1, shared_expert_rank_num=0, global_bs=0, out_dtype=0, comm_quant_mode=0, group_list_type=0):
    dim_list = []
    dim_list.append(expert_ids.size(0))
    dim_list.append(expand_x.size(1))

    return expand_x.new_empty(tuple(dim_list), dtype=expand_x.dtype)


@impl(m, "npu_moe_distribute_combine_v2")
def npu_moe_distribute_combine_v2_meta(expand_x, expert_ids, assist_info_for_combine, ep_send_counts, expert_scales, group_ep, ep_world_size, ep_rank_id, moe_expert_num,
                                       tp_send_counts=None, x_active_mask=None, expand_scales=None, shared_expert_x=None, elastic_info=None, ori_x=None, const_expert_alpha_1=None, const_expert_alpha_2=None, const_expert_v=None, performance_info=None, group_tp="", tp_world_size=0,
                                       tp_rank_id=0, expert_shard_type=0, shared_expert_num=1, shared_expert_rank_num=0, global_bs=0, comm_quant_mode=0, comm_alg="", zero_expert_num=0, copy_expert_num=0, const_expert_num=0):
    dim_tuple = (expert_ids.size(0), expand_x.size(1))

    return expand_x.new_empty(dim_tuple)


@impl(m, "npu_moe_distribute_combine_add_rms_norm")
def npu_moe_distribute_combine_add_rms_norm_meta(expand_x, expert_ids, expand_idx, ep_send_counts, expert_scales, residual_x, gamma, group_ep, ep_world_size, ep_rank_id, moe_expert_num,
                                    tp_send_counts=None, x_active_mask=None, activation_scale=None, weight_scale=None, group_list=None, expand_scales=None, shared_expert_x=None, elastic_info=None, ori_x=None, const_expert_alpha_1=None, const_expert_alpha_2=None, const_expert_v=None,
                                    group_tp="", tp_world_size=0, tp_rank_id=0, expert_shard_type=0, shared_expert_num=1, shared_expert_rank_num=0, global_bs=0, out_dtype=0, comm_quant_mode=0, group_list_type=0, comm_alg="", norm_eps=0, zero_expert_num=0, copy_expert_num=0, const_expert_num=0):
    dim_list = []
    dim_list.append(expert_ids.size(0))
    dim_list.append(1)
    dim_list.append(expand_x.size(1))
    dim_list2 = []
    dim_list2.append(expert_ids.size(0))
    dim_list2.append(1)
    dim_list2.append(1)

    return (expand_x.new_empty(tuple(dim_list), dtype=expand_x.dtype), expand_x.new_empty(tuple(dim_list2), dtype=torch.float32), expand_x.new_empty(tuple(dim_list), dtype=expand_x.dtype))


@impl(m, "_npu_distribute_barrier")
def _npu_distribute_barrier(x_ref, group, world_size, *, time_out=None, elastic_info=None):
    return torch.empty_like(x_ref)


@impl(m, "npu_moe_update_expert")
def npu_moe_update_expert_meta(expert_ids, eplb_table, expert_scales=None, pruning_threshold=None, active_mask=None, local_rank_id=-1, world_size=-1, balance_mode=0):
    dim_list = []
    dim_list.append(expert_ids.size(0))
    dim_list.append(expert_ids.size(1))

    return (expert_ids.new_empty(tuple(dim_list), dtype=expert_ids.dtype), expert_ids.new_empty(tuple(dim_list), dtype=torch.bool))


@impl(m, "npu_ffn")
def npu_ffn_meta(x, weight1, weight2, activation, *, expert_tokens=None, expert_tokens_index=None, bias1=None,
                 bias2=None, scale=None, offset=None, deq_scale1=None, deq_scale2=None, antiquant_scale1=None,
                 antiquant_scale2=None, antiquant_offset1=None, antiquant_offset2=None, inner_precise=0,
                 output_dtype=None):
    dim_list = []
    for i in range(0, x.dim() - 1):
        dim_list.append(x.size(i))
    dim_list.append(weight2.size(weight2.dim() - 1))
    if x.dtype == torch.int8:
        if output_dtype is not None and output_dtype == torch.bfloat16:
            return x.new_empty(tuple(dim_list), dtype=torch.bfloat16)
        else:
            return x.new_empty(tuple(dim_list), dtype=torch.float16)
    else:
        return x.new_empty(tuple(dim_list))


def gmm_get_dtype(output_dtype):
    if not output_dtype:
        return output_dtype
    elif output_dtype not in [TORCH_DTYPE_MAP[torch.float16], TORCH_DTYPE_MAP[torch.bfloat16], TORCH_DTYPE_MAP[torch.float32], TORCH_DTYPE_MAP[torch.int32]]:
        raise RuntimeError("The output dtype ", str(output_dtype), " is not supported for now.")
    else:
        return TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP.get(output_dtype)


def is_transpose_weight(weight):
    return weight.stride()[-2] == 1 and weight.stride()[-1] == weight.shape[-2]


@impl(m, "npu_grouped_matmul")
@impl(m, "npu_grouped_matmul.List")
def npu_grouped_matmul_meta(x, weight, *, bias=None, scale=None, offset=None, antiquant_scale=None,
                            antiquant_offset=None, per_token_scale=None, group_list=None,
                            activation_input=None, activation_quant_scale=None, activation_quant_offset=None,
                            split_item=0, group_type=None, group_list_type=0, act_type=0, tuning_config=None,
                            output_dtype=None, x_dtype=None, weight_dtype=None, scale_dtype=None, per_token_scale_dtype=None):
    torch._check(
        group_type == -1 or group_type == 0 or group_type == 2 or (isinstance(group_list, list) and group_type is None),
        lambda: f"group_type only supports -1, 0 and 2, but got {group_type} {ops_error(ErrCode.VALUE)}",
    )
    if x_dtype is not None:
        torch._check(
            x_dtype == torch_npu.hifloat8 or x_dtype == torch_npu.float4_e1m2fn_x2 or x_dtype == torch_npu.float4_e2m1fn_x2,
            lambda: "x_dtype supports hifloat8, mxfp4 for now, but it is " + npu_dtype_to_str(x_dtype),
        )
    if weight_dtype is not None:
        torch._check(
            weight_dtype == torch_npu.hifloat8 or weight_dtype == torch_npu.float4_e1m2fn_x2 or weight_dtype == torch_npu.float4_e2m1fn_x2,
            lambda: "weight_dtype only supports hifloat8, mxfp4 for now, but it is " + npu_dtype_to_str(weight_dtype),
        )
    if scale_dtype is not None:
        torch._check(
            scale_dtype == torch_npu.float8_e8m0fnu,
            lambda: "scale_dtype only supports float8_e8m0fnu for now, but it is " + npu_dtype_to_str(scale_dtype),
        )
    if per_token_scale_dtype is not None:
        torch._check(
            per_token_scale_dtype == torch_npu.float8_e8m0fnu,
            lambda: "per_token_scale_dtype only supports float8_e8m0fnu for now, but it is " + npu_dtype_to_str(per_token_scale_dtype),
        )
    y = []
    num_x = len(x)
    singleWeight = len(weight) == 1 and len(weight[0].shape) == 3
    n = weight[0].shape[2] if singleWeight else weight[0].shape[1]
    output_dtype = gmm_get_dtype(output_dtype)
    INT4_IN_INT32 = 8
    FP4_IN_INT8 = 2
    is_a4w4_mxfp = (x_dtype == torch_npu.float4_e1m2fn_x2 or x_dtype == torch_npu.float4_e2m1fn_x2) and \
                   (weight_dtype == torch_npu.float4_e1m2fn_x2 or weight_dtype == torch_npu.float4_e2m1fn_x2)
    if num_x > 0 and output_dtype is None:
        output_dtype = x[0].dtype
    if split_item == 0:
        for i in range(num_x):
            ni = n if singleWeight else weight[i].shape[1]
            dim_n = ni * INT4_IN_INT32 if weight[i].dtype == torch.int32 else ni
            y.append(x[i].new_empty((*x[i].shape[:-1], dim_n), dtype=output_dtype))
    elif split_item == 1:
        num_group_list = group_list.shape[0] if isinstance(group_list, torch.Tensor) else len(group_list)
        pre_offset = group_list[0]
        dim_n = n * INT4_IN_INT32 if weight[0].dtype == torch.int32 else n
        y.append(x[0].new_empty((pre_offset, dim_n), dtype=output_dtype))
        for i in range(1, num_group_list):
            ni = n if singleWeight else weight[i].shape[1]
            cur_offset = group_list[i]
            dim_n = ni * INT4_IN_INT32 if weight[i].dtype == torch.int32 else ni
            y.append(x[0].new_empty((cur_offset - pre_offset, dim_n), dtype=output_dtype))
            pre_offset = cur_offset
    elif split_item == 2:
        dim_m = 0
        dim_n = n * INT4_IN_INT32 if (weight[0].dtype == torch.int32 or weight[0].dtype == torch.float32) and \
                not is_transpose_weight(weight[0]) else n
        for i in range(num_x):
            dim_m += x[i].shape[0]
        if is_a4w4_mxfp:
            dim_n = n if x[0].size(x[0].dim() - 1) == weight[0].size(weight[0].dim() - 2) else n * FP4_IN_INT8
        if group_type != 2:
            y.append(x[0].new_empty((dim_m, dim_n), dtype=output_dtype))
        else:
            num_group_list = group_list.shape[0]
            y.append(x[0].new_empty((num_group_list, dim_m, dim_n), dtype=output_dtype))
    elif split_item == 3:
        dim_n = n * INT4_IN_INT32 if (weight[0].dtype == torch.int32 or weight[0].dtype == torch.float32) and \
                not is_transpose_weight(weight[0]) else n
        if is_a4w4_mxfp:
            dim_n = n if x[0].size(x[0].dim() - 1) == weight[0].size(weight[0].dim() - 2) else n * FP4_IN_INT8
        if group_type != 2:
            y.append(x[0].new_empty((x[0].shape[0], dim_n), dtype=output_dtype))
        else:
            num_group_list = group_list.shape[0]
            y.append(x[0].new_empty((num_group_list, x[0].shape[0], dim_n), dtype=output_dtype))

    return y


@impl(m, "npu_grouped_matmul_add_")
def npu_grouped_matmul_add__meta(y, x1, x2, group_list, *, transpose_x=True,
                                 transpose_weight=False, group_type=2, group_list_type=0):
    torch._check(
        group_type == 2,
        lambda: f"group_type only supports 2, but got {group_type} {ops_error(ErrCode.VALUE)}",
    )
    return y


@impl(m, "npu_matmul_all_to_all")
def npu_matmul_all_to_all_meta(x1, x2, hcom, world_size, bias=None, all2all_axes=None):
    # world_size为设备卡数，这里假设卡数为2
    world_size = 2
    # 推导output的shape
    # 因为该算子目前不支持转置，所以output的m维度==x1.size[0]，output的n维度==x2.size[1]
    out_m = x1.size(0) * world_size
    out_n = x2.size(1) // world_size
    size = [out_m, out_n]
    # matmul_all_to_all算子的输出dtype，必须等于输入x1和x2的dtype
    dtype = x1.dtype
    return torch.empty(size, dtype=dtype, device='meta')


@impl(m, "npu_quant_matmul_all_to_all")
def npu_quant_matmul_all_to_all_meta(x1, x2, hcom, world_size, bias=None, x1_scale=None, x2_scale=None, common_scale=None,
                                     x1_offset=None, x2_offset=None, x1_quant_mode=3, x2_quant_mode=2, common_quant_mode=0,
                                     group_sizes=None, all2all_axes=None, comm_quant_dtype=28, x1_dtype=None, x2_dtype=None,
                                     x1_scale_dtype=None, x2_scale_dtype=None,
                                     output_scale_dtype=None, comm_scale_dtype=None, y_dtype=None):
    # world_size为设备卡数，这里假设卡数为2
    world_size = 2
    # 推导output的shape
    # 因为该算子目前不支持转置，所以output的m维度==x1.size[0]，output的n维度==x2.size[1]
    out_m = x1.size(0) * world_size
    out_n = x2.size(1) // world_size
    size = [out_m, out_n]
    # 推导output的dtype，默认为float32
    if y_dtype is None:
        dtype = torch.float32
    else:
        dtype = TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP[y_dtype]
    return torch.empty(size, dtype=dtype, device='meta')


@impl(m, "npu_all_to_all_matmul")
def npu_all_to_all_matmul_meta(x1, x2, hcom, world_size, bias=None, all2all_axes=None, all2all_out_flag=True):
    # world_size为设备卡数，这里假设卡数为2
    world_size = 2
    # 推导output的shape
    # 因为该算子目前不支持转置，所以output的m维度==x1.size[0]，output的n维度==x2.size[1]
    out_m = x1.size(0) // world_size
    out_n = x2.size(1)
    size = [out_m, out_n]
    # all_to_all_matmul算子的输出dtype，必须等于输入x1和x2的dtype
    dtype = x1.dtype
    if all2all_out_flag:
        all2all_out_size = [out_m, x1.size(1) * world_size]
        return (torch.empty(size, dtype=dtype, device='meta'),
                torch.empty(all2all_out_size, dtype=dtype, device='meta'))
    else:
        return (torch.empty(size, dtype=dtype, device='meta'), None)


def add_quant_gmm_check(*args):
    group_sizes, x1_dtype, x2_dtype, x1_scale_dtype, x2_scale_dtype = args

    torch._check(
        group_sizes is None,
        lambda: "group_sizes is not supported for now",
    )
    if x1_dtype is not None:
        torch._check(
            x1_dtype == torch_npu.hifloat8,
            lambda: "x1_dtype is only supported hifloat8 for now, but it is " + str(x1_dtype),
        )
    if x2_dtype is not None:
        torch._check(
            x2_dtype == torch_npu.hifloat8,
            lambda: "x2_dtype is only supported hifloat8 for now, but it is " + str(x2_dtype),
        )

    if x1_scale_dtype is not None:
        torch._check(
            x1_scale_dtype == torch_npu.float8_e8m0fnu,
            lambda: "x1_scale_dtype is only supported float8_e8m0fnu for now, but it is " + str(x1_scale_dtype),
        )
    if x2_scale_dtype is not None:
        torch._check(
            x2_scale_dtype == torch_npu.float8_e8m0fnu,
            lambda: "x2_scale_dtype is only supported float8_e8m0fnu for now, but it is " + str(x2_scale_dtype),
        )


@impl(m, "npu_add_quant_gmm_")
def npu_add_quant_gmm__meta(y, x1, x2, x2_scale, group_list, *, x1_scale=None, group_list_type=0, group_sizes=None,
                            x1_dtype=None, x2_dtype=None, x1_scale_dtype=None, x2_scale_dtype=None):
    add_quant_gmm_check(group_sizes, x1_dtype, x2_dtype, x1_scale_dtype, x2_scale_dtype)
    return y


@impl(m, "npu_add_quant_gmm")
def npu_add_quant_gmm_meta(y, x1, x2, x2_scale, group_list, *, x1_scale=None, group_list_type=0, group_sizes=None,
                            x1_dtype=None, x2_dtype=None, x1_scale_dtype=None, x2_scale_dtype=None):
    add_quant_gmm_check(group_sizes, x1_dtype, x2_dtype, x1_scale_dtype, x2_scale_dtype)
    return torch.empty_like(y)


@impl(m, "npu_grouped_matmul_finalize_routing")
def npu_grouped_matmul_finalize_routing_meta(x, w, group_list, *, scale=None, bias=None, offset=None,
                                            pertoken_scale=None, shared_input=None, logit=None,
                                            row_index=None, dtype=None, shared_input_weight=1.0,
                                            shared_input_offset=0, output_bs=0, group_list_type=1, tuning_config=None,
                                            x_dtype=None, w_dtype=None, scale_dtype=None, pertoken_scale_dtype=None):
    torch._check(
        torch.is_tensor(x),
        lambda: "x must be tensor." + ops_error(ErrCode.VALUE)
    )
    torch._check(
        torch.is_tensor(w),
        lambda: "w must be tensor." + ops_error(ErrCode.VALUE)
    )

    if x_dtype is not None:
        torch._check(
            x_dtype == torch_npu.float4_e1m2fn_x2 or x_dtype == torch_npu.float4_e2m1fn_x2,
            lambda: "x_dtype supports float4_e1m2fn_x2, float4_e2m1fn_x2 for now, but it is " + npu_dtype_to_str(x_dtype),
        )
    if w_dtype is not None:
        torch._check(
            w_dtype == torch_npu.float4_e1m2fn_x2 or w_dtype == torch_npu.float4_e2m1fn_x2,
            lambda: "weight_dtype only supports float4_e1m2fn_x2, float4_e2m1fn_x2  for now, but it is " + npu_dtype_to_str(w_dtype),
        )
    if scale_dtype is not None:
        torch._check(
            scale_dtype == torch_npu.float8_e8m0fnu,
            lambda: "scale_dtype only supports float8_e8m0fnu for now, but it is " + npu_dtype_to_str(scale_dtype),
        )
    if pertoken_scale_dtype is not None:
        torch._check(
            pertoken_scale_dtype == torch_npu.float8_e8m0fnu,
            lambda: "pertoken_scale_dtype only supports float8_e8m0fnu for now, but it is " + npu_dtype_to_str(per_token_scale_dtype),
        )

    dimm = x.size(0)
    x_dim = x.dim()
    w_dim = w.dim()
    dimn = w.size(w_dim - 1)
    INT4_IN_INT32 = 8

    torch._check(
        x_dim == 2 and w_dim == 3,
        lambda: "x_dim should be 2 and w_dim should be 3." + ops_error(ErrCode.VALUE),
    )
    torch._check(
        dimn > 0,
        lambda: "n value must bigger than 0." + ops_error(ErrCode.VALUE),
    )

    if dtype is None:
        dtype = torch.float32
    if shared_input is not None and logit is not None:
        torch._check(
            dtype == torch.float32,
            lambda: "When shared_input is not None, output_dtype must be float32, but it is " +
                    str(dtype) + ops_error(ErrCode.TYPE),
        )

    y_dimm = output_bs
    if output_bs == 0:
        y_dimm = dimm

    FP4_IN_INT8 = 2
    w_trans = x.size(-1) == w.size(-2)
    is_a4w4_input = False
    if x_dtype is not None and w_dtype is not None:
        is_a4w4_input = (x_dtype == torch_npu.float4_e1m2fn_x2 or x_dtype == torch_npu.float4_e2m1fn_x2) and \
                        (w_dtype == torch_npu.float4_e1m2fn_x2 or w_dtype == torch_npu.float4_e2m1fn_x2)
    if w.dtype == torch.int32:
        dim_n = dimn * INT4_IN_INT32
    elif is_a4w4_input and not w_trans:
        dim_n = dimn * FP4_IN_INT8
    else:
        dim_n = dimn

    dim_list = [y_dimm, dim_n]

    if dtype == torch.float32:
        return x.new_empty(tuple(dim_list), dtype=torch.float32)
    else:
        raise RuntimeError("Not supportted output dtype is " + str(dtype))


@impl(m, "npu_group_norm_silu")
def group_norm_silu_meta(self, gemma, beta, group, eps=0.00001):
    N = self.size(1)
    if gemma is None or beta is None:
        return (torch.empty_like(self, dtype=self.dtype), self.new_empty((N, group), dtype=self.dtype), self.new_empty((N, group), dtype=self.dtype))
    else:
        return (torch.empty_like(self, dtype=self.dtype), gemma.new_empty((N, group), dtype=gemma.dtype), beta.new_empty((N, group), dtype=beta.dtype))


@impl(m, "npu_mm_all_reduce_base")
def npu_mm_all_reduce_base_forward(x1, x2, hcom, reduce_op='sum', bias=None, antiquant_scale=None,
                                   antiquant_offset=None, x3=None, dequant_scale=None, pertoken_scale=None,
                                   comm_quant_scale_1=None, comm_quant_scale_2=None, antiquant_group_size=0,
                                   comm_turn=0, group_sizes=None, y_dtype=None, x1_dtype=None, x2_dtype=None,
                                   dequant_scale_dtype=None, pertoken_scale_dtype=None, comm_quant_mode=0):
    dim_list = []
    for i in range(x1.dim()):
        dim_list.append(x1.size(i))
    dim_list[-1] = x2.size(1)
    dim_tuple = tuple(dim_list)
    if dequant_scale is not None:
        if y_dtype is None:
            dtype = torch.bfloat16 if dequant_scale.dtype == torch.bfloat16 else torch.float16
            return x1.new_empty(dim_tuple, dtype=dtype)
        else:
            return x1.new_empty(dim_tuple, dtype=TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP.get(y_dtype, torch.float16))
    return x1.new_empty(dim_tuple)



@impl(m, "npu_weight_quant_batchmatmul")
def npu_weight_quant_batchmatmul_meta(x, weight, antiquant_scale, antiquant_offset=None, quant_scale=None, quant_offset=None, bias=None, antiquant_group_size=0, inner_precise=0,
                                      weight_dtype=None):
    dim_m = x.size(0)
    if (weight.dtype == torch.int32 or weight.dtype == torch.float32) and weight.is_contiguous():
        dim_n = weight.size(1) * 8
    else:
        dim_n = weight.size(1)
    if quant_scale is not None:
        return x.new_empty((dim_m, dim_n), dtype=torch.int8)
    return x.new_empty((dim_m, dim_n), dtype=x.dtype)


def bias_shape_check(*args):
    x2, bias, batch_val, is_a4w4, is_a8w4_float, transpose_x2 = args
    bias_dim_num = bias.dim()
    if is_a4w4:
        torch._check(
            bias_dim_num == 1,
            lambda: "bias_dim_num should be 1 when x1's dtype is int32, please check bias dim num " + ops_error(ErrCode.VALUE),
        )
    elif is_a8w4_float:
        torch._check(bias_dim_num == 2,
            lambda: "in a8w4 float, bias_dim_num should be 2 , please check bias dim num " + ops_error(ErrCode.VALUE),
        )
        return
    else:
        torch._check(
            bias_dim_num == 1 or bias_dim_num == 3,
            lambda: "bias_dim_num should be 1 or 3 when x1's dtype is int8, please check bias dim num " + ops_error(ErrCode.VALUE),
        )
    x2_dim_num = x2.dim()
    x2_n_dim = x2.size(x2_dim_num - 1) * 8 if (is_a4w4 and not transpose_x2) else x2.size(x2_dim_num - 1)
    bias_first_dim = bias.size(0)
    if bias_dim_num == 1:
        torch._check(
            bias_first_dim == x2_n_dim,
            lambda: "bias_first_dim should be equal to x2 n dim, please check bias 1st dim value " + ops_error(ErrCode.VALUE),
        )
        return
    bias_second_dim = bias.size(1)
    bias_third_dim = bias.size(2)
    torch._check(
        bias_first_dim == batch_val,
        lambda: "infered batch value should be equal to bias batch dim value, please check bias batch dim value" + ops_error(ErrCode.VALUE),
    )
    torch._check(
        bias_second_dim == 1,
        lambda: "bias_second_dim should be 1, please check bias second dim value " + ops_error(ErrCode.VALUE),
    )
    torch._check(
        bias_third_dim == x2_n_dim,
        lambda: "bias_third_dim should be equal to x2_n_dim, please check bias third dim value " + ops_error(ErrCode.VALUE),
    )


def quant_matmul_shape_check(*args):
    x1, x2, scale, offset, pertoken_scale, is_a4w4, transpose_x2, is_a8w4_int, is_a8w4_float, group_sizes = args
    X_MAX_DIM = 6
    X_MIN_DIM = 2
    INT4_IN_INT32 = 8
    FP4_IN_INT8 = 2
    GROUP_SIZE_A8W4 = 256
    x1_dim_num = x1.dim()
    x2_dim_num = x2.dim()
    x1_m_dim = x1.size(x1_dim_num - 2)
    x1_k_dim = x1.size(x1_dim_num - 1)
    x2_k_dim = x2.size(x2_dim_num - 2)
    x2_n_dim = x2.size(x2_dim_num - 1) * INT4_IN_INT32 if ((is_a4w4 and not transpose_x2) or is_a8w4_int) else x2.size(x2_dim_num - 1)
    torch._check(
        x1_dim_num >= X_MIN_DIM and x1_dim_num <= X_MAX_DIM,
        lambda: f"x1 dim num should be 2 ~ 6, please check x1 dim num {ops_error(ErrCode.VALUE)}",
    )
    if is_a4w4 and not transpose_x2:
        torch._check(
            x1_k_dim * INT4_IN_INT32 == x2_k_dim,
            lambda: f"k dim of x2 should be 8 multiple of k dim of x1, \
                please check k dim of x1 and x2 {ops_error(ErrCode.VALUE)}",
        )
    elif is_a8w4_float:
        if (x2.dtype == torch.float32):
            if pertoken_scale is not None:
                torch._check(
                    x1_k_dim == x2_k_dim * INT4_IN_INT32,
                    lambda: "a8w4 nz mx quant only support x1 not transpose and x2 transpose and k dim of x1 should be 8 multiple of k dim of x2." + ops_error(ErrCode.VALUE),
                )
            else:
                torch._check(
                    x1_k_dim == x2_k_dim,
                    lambda: "a8w4 nz t-cg quant only support x1 not transpose and x2 not transpose and k dim of x1 and x2 need be same." + ops_error(ErrCode.VALUE),
                )
        else:
            torch._check(
                x1_k_dim == x2_k_dim * FP4_IN_INT8,
                lambda: "a8w4_float nd only support x1 not transpose and x2 transpose and k dim of x1 should be 2 multiple of k dim of x2, please check k dim of x1 and x2" + ops_error(ErrCode.VALUE),
            )
    else:
        torch._check(
            x1_k_dim == x2_k_dim,
            lambda: f"k dim of x1 and x2 need be same, please check k dim of x1 and x2 {ops_error(ErrCode.VALUE)}",
        )

    if is_a4w4:
        torch._check(
            x2_dim_num == X_MIN_DIM,
            lambda: f"x2 dim num should be 2 when x1's dtype is int32, \
                please check x2 dim num {ops_error(ErrCode.VALUE)}",
        )
    else:
        torch._check(
            x2_dim_num >= X_MIN_DIM and x2_dim_num <= X_MAX_DIM,
            lambda: f"x2 dim num should be 2 ~ 6 when x1's dtype is int8, \
                please check x2 dim num {ops_error(ErrCode.VALUE)}",
        )

    if offset is not None:
        offset_dim_num = offset.dim()
        torch._check(
            offset_dim_num == 1,
            lambda: f"the offset dim num must be 1, please check offset dim num {ops_error(ErrCode.VALUE)}",
        )
        offset_first_dim = offset.size(0)
        torch._check(
            offset_first_dim == 1 or offset_first_dim == x2_n_dim,
            lambda: f"the offset 1st dim value must be 1 or x2 n dim value, \
                please check offset 1st dim value {ops_error(ErrCode.VALUE)}",
        )
    if group_sizes is None:
        if pertoken_scale is not None:
            pertoken_scale_dim_num = pertoken_scale.dim()
            if is_a8w4_int:
                torch._check(
                    pertoken_scale_dim_num == 2,
                    lambda: f"the pertoken_scale dim num must be 2, please check scale dim num {ops_error(ErrCode.VALUE)}",
                )
            else:
                torch._check(
                    pertoken_scale_dim_num == 1,
                    lambda: f"the pertoken_scale dim num must be 1, please check scale dim num {ops_error(ErrCode.VALUE)}",
                )

        scale_dim_num = scale.dim()
        if is_a8w4_int:
            torch._check(
                scale_dim_num == 2,
                lambda: f"the scale dim num must be 2, please check scale dim num {ops_error(ErrCode.VALUE)}",
            )
            scale_first_dim = scale.size(0)
            torch._check(
                scale_first_dim == x1_k_dim // GROUP_SIZE_A8W4,
                lambda: f"the scale 1st dim value must equal to x1 k dim divide 256, \
                    please check scale 1st dim value {ops_error(ErrCode.VALUE)}",
            )
            scale_last_dim = scale.size(1)
            torch._check(
                scale_last_dim == x2_n_dim,
                lambda: f"the scale last dim value must equal to x2 n dim value, \
                    please check scale last dim value {ops_error(ErrCode.VALUE)}",
            )
        else:
            torch._check(
                scale_dim_num == 1,
                lambda: f"the scale dim num must be 1, please check scale dim num {ops_error(ErrCode.VALUE)}",
            )
            scale_first_dim = scale.size(0)
            torch._check(
                scale_first_dim == 1 or scale_first_dim == x2_n_dim,
                lambda: f"the scale 1st dim value must be 1 or x2 n dim value, \
                    please check scale 1st dim value {ops_error(ErrCode.VALUE)}",
            )


def quant_matmul_bias_dtype_check(bias, pertoken_scale, output_dtype):
    bias_dtype_supported_list = [torch.int32, torch.bfloat16, torch.float32, torch.float16]
    torch._check(
        bias.dtype in bias_dtype_supported_list,
        lambda: "bias's type supported for int32, bfloat16, float16 and float32, but bias.dtype is " + str(bias.dtype) + ops_error(ErrCode.TYPE),
    )
    if bias.dtype == torch.bfloat16:
        torch._check(
            output_dtype == TORCH_DTYPE_MAP[torch.bfloat16],
            lambda: "When bias dtype is bfloat16, output_dtype must be bfloat16, but it is " +
                    str(output_dtype) + ops_error(ErrCode.TYPE),
        )
    if output_dtype == TORCH_DTYPE_MAP[torch.int32]:
        torch._check(
            bias.dtype == torch.int32,
            lambda: "When output_dtype dtype is int32, bias_dtype must be int32, but it is " +
                    str(bias.dtype) + ops_error(ErrCode.TYPE),
        )
    if pertoken_scale is not None:
        if bias.dtype == torch.float16:
            torch._check(
                output_dtype == TORCH_DTYPE_MAP[torch.float16],
                lambda: "When bias dtype is float16 and pertoken is given, output_dtype must be float16, but it is " +
                        str(output_dtype) + ops_error(ErrCode.TYPE),
            )
    else:
        torch._check(
            bias.dtype != torch.float16,
            lambda: "Bias dtype cannot be float16 when pertoken not given." + ops_error(ErrCode.TYPE),
        )


def quant_matmul_extra_dtype_check(*args):
    x1, x2, scale, pertoken_scale, x1_dtype, x2_dtype, scale_dtype, is_a8w4_float, pertoken_scale_dtype = args
    if x1_dtype is not None:
        torch._check(
            x1_dtype == torch_npu.float4_e2m1fn_x2 or x1_dtype == torch_npu.float4_e1m2fn_x2 or x1_dtype == torch_npu.hifloat8,
            lambda: "The x1_dtype supported for torch_npu.float4_e2m1fn_x2, torch_npu.float4_e1m2fn_x2, torch_npu.hifloat8, but x1_dtype is " +
                    npu_dtype_to_str(x2_dtype) + ops_error(ErrCode.TYPE),
        )
        torch._check(
            x1.element_size() == 1,
            lambda: "When x1_dtype is not None, x1 must be a 1 byte tensor, but the byte size of x1 is" +
                    str(x1.element_size()) + ops_error(ErrCode.TYPE),
        )
    if x2_dtype is not None and not is_a8w4_float:
        torch._check(
            x2_dtype == torch_npu.float4_e2m1fn_x2 or x2_dtype == torch_npu.float4_e1m2fn_x2 or x2_dtype == torch_npu.hifloat8,
            lambda: "The x1_dtype supported for torch_npu.float4_e2m1fn_x2, torch_npu.float4_e1m2fn_x2, torch_npu.hifloat8, but x1_dtype is " +
                    npu_dtype_to_str(x2_dtype) + ops_error(ErrCode.TYPE),
        )
        torch._check(
            x2.element_size() == 1,
            lambda: "When x2_dtype is not None, x2 must be a 1 byte tensor, but the byte size of x2 is" +
                    str(x2.element_size()) + ops_error(ErrCode.TYPE),
        )
    if scale_dtype is not None:
        torch._check(
            scale_dtype == torch_npu.float8_e8m0fnu,
            lambda: "The scale_dtype supported for torch_npu.float8_e8m0fnu, but scale_dtype is " +
                    npu_dtype_to_str(scale_dtype) + ops_error(ErrCode.TYPE),
        )
        torch._check(
            scale.element_size() == 1,
            lambda: "When scale_dtype is not None, scale must be a 1 byte tensor, but the byte size of scale is" +
                    str(scale.element_size()) + ops_error(ErrCode.TYPE),
        )
    if pertoken_scale_dtype is not None:
        torch._check(
            pertoken_scale_dtype == torch_npu.float8_e8m0fnu,
            lambda: "The pertoken_scale_dtype supported for torch_npu.float8_e8m0fnu, but pertoken_scale_dtype is " +
                    npu_dtype_to_str(pertoken_scale_dtype) + ops_error(ErrCode.TYPE),
        )
        torch._check(
            pertoken_scale.element_size() == 1,
            lambda: "When pertoken_scale_dtype is not None, pertoken_scale must be a 1 byte tensor, but the byte size of pertoken_scale is" +
                    str(pertoken_scale.element_size()) + ops_error(ErrCode.TYPE),
        )


def quant_matmul_dtype_check(*args):
    x1, x2, scale, offset, pertoken_scale, bias, output_dtype, is_a4w4, is_a8w4_int, is_a8w4_float, y_scale = args
    if is_a8w4_int:
        torch._check(
            x1.dtype == torch.int8,
            lambda: f"x1's type should be torch.int8 in A8W4, but x1.dtype is {str(x1.dtype)} {ops_error(ErrCode.TYPE)}",
        )
        torch._check(
            x2.dtype == torch.int32,
            lambda: f"x2's type should be torch.int32 in A8W4, but x2.dtype is {str(x2.dtype)} {ops_error(ErrCode.TYPE)}",
        )
        torch._check(
            scale.dtype == torch.int64,
            lambda: f"scale's type should be torch.int64 in A8W4, \
                but scale.dtype is {str(scale.dtype)} {ops_error(ErrCode.TYPE)}",
        )
        if offset is not None:
            torch._check(
                offset.dtype == torch.float32,
                lambda: f"offset's type should be torch.float32 in A8W4, \
                    but offset.dtype is {str(offset.dtype)} {ops_error(ErrCode.TYPE)}",
            )
        if pertoken_scale is not None:
            torch._check(
                pertoken_scale.dtype == torch.float32,
                lambda: f"pertoken_scale's type should be torch.float32 in A8W4, \
                    but pertoken_scale.dtype is {str(pertoken_scale.dtype)} {ops_error(ErrCode.TYPE)}",
            )
        if bias is not None:
            torch._check(
                bias.dtype == torch.int32,
                lambda: f"bias's type should be torch.int32 in A8W4, \
                    but bias.dtype is {str(bias.dtype)} {ops_error(ErrCode.TYPE)}",
            )
        if output_dtype is not None:
            torch._check(
                output_dtype == TORCH_DTYPE_MAP[torch.float16] or output_dtype == TORCH_DTYPE_MAP[torch.bfloat16],
                lambda: f"output_dtype's type should be torch.int32 or torch.bfloat16 in A8W4, \
                    but output_dtype.dtype is {npu_dtype_to_str(output_dtype)} {ops_error(ErrCode.TYPE)}",
            )
    else:
        if offset is not None:
            torch._check(
                offset.dtype == torch.float32,
                lambda: f"offset's type supported for float32, \
                    but offset.dtype is {str(offset.dtype)} {ops_error(ErrCode.TYPE)}",
            )

        if bias is not None:
            quant_matmul_bias_dtype_check(bias, pertoken_scale, output_dtype)
        if is_a8w4_float and y_scale is not None:
            torch._check(
                y_scale.dtype == torch.int64,
                lambda: "y_scale's type supported for int64, but y_scale.dtype is " + str(y_scale.dtype) + ops_error(ErrCode.TYPE),
            )


def quant_matmul_scale_offset_out_check(scale, offset, pertoken_scale, output_dtype, is_a4w4):
    if scale.dtype == torch.bfloat16:
        torch._check(
            output_dtype in [TORCH_DTYPE_MAP[torch.bfloat16], TORCH_DTYPE_MAP[torch.int32]],
            lambda: "When scale's dtype is bfloat16, output_dtype must be bfloat16 or int32, but output_dtype is " +
                    npu_dtype_to_str(output_dtype) + ops_error(ErrCode.TYPE),
        )
    if output_dtype == TORCH_DTYPE_MAP[torch.int32]:
        torch._check(
            scale.dtype in [torch.bfloat16, torch.float32],
            lambda: "When output_dtype is int32, scale's dtype must be bfloat16 or float32, but scale's dtype is " +
                    str(scale.dtype) + ops_error(ErrCode.TYPE),
        )
    if is_a4w4:
        torch._check(
            output_dtype == TORCH_DTYPE_MAP[torch.float16],
            lambda: "When input's dtype is int32, output_dtype must be float16, but output_dtype is " +
                    npu_dtype_to_str(output_dtype) + ops_error(ErrCode.TYPE),
        )


def quant_matmul_group_sizes_check(*args):
    x1, x2, scale, pertoken_scale, group_sizes, x1_dtype, x2_dtype, scale_dtype, pertoken_scale_dtype, is_a8w4_float = args
    if not is_a8w4_float and pertoken_scale is not None and pertoken_scale.dim() >= 2 and scale.dim() >= 2:
        if pertoken_scale_dtype is not None and pertoken_scale_dtype == torch_npu.float8_e8m0fnu:
            pertoken_scale_k_idx = pertoken_scale.dim() - 2
            scale_k_idx = scale.dim() - 3
        else:
            pertoken_scale_k_idx = pertoken_scale.dim() - 1
            scale_k_idx = scale.dim() - 2
        torch._check(
            (pertoken_scale.size(pertoken_scale_k_idx) == scale.size(scale_k_idx)),
            lambda: "In mx, B-B, G-B quantification, k dimension of scale and pertoken_scale must be equal, \
please check the sizes of scale and pertoken_scale" + ops_error(ErrCode.VALUE),
        )
    if group_sizes is None:
        return
    torch._check(
        len(group_sizes) == 3,
        lambda: "group_sizes's length must be 3, please check group_sizes's length" + ops_error(ErrCode.VALUE),
    )
    if is_a8w4_float:
        torch._check(
            (group_sizes[0] == 0 and group_sizes[1] == 0 and group_sizes[2] == 32) or \
            (group_sizes[0] == 1 and group_sizes[1] == 1 and group_sizes[2] == 32),
            lambda: "when the dtype of input is A8W4, group_sizes's value must be [0,0,32] or [1,1,32], please check group_sizes's value" + ops_error(ErrCode.VALUE),
        )
        return
    is_a8w8_int = x1_dtype is None and x2_dtype is None and x1.dtype == torch.int8 and x2.dtype == torch.int8
    if is_a8w8_int:
        torch._check(
            (group_sizes[0] == 0 and group_sizes[1] == 0 or group_sizes[2] == 0),
            lambda: "when the dtype of input is int8, group_sizes's value must be 0, please check group_sizes's value" + ops_error(ErrCode.VALUE),
        )
    if pertoken_scale is None:
        torch._check(
            group_sizes[0] == 0 or group_sizes[1] == 0 or group_sizes[2] == 0,
            lambda: "when the pertoken_scale is None, group_sizes's value must be 0, please check group_sizes's value" + ops_error(ErrCode.VALUE),
        )
    group_input_dtype_lst = [torch.uint8, torch.bits8, torch.float8_e4m3fn, torch.float8_e5m2]
    group_scale_dtype_lst = [torch.float32]
    has_group = (group_sizes[0] > 1 or group_sizes[1] > 1 or group_sizes[2] > 1)
    if group_sizes is not None and has_group:
        torch._check(
            (scale_dtype is not None and pertoken_scale_dtype is not None) or (scale.dtype in group_scale_dtype_lst or pertoken_scale.dtype in group_scale_dtype_lst),
            lambda: "When group_sizes's value is not 0, scale_dtype and pertoken_scale_dtype are None, dtype of scale and pertoken_scale must be both float32, but " +
                    "scale's dtype is " + str(scale.dtype) + " pertoken_scale's dtype is " + str(pertoken_scale.dtype) + ops_error(ErrCode.TYPE),
        )
        torch._check(
            (x1_dtype is not None and x2_dtype is not None) or (x1.dtype in group_input_dtype_lst or x2.dtype in group_input_dtype_lst),
            lambda: "When group_sizes's value is not 0, x1_dtype and x2_dtype are None, dtype of input must be uint8, float8_e4m3fn, float8_e5m2 or int32, but x1's dtype is " +
                    str(x1.dtype) + " x2's dtype is " + str(x2.dtype) + ops_error(ErrCode.TYPE),
        )
        if group_sizes[0] > 1:
            torch._check(
                pertoken_scale.dim() >= 2 and pertoken_scale.size(pertoken_scale.dim() - 2) == math.ceil(x1.size(x1.dim() - 2) / group_sizes[0]),
                lambda: "When group_sizes[0] > 1, ceil(x1.size(-2) / group_sizes[0]) must be equal to " +
                        "pertoken_scale's size(-2), please check your input" + ops_error(ErrCode.VALUE),
            )
            torch._check(
                group_sizes[1] == group_sizes[0] and group_sizes[1] == 128,
                lambda: "When group_sizes[1] > 1, group_sizes[1] must be equal to group_sizes[2] and must be equal to 128" + ops_error(ErrCode.VALUE),
            )
        if group_sizes[2] > 1:
            group_k_support_lst = [32, 128]
            torch._check(
                group_sizes[2] in group_k_support_lst,
                lambda: "When group_sizes[2] > 1, group_sizes[2] must be equal to 32 or 128, but group_sizes[2] is " +
                        str(group_sizes[2]) + ops_error(ErrCode.VALUE),
            )
        if group_sizes[1] > 1:
            torch._check(
                scale.dim() >= 2 and scale.size(scale.dim() - 1) == math.ceil(x2.size(x2.dim() - 1) / group_sizes[1]),
                lambda: "When group_sizes[2] > 1, ceil(x2.size(-1) / group_sizes[2]) must be equal to scale's size(-1), " +
                        "please check your input" + ops_error(ErrCode.VALUE),
            )
            torch._check(
                group_sizes[1] == group_sizes[2] and group_sizes[1] == 128,
                lambda: "When group_sizes[1] > 1, group_sizes[1] must be equal to group_sizes[2] and must be equal to 128" + ops_error(ErrCode.VALUE),
            )


@impl(m, "obfuscation_calculate")
def obfuscation_calculate_meta(fd, x, param, cmd):
    return torch.empty_like(x)


@impl(m, "obfuscation_finalize")
def obfuscation_finalize_meta(fd_to_close):
    return torch.empty_like(fd_to_close)


@impl(m, "npu_quant_matmul")
def npu_quant_matmul_meta(x1, x2, scale, *, offset=None, pertoken_scale=None, bias=None, output_dtype=None,
                          x1_dtype=None, x2_dtype=None, pertoken_scale_dtype=None, scale_dtype=None,
                          group_sizes=None, y_scale=None):
    INT4_IN_INT32 = 8
    FP4_IN_FP32 = 8
    batch_val = 1
    x1_dim_num = x1.dim()
    x2_dim_num = x2.dim()
    out_dim_num = max(x1_dim_num, x2_dim_num)
    shape_long = x1 if x1_dim_num > x2_dim_num else x2
    shape_short = x2 if x1_dim_num > x2_dim_num else x1
    vaild_offset = out_dim_num - min(x1_dim_num, x2_dim_num)
    is_a4w4 = x1.dtype == torch.int32 and x2.dtype == torch.int32
    is_a8w4_int = x1.dtype == torch.int8 and x2.dtype == torch.int32
    is_a8w4_float = x1.dtype == torch.float8_e4m3fn and (x2_dtype == torch_npu.float4_e2m1fn_x2 or x2.dtype == torch.float32)
    dim_list = []
    if is_a8w4_int:
        dim_list = [x1.shape[0], x2.shape[1] * INT4_IN_INT32]
        transpose_x2 = False
    else:
        for i in range(0, out_dim_num - 2):
            short_dim = 1 if i < vaild_offset else shape_short.size(i - vaild_offset)
            long_dim = shape_long.size(i)
            torch._check(
                not (short_dim > 1 and long_dim > 1 and short_dim != long_dim),
                lambda: "the batch shape cannot be broadcast" + ops_error(ErrCode.VALUE),
            )
            cur_batch_val = max(short_dim, long_dim)
            batch_val = batch_val * cur_batch_val
            dim_list.append(cur_batch_val)
        dimm = x1.size(x1.dim() - 2)
        transpose_x2 = x1.size(x1.dim() - 1) == x2.size(x2.dim() - 2)

        dimn = x2.size(x2.dim() - 1)
        if (is_a4w4 and not transpose_x2):
            dimn = x2.size(x2.dim() - 1) * INT4_IN_INT32
        elif (is_a8w4_float and x2.dtype == torch.float32 and pertoken_scale is None):
            dimn = x2.size(x2.dim() - 1) * FP4_IN_FP32

        dim_list.append(dimm)
        dim_list.append(dimn)
        if bias is not None:
            if bias.dim() == 3:
                torch._check(
                    len(dim_list) == 3,
                    lambda: "when bias dim is 3, out dim need to be 3" + ops_error(ErrCode.TYPE),
                )
            bias_shape_check(x2, bias, batch_val, is_a4w4, is_a8w4_float, transpose_x2)
        quant_matmul_scale_offset_out_check(scale, offset, pertoken_scale, output_dtype, is_a4w4)
        quant_matmul_extra_dtype_check(x1, x2, scale, pertoken_scale,
                                   x1_dtype, x2_dtype, scale_dtype, is_a8w4_float, pertoken_scale_dtype)
        quant_matmul_group_sizes_check(x1, x2, scale, pertoken_scale, group_sizes,
                                    x1_dtype, x2_dtype, scale_dtype, pertoken_scale_dtype, is_a8w4_float)
    quant_matmul_dtype_check(x1, x2, scale, offset, pertoken_scale, bias, output_dtype, is_a4w4, is_a8w4_int, is_a8w4_float, y_scale)
    quant_matmul_shape_check(x1, x2, scale, offset, pertoken_scale, is_a4w4, transpose_x2, is_a8w4_int, is_a8w4_float, group_sizes)

    tensor_dtype = torch.int8
    if output_dtype is not None:
        tensor_dtype = TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP.get(output_dtype)
        if tensor_dtype is None or (tensor_dtype not in TORCH_DTYPE_MAP.keys() and tensor_dtype != torch.uint8):
            raise RuntimeError("Not supported output dtype is " + npu_dtype_to_str(output_dtype))
    return shape_long.new_empty(tuple(dim_list), dtype=tensor_dtype)


@impl(m, "npu_quant_matmul_dequant")
def npu_quant_matmul_dequant_meta(x, quantized_weight, weight_scale, *,
                                  bias=None, x_scale=None, x_offset=None, smooth_scale=None, quant_mode="pertoken"):
    torch._check(
        x.dim() == 2,
        lambda: "the x dim support only 2" + ops_error(ErrCode.VALUE),
    )
    torch._check(
        x.dtype == torch.float16,
        lambda: "the x dtype support only float16" + ops_error(ErrCode.VALUE),
    )
    torch._check(
        quantized_weight.dim() == 2,
        lambda: "the quantized_weight dim support only 2" + ops_error(ErrCode.VALUE),
    )
    torch._check(
        quantized_weight.dtype == torch.int8,
        lambda: "the quantized_weight dtype support only int8" + ops_error(ErrCode.VALUE),
    )
    torch._check(
        weight_scale.dim() == 1,
        lambda: "the weight_scale dim support only 1" + ops_error(ErrCode.VALUE),
    )
    torch._check(
        weight_scale.dtype == torch.float,
        lambda: "the weight_scale dtype support only float" + ops_error(ErrCode.VALUE),
    )
    torch._check(
        x.shape[1] == quantized_weight.shape[1],
        lambda: "x shape[1] not equal to quantized_weight shape[1]" + ops_error(ErrCode.VALUE),
    )
    torch._check(
        weight_scale.shape[0] == quantized_weight.shape[0],
        lambda: "weight_scale shape[0] not equal to quantized_weight shape[0]" + ops_error(ErrCode.VALUE),
    )
    return torch.empty((x.shape[0], weight_scale.shape[0]), dtype=x.dtype, device='meta')


@impl(m, "npu_quant_matmul_reduce_sum")
def npu_quant_matmul_reduce_sum_meta(x1, x2, *, x1_scale=None, x2_scale=None):
    torch._check(x1.dim() == 3, lambda: f"x1 dim must be 3, but got {x.dim()}.")
    torch._check(x2.dim() == 3, lambda: f"x2 dim must be 3, but got {w.dim()}.")
    torch._check(x1.size(2) == x2.size(1), lambda: f"K dim of x1 must be same as x2.")
    torch._check(x1_scale is not None, lambda: f"x1_scale should not be None.")
    torch._check(x1_scale.dim() == 2, lambda: f"x1_scale dim must be 2, but got {x1_scale.dim()}.")
    torch._check(x2_scale is not None, lambda: f"x2_scale should not be None.")
    torch._check(x2_scale.dim() == 1, lambda: f"x2_scale dim must be 1, but got {x2_scale.dim()}.")

    dst_shape = (x1.size(1), x2.size(2))
    return torch.empty(dst_shape, dtype=torch.bfloat16, device=x1.device)


@impl(m, "npu_quant_grouped_matmul_dequant")
def npu_quant_grouped_matmul_dequant_meta(x, quantized_weight, weight_scale, group_list, *,
                                          bias=None, x_scale=None, x_offset=None, smooth_scale=None, quant_mode="pertoken"):
    torch._check(
        x.dim() == 2,
        lambda: "the x dim support only 2" + ops_error(ErrCode.VALUE),
    )
    torch._check(
        x.dtype == torch.float16,
        lambda: "the x dtype support only float16" + ops_error(ErrCode.VALUE),
    )
    torch._check(
        quantized_weight.dim() == 3,
        lambda: "the quantized_weight dim support only 3" + ops_error(ErrCode.VALUE),
    )
    torch._check(
        quantized_weight.dtype == torch.int8,
        lambda: "the quantized_weight dtype support only int8" + ops_error(ErrCode.VALUE),
    )
    torch._check(
        weight_scale.dim() == 2,
        lambda: "the weight_scale dim support only 2" + ops_error(ErrCode.VALUE),
    )
    torch._check(
        weight_scale.dtype == torch.float,
        lambda: "the weight_scale dtype support only float" + ops_error(ErrCode.VALUE),
    )
    torch._check(
        x.shape[1] == quantized_weight.shape[2],
        lambda: "x shape[1] not equal to quantized_weight shape[1]" + ops_error(ErrCode.VALUE),
    )
    torch._check(
        weight_scale.shape[0] == quantized_weight.shape[0],
        lambda: "weight_scale shape[0] not equal to quantized_weight shape[0]" + ops_error(ErrCode.VALUE),
    )
    torch._check(
        weight_scale.shape[1] == quantized_weight.shape[1],
        lambda: "weight_scale shape[1] not equal to quantized_weight shape[1]" + ops_error(ErrCode.VALUE),
    )
    return torch.empty((x.shape[0], weight_scale.shape[1]), dtype=x.dtype, device='meta')


@impl(m, "npu_transpose_batchmatmul")
def npu_transpose_batchmatmul_meta(input_, weight, *, bias=None, scale=None,
                                   perm_x1=None, perm_x2=None, perm_y=None,
                                   batch_split_factor=1):
    perm_x1 = perm_x1 or [0, 1, 2]
    perm_x2 = perm_x2 or [0, 1, 2]
    perm_y = perm_y or [1, 0, 2]
    check_perm_x1 = ((perm_x1[0] == 0 and perm_x1[1] == 1 and perm_x1[2] == 2) or
                     (perm_x1[0] == 1 and perm_x1[1] == 0 and perm_x1[2] == 2))
    torch._check(
        check_perm_x1,
        lambda: "perm_x1 should be [0, 1, 2] or [1, 0, 2]" + ops_error(ErrCode.VALUE),
    )
    if get_cann_version() >= "8.5.0":
        check_perm_x2 = ((perm_x2[0] == 0 and perm_x2[1] == 1 and perm_x2[2] == 2) or
                        (perm_x2[0] == 0 and perm_x2[1] == 2 and perm_x2[2] == 1))
        torch._check(
            check_perm_x2,
            lambda: "perm_x2 should be [0, 1, 2] or [0, 2, 1]" + ops_error(ErrCode.VALUE),
        )
    else:
        check_perm_x2 = (perm_x2[0] == 0 and perm_x2[1] == 1 and perm_x2[2] == 2)
        torch._check(
            check_perm_x2,
            lambda: "perm_x2 should be [0, 1, 2]" + ops_error(ErrCode.VALUE),
        )
    check_perm_y = perm_y[0] == 1 and perm_y[1] == 0 and perm_y[2] == 2
    torch._check(
        check_perm_y,
        lambda: "perm_y should be [1, 0, 2]" + ops_error(ErrCode.VALUE),
    )
    input_dtype_supported_list = [torch.float16, torch.float32, torch.bfloat16]
    torch._check(
        input_.dtype in input_dtype_supported_list,
        lambda: "input's type supported for float16, float32 and bfloat16, but now is " + str(input_.dtype) + ops_error(ErrCode.TYPE),
    )
    torch._check(
        weight.dtype in input_dtype_supported_list,
        lambda: "weight's type supported for float16, float32 and bfloat16, but now is " + str(weight.dtype) + ops_error(ErrCode.TYPE),
    )
    torch._check(
        bias is None,
        lambda: "The bias is not supported in TransposeBatchMatMul" + ops_error(ErrCode.TYPE),
    )
    M = input_.size(perm_x1.index(1))
    batchM = input_.size(perm_x1.index(0))
    N = weight.size(perm_x2.index(2))
    dim_list = (M, batchM, N)

    dtype = input_.dtype
    if scale is not None:
        dtype = torch.int8
        dim_list = (M, 1, batchM * N)
    if batch_split_factor > 1:
        dim_list = (batch_split_factor, M, batchM * N // batch_split_factor)
    return input_.new_empty(dim_list, dtype=dtype)


@impl(m, "npu_trans_quant_param")
def npu_trans_quant_param_meta(scale, offset=None, round_mode=0):
    scale_dim_num = scale.dim()
    torch._check(
        scale_dim_num == 1 or (scale_dim_num == 2 and scale.size(0) == 1),
        lambda: "the scale shape support only (1, ) and (1, n)" + ops_error(ErrCode.VALUE),
    )
    torch._check(
        round_mode == 0 or round_mode == 1,
        lambda: "round_mode should be 0 or 1, but round_mode is " + int(round_mode) + ops_error(ErrCode.VALUE),
    )
    output_shape = scale.size()
    if scale_dim_num == 1:
        scale_first_dim = scale.size(0)
        dim_max = scale_first_dim
        if offset is not None:
            offset_first_dim = offset.size(0)
            dim_max = max(dim_max, offset_first_dim)
            if offset_first_dim != 1 and scale_first_dim != 1:
                torch._check(
                    offset_first_dim == scale_first_dim,
                    lambda: "offset first dim should be equal to scale first dim if none of them are equal to one" + ops_error(ErrCode.VALUE),
                )
        output_shape = (dim_max)
    else:
        if offset is not None:
            torch._check(
                scale.size() == offset.size(),
                lambda: "when the input shape of scale is (1, n), shape of scale and offset should be equal" + ops_error(ErrCode.VALUE),
            )
    return scale.new_empty(output_shape, dtype=torch.int64)


@impl(m, "npu_quantize")
def npu_quantize_meta(self, scales, zero_points, dtype, axis=1, div_mode=True):
    torch_dtype = TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP.get(dtype, torch.int8)
    if torch_dtype == torch.quint8 or torch_dtype == torch.uint8:
        return torch.empty_like(self, dtype=torch.uint8)
    elif torch_dtype == torch.qint32 or torch_dtype == torch.int32:
        return torch.empty_like(self, dtype=torch.int32)
    elif torch_dtype == torch.int8:
        return torch.empty_like(self, dtype=torch.int8)
    elif torch_dtype == torch.float8_e4m3fn:
        return torch.empty_like(self, dtype=torch.float8_e4m3fn)
    elif torch_dtype == torch.float8_e5m2:
        return torch.empty_like(self, dtype=torch.float8_e5m2)
    elif dtype == 290:          # torch_npu.hifloat8
        return torch.empty_like(self, dtype=torch.bits8)
    elif torch_dtype == torch.quint4x2:
        dim_num = self.dim()
        if self.size(dim_num - 1) % 8:
            raise RuntimeError("If dtype is quint4x2, the last dim of input must be divided by 8" +
                               ops_error(ErrCode.NOT_SUPPORT))
        output_shape = []
        for dim in range(dim_num - 1):
            output_shape.append(self.size(dim))
        output_shape.append(self.size(dim_num - 1) // 8)
        return self.new_empty(output_shape, dtype=torch.int32)
    return torch.empty_like(self, dtype=torch.int8)


@impl(m, "npu_group_quant")
def npu_group_quant_meta(x, scale, group_index, *, offset=None, dst_dtype=None):
    if dst_dtype == torch.quint8:
        return torch.empty_like(x, dtype=torch.uint8)
    elif dst_dtype == torch.qint8:
        return torch.empty_like(x, dtype=torch.int8)
    elif dst_dtype == torch.quint4x2:
        dim_num = x.dim()
        if x.size(dim_num - 1) % 8:
            raise RuntimeError("If dst_dtype is quint4x2, last dim must be divisible by 8" +
                               ops_error(ErrCode.NOT_SUPPORT))
        output_shape = []
        for dim in range(dim_num - 1):
            output_shape.append(x.size(dim))
        output_shape.append(x.size(dim_num - 1) // 8)
        return x.new_empty(output_shape, dtype=torch.int32)
    return torch.empty_like(x, dtype=torch.int8)


@impl(m, "npu_dynamic_quant")
def npu_dynamic_quant(input_dummy, *, smooth_scales=None, group_index=None, dst_type=1, quant_mode="pertoken"):
    # default dst_type 1 is the enum of torch.int8
    dim_num = input_dummy.dim()
    scale_shape = []
    for dim in range(dim_num - 2):
        scale_shape.append(input_dummy.size(dim))
    if quant_mode == "perchannel":
        scale_shape.append(input_dummy.size(dim_num - 1))
    else:
        scale_shape.append(input_dummy.size(dim_num - 2))

    scale = input_dummy.new_empty(scale_shape, dtype=torch.float32)
    if quant_mode == "pertensor":
        scale = input_dummy.new_empty([1], dtype=torch.float32)
    torch_dtype = TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP.get(dst_type, torch.int8)
    if torch_dtype == torch.quint4x2:
        if input_dummy.size(dim_num - 1) % 8:
            raise RuntimeError("If dst_dtype is quint4x2, the last dim of input must be divisible by 8" +
                               ops_error(ErrCode.PARAM))
        scale_shape.append(input_dummy.size(dim_num - 1) // 8)
        output = input_dummy.new_empty(scale_shape, dtype=torch.int32)
    elif dst_type == 290:           # torch_npu.hifloat8
        output = torch.empty_like(input_dummy, dtype=torch.uint8)
    elif torch_dtype == torch.float8_e5m2:
        output = torch.empty_like(input_dummy, dtype=torch.float8_e5m2)
    elif torch_dtype == torch.float8_e4m3fn:
        output = torch.empty_like(input_dummy, dtype=torch.float8_e4m3fn)
    else:
        output = torch.empty_like(input_dummy, dtype=torch.int8)
    return (output, scale)


@impl(m, "npu_dynamic_quant_asymmetric")
def npu_dynamic_quant_asymmetric(input_dummy, *, smooth_scales=None, group_index=None, dst_type=1, quant_mode="pertoken"):
    # default dst_type 1 is the enum of torch.int8
    dim_num = input_dummy.dim()
    scale_offset_shape = []
    for dim in range(dim_num - 2):
        scale_offset_shape.append(input_dummy.size(dim))
    if quant_mode == "perchannel":
        scale_offset_shape.append(input_dummy.size(dim_num - 1))
    else:
        scale_offset_shape.append(input_dummy.size(dim_num - 2))
    scale = input_dummy.new_empty(scale_offset_shape, dtype=torch.float32)
    offset = input_dummy.new_empty(scale_offset_shape, dtype=torch.float32)
    if quant_mode == "pertensor":
        scale = input_dummy.new_empty([1], dtype=torch.float32)
        offset = input_dummy.new_empty([1], dtype=torch.float32)
    torch_dtype = TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP.get(dst_type, torch.int8)
    if torch_dtype == torch.quint4x2:
        if input_dummy.size(dim_num - 1) % 8:
            raise RuntimeError("If dst_dtype is quint4x2, the last dim of input must be divisible by 8" +
                               ops_error(ErrCode.PARAM))
        scale_offset_shape.append(input_dummy.size(dim_num - 1) // 8)
        output = input_dummy.new_empty(scale_offset_shape, dtype=torch.int32)
    elif dst_type == 290:           # torch_npu.hifloat8
        output = torch.empty_like(input_dummy, dtype=torch.uint8)
    elif torch_dtype == torch.float8_e5m2:
        output = torch.empty_like(input_dummy, dtype=torch.float8_e5m2)
    elif torch_dtype == torch.float8_e4m3fn:
        output = torch.empty_like(input_dummy, dtype=torch.float8_e4m3fn)
    else:
        output = torch.empty_like(input_dummy, dtype=torch.int8)
    return (output, scale, offset)


@impl(m, "npu_dynamic_mx_quant")
def npu_dynamic_mx_quant(input_dummy, *, axis=-1, round_mode="rint", dst_type=296, block_size=32, scale_alg=0):
    dim_num = input_dummy.dim()
    mxscale_shape = []
    if axis < -dim_num or axis >= dim_num:
        raise RuntimeError("Parameter axis is out of input dimension range [{0}, {1}]".format(-dim_num, dim_num - 1) +
                           ops_error(ErrCode.PARAM))
    if not (block_size % 32 == 0 and block_size > 0 and block_size <= 1024):
        raise RuntimeError("Parameter block_size must be divisible by 32 and no greater than 1024, greater than 0" +
                           ops_error(ErrCode.PARAM))
    if scale_alg not in [0, 1]:
        raise RuntimeError("Invalid scale_alg value: {scale_alg}. Expected 0 or 1." +
                            ops_error(ErrCode.PARAM))
    axis_change = axis if axis >= 0 else axis + dim_num
    for dim in range(dim_num):
        mxscale_shape.append(input_dummy.size(dim))
    mxscale_shape.append(2)

    dim_size = int(math.ceil(mxscale_shape[axis_change] / block_size))
    dim_size = (dim_size + 2 - 1) // 2
    mxscale_shape[axis_change] = dim_size

    torch_dtype = TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP.get(dst_type, torch.int8)
    if torch_dtype == torch.float8_e5m2 or dst_type == 291:
        output = torch.empty_like(input_dummy, dtype=torch.float8_e5m2)
    elif torch_dtype == torch.float8_e4m3fn or dst_type == 292:
        output = torch.empty_like(input_dummy, dtype=torch.float8_e4m3fn)
    else: # float4_e2m1, float4_e1m2
        if input_dummy.size(dim_num - 1) % 2:
            raise RuntimeError("If output dtype is float4_e2m1 or float4_e1m2, " \
                                "the last dim of input must be divisible by 2, " +
                               ops_error(ErrCode.PARAM))
        output_shape = []
        for dim in range(dim_num - 1):
            output_shape.append(input_dummy.size(dim))
        output_shape.append(input_dummy.size(dim_num - 1) // 2)
        output = input_dummy.new_empty(output_shape, dtype=torch.uint8)
    mxscale = input_dummy.new_empty(mxscale_shape, dtype=torch.uint8)
    return (output, mxscale)


@impl(m, "npu_grouped_dynamic_mx_quant")
def npu_grouped_dynamic_mx_quant(x, group_index, *, round_mode="rint", dst_type=23, blocksize=32):
    if x is None or group_index is None:
        raise RuntimeError("Input x and group_index should must not be None" + ops_error(ErrCode.VALUE))
    if x.dim() != 2:
        raise RuntimeError("Input x must be 2-dimensional, got dimNum " +
                            str(x.dim()) + ops_error(ErrCode.VALUE))
    if group_index.dim() != 1:
        raise RuntimeError("Input group_index must be 1-dimensional, got dimNum " +
                            str(group_index.dim()) + ops_error(ErrCode.VALUE))
    # zero division protection
    if blocksize != 32:
        raise RuntimeError("Parameter blocksize only supports 32,  got " +
                            str(blocksize) + ops_error(ErrCode.PARAM))
    mxscale_shape = [x.shape[0] // 2 // blocksize + group_index.shape[0], x.shape[-1], 2]

    if TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP.get(dst_type) == torch.float8_e5m2:
        output = torch.empty_like(x, dtype=torch.float8_e5m2)
    elif TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP.get(dst_type) == torch.float8_e4m3fn:
        output = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    else:
        raise RuntimeError("Parameter dst_type only supports torch.float8_e5m2(23), torch.float8_e4m3fn(24), "
                           "got " + str(dst_type) + ops_error(ErrCode.PARAM))
    mxscale = x.new_empty(mxscale_shape, dtype=torch.uint8)
    return (output, mxscale)


@impl(m, "npu_moe_compute_expert_tokens")
def npu_moe_compute_expert_tokens_meta(sorted_experts, num_experts=1):
    out = torch.zeros(num_experts, dtype=torch.int32, device='meta')
    return torch.empty_like(out)


@impl(m, "npu_anti_quant")
def npu_anti_quant_meta(x, scale, *, offset=None, dst_dtype=None, src_dtype=None):
    if dst_dtype is None:
        dst_dtype = torch.float16

    if x.dtype == torch.int32:
        x_shape = x.size()
        if len(x_shape) == 0:
            raise RuntimeError("Not supported for x is scalar when x dtype is int32" + ops_error(ErrCode.NOT_SUPPORT))
        y_shape = (*(x_shape[:-1]), x_shape[-1] * 8)
        y = x.new_empty(y_shape, dtype=dst_dtype)
        return torch.empty_like(y)
    else:
        return torch.empty_like(x, dtype=dst_dtype)


@impl(m, "npu_kronecker_quant")
def npu_kronecker_quant_meta(x, kronecker_p1, kronecker_p2, clip_ratio=1.0, dst_dtype=None):
    if dst_dtype is None:
        dst_dtype = torch.int32
    if dst_dtype != torch.int32 and dst_dtype != torch_npu.float4_e2m1fn_x2:
        raise RuntimeError("the dtype of dst_dtype must be int32, or mxfp4" + ops_error(ErrCode.NOT_SUPPORT))
    dim_num = x.dim()
    if (dst_dtype == torch_npu.float4_e2m1fn_x2):
        if dim_num != 3:
            raise RuntimeError("the dim num of input x must be 3" + ops_error(ErrCode.NOT_SUPPORT))
        output_shape = [x.size(0), x.size(dim_num - 1) * x.size(dim_num - 2) // 2]
        align_base = 64
        align_size = (x.size(dim_num - 1) * x.size(dim_num - 2) + align_base - 1) // align_base
        scale_shape = [x.size(0), align_size, 2]
        return x.new_empty(output_shape, dtype=torch.uint8), x.new_empty(scale_shape, dtype=torch.uint8)
    else:
        if x.size(dim_num - 1) % 8:
            raise RuntimeError("last dim of input x must be divisible by 8" + ops_error(ErrCode.NOT_SUPPORT))
        output_shape = []
        for dim in range(dim_num - 1):
            output_shape.append(x.size(dim))
        if dst_dtype == torch.int32:
            output_shape.append(x.size(dim_num - 1) // 8)
        scale_shape = []
        scale_shape.append(x.size(0))
        return x.new_empty(output_shape, dtype=torch.int32), x.new_empty(scale_shape, dtype=torch.float32)


@impl(m, "npu_kv_rmsnorm_rope_cache")
def npu_kv_rmsnorm_rope_cache_meta(kv, gamma, cos, sin, index, k_cache, ckv_cache, *, k_rope_scale=None,
                                   c_kv_scale=None, k_rope_offset=None, c_kv_offset=None, v=None, epsilon=1e-5,
                                   cache_mode='Norm', is_output_kv=False):
    if kv.dim() != 4:
        raise RuntimeError("4D tensor expected for input kv" + ops_error(ErrCode.PARAM))
    if v is not None and v.dim() > 0:
        if v.dtype != kv.dtype:
            raise RuntimeError("v MUST have same data type as kv!" + ops_error(ErrCode.PARAM))
        if v.dim() != 4:
            raise RuntimeError("4D tensor expected for input v" + ops_error(ErrCode.PARAM))
        if v.size(0) != kv.size(0) or v.size(1) != kv.size(1) or v.size(2) != kv.size(2):
            raise RuntimeError("v MUST have same token shape as kv!" + ops_error(ErrCode.PARAM))
    if gamma.dim() != 1:
        raise RuntimeError("1D tensor expected for input gamma" + ops_error(ErrCode.PARAM))
    if cos.dim() != 4:
        raise RuntimeError("4D tensor expected for input cos" + ops_error(ErrCode.PARAM))
    k_rope_size = []
    c_kv_size = []

    for i in range(kv.dim() - 1):
        k_rope_size.append(kv.size(i))
        c_kv_size.append(kv.size(i))

    # intermediate results as optional outputs
    if v is None:
        k_rope_size.append(cos.size(3))
        c_kv_size.append(gamma.size(0))
    else:
        k_rope_size.append(kv.size(3))
        c_kv_size.append(v.size(3))
    return (torch.empty_like(k_cache), torch.empty_like(ckv_cache),
            torch.empty(k_rope_size, dtype=kv.dtype, device=kv.device),
            torch.empty(c_kv_size, dtype=kv.dtype, device=kv.device))


@impl(m, "npu_kv_rmsnorm_rope_cache_v2")
def npu_kv_rmsnorm_rope_cache_v2_meta(kv, gamma, cos, sin, index, k_cache, ckv_cache, *, k_rope_scale=None,
                                           c_kv_scale=None, k_rope_offset=None, c_kv_offset=None, v=None, epsilon=1e-5,
                                           cache_mode='Norm', is_output_kv=False):
    if kv.dim() != 4:
        raise RuntimeError("4D tensor expected for input kv" + ops_error(ErrCode.PARAM))
    if v is not None and v.dim() > 0:
        if v.dtype != kv.dtype:
            raise RuntimeError("v MUST have same data type as kv!" + ops_error(ErrCode.PARAM))
        if v.dim() != 4:
            raise RuntimeError("4D tensor expected for input v" + ops_error(ErrCode.PARAM))
        if v.size(0) != kv.size(0) or v.size(1) != kv.size(1) or v.size(2) != kv.size(2):
            raise RuntimeError("v MUST have same token shape [B,N,S] as kv!" + ops_error(ErrCode.PARAM))
    if gamma.dim() != 1:
        raise RuntimeError("1D tensor expected for input gamma" + ops_error(ErrCode.PARAM))
    if cos.dim() != 4:
        raise RuntimeError("4D tensor expected for input cos" + ops_error(ErrCode.PARAM))
    k_rope_size = []
    c_kv_size = []
    for i in range(kv.dim() - 1):
        k_rope_size.append(kv.size(i))
        c_kv_size.append(kv.size(i))
    if v is None:
        k_rope_size.append(cos.size(3))
        c_kv_size.append(gamma.size(0))
    else:
        k_rope_size.append(kv.size(3))
        c_kv_size.append(v.size(3))
    return (torch.empty(k_rope_size, dtype=kv.dtype, device=kv.device),
            torch.empty(c_kv_size, dtype=kv.dtype, device=kv.device))


@impl(m, "npu_kv_rmsnorm_rope_cache_v2_functional")
def npu_kv_rmsnorm_rope_cache_v2_functional_meta(kv, gamma, cos, sin, index, k_cache, ckv_cache, *,
                                                      k_rope_scale=None, c_kv_scale=None, k_rope_offset=None,
                                                      c_kv_offset=None, v=None, epsilon=1e-5, cache_mode='Norm',
                                                      is_output_kv=False):
    if kv.dim() != 4:
        raise RuntimeError("4D tensor expected for input kv" + ops_error(ErrCode.PARAM))
    if v is not None and v.dim() > 0:
        if v.dtype != kv.dtype:
            raise RuntimeError("v MUST have same data type as kv!" + ops_error(ErrCode.PARAM))
        if v.dim() != 4:
            raise RuntimeError("4D tensor expected for input v" + ops_error(ErrCode.PARAM))
        if v.size(0) != kv.size(0) or v.size(1) != kv.size(1) or v.size(2) != kv.size(2):
            raise RuntimeError("v MUST have same token shape as kv!" + ops_error(ErrCode.PARAM))
    if gamma.dim() != 1:
        raise RuntimeError("1D tensor expected for input gamma" + ops_error(ErrCode.PARAM))
    if cos.dim() != 4:
        raise RuntimeError("4D tensor expected for input cos" + ops_error(ErrCode.PARAM))
    k_rope_size = []
    c_kv_size = []
    for i in range(kv.dim() - 1):
        k_rope_size.append(kv.size(i))
        c_kv_size.append(kv.size(i))
    if v is None:
        k_rope_size.append(cos.size(3))
        c_kv_size.append(gamma.size(0))
    else:
        k_rope_size.append(kv.size(3))
        c_kv_size.append(v.size(3))
    return (torch.empty(k_rope_size, dtype=kv.dtype, device=kv.device),
            torch.empty(c_kv_size, dtype=kv.dtype, device=kv.device),
            torch.empty_like(k_cache), torch.empty_like(ckv_cache))


@impl(m, "npu_qkv_rms_norm_rope_cache")
def npu_qkv_rms_norm_rope_cache_meta(qkv, q_gamma, k_gamma, cos, sin, index, q_out, k_cache, v_cache, qkv_size, head_nums, 
                                     *, k_scale=None, v_scale=None, k_offset=None, v_offset=None, epsilon=1e-6,
                                     cache_mode='PA_NZ', is_output_qkv=False):
    if qkv_size is None:
        raise RuntimeError("qkv_size must not be None" + ops_error(ErrCode.PARAM))
    if head_nums is None:
        raise RuntimeError("head_nums must not be None" + ops_error(ErrCode.PARAM))
    if len(qkv_size) != 4:
        raise RuntimeError("qkv_size must be length 4 [B, S, N, D]" + ops_error(ErrCode.PARAM))
    if len(head_nums) != 3:
        raise RuntimeError("head_nums must be length 3 [n_q, n_k, n_v]" + ops_error(ErrCode.PARAM))
    if qkv.dim() != 2:
        raise RuntimeError("2D tensor expected for input qkv" + ops_error(ErrCode.PARAM))
    if q_gamma.dim() != 1:
        raise RuntimeError("1D tensor expected for input q_gamma" + ops_error(ErrCode.PARAM))
    if k_gamma.dim() != 1:
        raise RuntimeError("1D tensor expected for input k_gamma" + ops_error(ErrCode.PARAM))
    if cos.dim() != 2:
        raise RuntimeError("2D tensor expected for input cos" + ops_error(ErrCode.PARAM))
    if sin.dim() != 2:
        raise RuntimeError("2D tensor expected for input sin" + ops_error(ErrCode.PARAM))
    q_out_before_quant_size = []
    k_out_before_quant_size = []
    v_out_before_quant_size = []
    q_out_before_quant_size.append(qkv.size(0))
    k_out_before_quant_size.append(qkv.size(0))
    v_out_before_quant_size.append(qkv.size(0))
    q_out_before_quant_size.append(head_nums[0] * qkv_size[3])
    k_out_before_quant_size.append(head_nums[1] * qkv_size[3])
    v_out_before_quant_size.append(head_nums[2] * qkv_size[3])
    if is_output_qkv:        
        return (torch.empty(q_out_before_quant_size, dtype=qkv.dtype, device=qkv.device),
                torch.empty(k_out_before_quant_size, dtype=qkv.dtype, device=qkv.device),
                torch.empty(v_out_before_quant_size, dtype=qkv.dtype, device=qkv.device))
    return (torch.empty([], dtype=qkv.dtype, device=qkv.device),
                torch.empty([], dtype=qkv.dtype, device=qkv.device),
                torch.empty([], dtype=qkv.dtype, device=qkv.device))


@impl(m, "npu_qkv_rms_norm_rope_cache_functional")
def npu_qkv_rms_norm_rope_cache_functional_meta(qkv, q_gamma, k_gamma, cos, sin, index, q_out, k_cache, v_cache, qkv_size, head_nums, 
                                        *, k_scale=None, v_scale=None, k_offset=None, v_offset=None, epsilon=1e-6,
                                        cache_mode='PA_NZ', is_output_qkv=False):
    if qkv_size is None:
        raise RuntimeError("qkv_size must not be None" + ops_error(ErrCode.PARAM))
    if head_nums is None:
        raise RuntimeError("head_nums must not be None" + ops_error(ErrCode.PARAM))
    if len(qkv_size) != 4:
        raise RuntimeError("qkv_size must be length 4 [B, S, N, D]" + ops_error(ErrCode.PARAM))
    if len(head_nums) != 3:
        raise RuntimeError("head_nums must be length 3 [n_q, n_k, n_v]" + ops_error(ErrCode.PARAM))
    if qkv.dim() != 2:
        raise RuntimeError("2D tensor expected for input qkv" + ops_error(ErrCode.PARAM))
    if q_gamma.dim() != 1:
        raise RuntimeError("1D tensor expected for input q_gamma" + ops_error(ErrCode.PARAM))
    if k_gamma.dim() != 1:
        raise RuntimeError("1D tensor expected for input k_gamma" + ops_error(ErrCode.PARAM))
    if cos.dim() != 2:
        raise RuntimeError("2D tensor expected for input cos" + ops_error(ErrCode.PARAM))
    if sin.dim() != 2:
        raise RuntimeError("2D tensor expected for input sin" + ops_error(ErrCode.PARAM))
    q_out_before_quant_size = []
    k_out_before_quant_size = []
    v_out_before_quant_size = []
    q_out_before_quant_size.append(qkv.size(0))
    k_out_before_quant_size.append(qkv.size(0))
    v_out_before_quant_size.append(qkv.size(0))
    q_out_before_quant_size.append(head_nums[0] * qkv_size[3])
    k_out_before_quant_size.append(head_nums[1] * qkv_size[3])
    v_out_before_quant_size.append(head_nums[2] * qkv_size[3])
    if is_output_qkv:        
        return (torch.empty(q_out_before_quant_size, dtype=qkv.dtype, device=qkv.device),
                torch.empty(k_out_before_quant_size, dtype=qkv.dtype, device=qkv.device),
                torch.empty(v_out_before_quant_size, dtype=qkv.dtype, device=qkv.device),
                torch.empty_like(q_out), torch.empty_like(k_cache), torch.empty_like(v_cache))
    return (torch.empty([], dtype=qkv.dtype, device=qkv.device),
            torch.empty([], dtype=qkv.dtype, device=qkv.device),
            torch.empty([], dtype=qkv.dtype, device=qkv.device),
            torch.empty_like(q_out), torch.empty_like(k_cache), torch.empty_like(v_cache))

            
@impl(m, "npu_apply_rotary_pos_emb")
def npu_apply_rotary_pos_emb_meta(query, key, cos, sin, layout=1, rotary_mode='half'):
    return (torch.empty_like(query, dtype=query.dtype), torch.empty_like(key, dtype=key.dtype))


@impl(m, "npu_quant_conv2d")
def npu_quant_conv2d(input_, weight, scale, strides, pads, dilations,
                     groups=1, offset_x=0, round_mode='rint', output_dtype=None,
                     bias=None, offset=None, input_dtype=None, weight_dtype=None):

    input_shape = input_.size()
    weight_shape = weight.size()
    scale_shape = scale.size()

    input_dim = input_.dim()
    weight_dim = weight.dim()
    scale_dim = scale.dim()

    def check_basic_inputs_dim_shape():

        torch._check(
            input_dim == weight_dim and weight_dim == INPUTS_DIM_LIMIT_QUANTCONV2D,
            lambda: "input dim or weight dim is not equal to 4, but now input dim is " + str(input_dim) +
                    ", and weight dim is " + str(weight_dim) + ops_error(ErrCode.VALUE),
        )

        torch._check(
            scale_dim == 1,
            lambda: "scale dim is not equal to 1, but now scale dim is " + str(scale_dim) + ops_error(ErrCode.VALUE),
        )

        torch._check(
            input_shape[1] == weight_shape[1] * groups,
            lambda: "input cin should equal to weight cin * groups, but now input cin is " + str(input_shape[1]) +
                    ", weight cin is " + str(weight_shape[1]) + ", and groups is " + str(groups) +
                    ops_error(ErrCode.VALUE),
        )

        torch._check(
            input_shape[1] % groups == 0,
            lambda: "input cin should be an integer multiple of groups, but now input cin is " + str(input_shape[1]) +
                    ", and groups is " + str(groups) + ops_error(ErrCode.VALUE),
        )

        torch._check(
            weight_shape[0] % groups == 0,
            lambda: "cout should be an integer multiple of groups, but now cout is " + str(weight_shape[0]) +
                    ", and groups is " + str(groups) + ops_error(ErrCode.VALUE),
        )

        torch._check(
            scale_shape[0] == weight_shape[0],
            lambda: "scale shape should equal to cout, but now scale shape is " + str(scale_shape[0]) +
                    ", and cout is " + str(weight_shape[0]) + ops_error(ErrCode.VALUE),
        )

    def check_basic_inputs_dtype():
        torch._check(
            (input_dtype is not None and weight_dtype is not None) or (input_dtype is None and weight_dtype is None),
            lambda: "input_dtype and weight_dtype are only support both None or not None, " +
                    "but got input_dtype: " + str(input_dtype) + " and weight_dtype: " +
                    str(weight_dtype) + ops_error(ErrCode.TYPE))
        if input_dtype is not None:
            torch._check((input_dtype == torch_npu.hifloat8 and weight_dtype == torch_npu.hifloat8),
                lambda: "input_dtype and weight_dtype are only support torch_npu.hifloat8, " +
                        "but got input_dtype: " + str(input_dtype) +
                        " and weight_dtype: " + str(weight_dtype) +
                        ops_error(ErrCode.TYPE))
        if input_dtype is not None:
            torch._check(((input_.dtype == torch.int8 or input_.dtype == torch.uint8) and
                (weight.dtype == torch.int8 or weight.dtype == torch.uint8)),
                lambda: "input and weight tensor dtype must be torch.int8 or torch.uint8 " +
                        "when input_dtype and weight_dtype is torch_npu.hifloat8, " +
                        "but got input tensor dtype: " + str(input_.dtype) + " and weight tensor dtype: " +
                        str(weight.dtype) + ops_error(ErrCode.TYPE))
        if input_dtype is None:
            torch._check(
                ((input_.dtype == torch.int8 and weight.dtype == torch.int8) or
                (input_.dtype == torch.float8_e4m3fn and weight.dtype == torch.float8_e4m3fn)),
                lambda: "input.dtype and weight.dtype should be torch.int8 or torch.float8_e4m3fn " +
                        "when not enable hifloat8 calculation, but got input.dtype: " + str(input_.dtype) +
                        " and weight.dtype is " + str(weight.dtype) + ops_error(ErrCode.TYPE)
            )

        torch._check(
            scale.dtype == torch.int64,
            lambda: "scale.dtype should be torch.int64, but scale.dtype is " + str(scale.dtype) +
                    ops_error(ErrCode.TYPE),
        )
        torch._check(
            (output_dtype is not None),
            lambda: "output_dtype can not be None " + ops_error(ErrCode.TYPE)
        )
        if input_.dtype == torch.int8 and input_dtype != torch_npu.hifloat8:
            torch._check(
                output_dtype == TORCH_DTYPE_MAP[torch.float16],
                lambda: "output_dtype should be torch.float16 when input.dtype is torch.int8, but now dtype is " +
                        str(output_dtype) + ops_error(ErrCode.TYPE),
            )
        elif (input_.dtype == torch.float8_e4m3fn):
            torch._check(
                (output_dtype == TORCH_DTYPE_MAP[torch.float16] or
                 output_dtype == TORCH_DTYPE_MAP[torch.bfloat16] or
                 output_dtype == TORCH_DTYPE_MAP[torch.float32]),
                lambda: "output_dtype should be one of "
                        "[torch.float16, torch.bfloat16, torch.float32] "
                        "when input.dtype is torch.float8_e4m3fn, but now output_dtype is " +
                        str(output_dtype) + ops_error(ErrCode.TYPE),
            )

        if (input_dtype == torch_npu.hifloat8):
            torch._check((output_dtype == torch_npu.hifloat8 or
                          output_dtype == TORCH_DTYPE_MAP[torch.float16] or
                          output_dtype == TORCH_DTYPE_MAP[torch.bfloat16] or
                          output_dtype == TORCH_DTYPE_MAP[torch.float32]),
                          lambda: "output_dtype should be one of " +
                                  "[torch.float16, torch.bfloat16, torch.float32, torch_npu.hifloat8] " +
                                  "when input_dtype is torch_npu.hifloat8, but now output_dtype is " +
                                  str(output_dtype) + ops_error(ErrCode.TYPE)
            )

    def check_bias_dim_shape_dtype():
        bias_dim = bias.dim()
        bias_shape = bias.size()
        torch._check(
            bias_dim == 1,
            lambda: "bias dim is not equal to 1, but now bias dim is " + str(bias_dim) + ops_error(ErrCode.VALUE),
        )

        if input_.dtype == torch.int8 and input_dtype != torch_npu.hifloat8:
            torch._check(
                bias.dtype == torch.int32,
                lambda: "bias.dtype should be torch.int32 when input.dtype is torch.int8, but bias.dtype is " +
                        str(bias.dtype) + ops_error(ErrCode.TYPE),
            )
        elif (input_dtype == torch_npu.hifloat8 or input_.dtype == torch.float8_e4m3fn):
            torch._check(
                bias.dtype == torch.float32,
                lambda: "bias.dtype should be torch.float32 when input_dtype is " +
                        "torch_npu.hifloat8 or input.dtype is float8_e4m3fn, but bias.dtype is " +
                        str(bias.dtype) + ops_error(ErrCode.TYPE),
            )

        torch._check(
            bias_shape[0] == weight_shape[0],
            lambda: "bias shape should equal to cout, but now bias shape is " + str(bias_shape[0]) + ", and cout is " +
                    str(weight_shape[0]) + ops_error(ErrCode.VALUE),
        )

    def check_attrs():
        pads_dim = len(pads)
        strides_dim = len(strides)
        dilations_dim = len(dilations)
        torch._check(
            pads_dim == ATTR_DIM_LIMIT_QUANTCONV2D and strides_dim == ATTR_DIM_LIMIT_QUANTCONV2D and
            dilations_dim == ATTR_DIM_LIMIT_QUANTCONV2D,
            lambda: "attrs's dim should be 2, but pads dim is " + str(pads_dim) + ", strides dim is "
                    + str(strides_dim) + ", dilations dim is " + str(dilations_dim) + ops_error(ErrCode.VALUE),
        )
        torch._check(
            pads[0] >= 0 and pads[1] >= 0,
            lambda: "pads's value should large or equal to 0, but pads is " + str(pads[0]) + ", "
                    + str(pads[1]) + ops_error(ErrCode.VALUE),
        )
        torch._check(
            strides[0] > 0 and strides[1] > 0,
            lambda: "strides's value should large than 0, but strides is " + str(strides[0]) + ", "
                    + str(strides[1]) + ops_error(ErrCode.VALUE),
        )
        torch._check(
            dilations[0] > 0 and dilations[1] > 0,
            lambda: "dilations's value should large than 0, but dilations is " + str(dilations[0]) + ", "
                    + str(dilations[1]) + ops_error(ErrCode.VALUE),
        )
        torch._check(
            groups >= 1,
            lambda: "groups should large than 0, but now " + str(groups) + ops_error(ErrCode.VALUE),
        )
        torch._check(
            offset_x <= 127 and offset_x >= -128,
            lambda: "offset_x should be [-128,127], but offset_x is " + str(offset_x) + ops_error(ErrCode.VALUE),
        )

    check_basic_inputs_dim_shape()
    check_basic_inputs_dtype()
    if bias is not None:
        check_bias_dim_shape_dtype()
    check_attrs()

    nout = input_shape[0]
    cout = weight_shape[0]
    hout = (input_shape[2] + pads[0] * 2 - dilations[0] * (weight_shape[2] - 1) - 1) // strides[0] + 1
    wout = (input_shape[3] + pads[1] * 2 - dilations[1] * (weight_shape[3] - 1) - 1) // strides[1] + 1

    torch._check(
        hout > 0 and wout > 0,
        lambda: "ho, wo should larger than 0, but now ho is " + str(hout) + ", and wo is " + str(wout) +
                ops_error(ErrCode.VALUE),
    )

    output_dim_list = [nout, cout, hout, wout]

    if output_dtype == TORCH_DTYPE_MAP[torch.float16] or \
       output_dtype == TORCH_DTYPE_MAP[torch.bfloat16] or \
       output_dtype == TORCH_DTYPE_MAP[torch.float32]:
        return scale.new_empty(tuple(output_dim_list), dtype=TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP[output_dtype])
    elif output_dtype == torch_npu.hifloat8:
        return scale.new_empty(tuple(output_dim_list), dtype=torch.uint8)
    else:
        raise RuntimeError("output_dtype should be one of " +
            "[torch.float16, torch.bfloat16, torch.float32, torch_npu.hifloat8], but got " +
            str(output_dtype))


@impl(m, "npu_linear")
def npu_linear_meta(input_, weight, bias=None):
    dimm = input_.size(0)
    dimn = weight.size(0)
    return input_.new_empty((dimm, dimn))


@impl(m, "npu_moe_finalize_routing")
def npu_moe_finalize_routing_meta(expanded_permuted_rows, skip1, skip2_optional, bias, scales, expanded_src_to_dst_row,
                                  expert_for_source_row, drop_pad_mode=0):
    if scales is None:
        return torch.empty_like(expanded_permuted_rows, dtype=expanded_permuted_rows.dtype)
    dimm = scales.size(0)
    if drop_pad_mode == 1 or drop_pad_mode == 3:
        # dropPad场景, expanded_permuted_rows shape 为[E, C, H]
        dimn = expanded_permuted_rows.size(2)
    else:
        # dropLess场景, expanded_permuted_rows shape 为[NUM_ROWS * K, H]
        dimn = expanded_permuted_rows.size(1)
    return expanded_permuted_rows.new_empty((dimm, dimn))


has_side_effect(torch.ops.npu.npu_prefetch.default)


@impl(m, "npu_prefetch")
def npu_prefetch_meta(self, dependency, max_size, offset=0):
    torch._check(
        max_size > 0,
        lambda: f"The max_size should be greater than zero, but got {max_size}.",
    )
    torch._check(
        offset >= 0,
        lambda: f"The offset should be nonnegative, but got {offset}.",
    )


@impl(m, "npu_swiglu")
def npu_swiglu_meta(x, dim=-1):
    output_size = []
    for i in range(x.dim()):
        output_size.append(x.size(i))
    output_size[dim] = math.floor(output_size[dim] / 2)
    return torch.empty(output_size, dtype=x.dtype, device=x.device)


@impl(m, "npu_swiglu_backward")
def npu_swiglugrad_meta(y, x, dim=-1):
    return torch.empty_like(x)


def rope_quant_kvcache(x, cos, k_cache, v_cache, size_splits, kv_output=False):
    torch._check(
        x.dim() == 3 or x.dim() == 2,
        lambda: f"The x's dim should be 2 or 3, but got {x.dim()}.",
        )
    torch._check(
        k_cache.dim() == 4,
        lambda: f"The k_cache's dim should be 4, but got {k_cache.dim()}.",
        )
    num_size_splits = len(size_splits)
    torch._check(
        num_size_splits == 3,
        lambda: f"The size_splits should be 3, but got {num_size_splits}.",
        )
    torch._check(
        size_splits[0] >= 0,
        lambda: f"size_splits[0] should not less than 0, but got {size_splits[0]}.",
        )
    batch = x.size(0)
    seqlen = x.size(1)
    k_headdim = k_cache.size(2)
    hidden_size = k_cache.size(3)
    q_headdim = 0
    if hidden_size != 0:
        q_headdim = size_splits[0] // hidden_size
    out_q_size = [batch, seqlen, q_headdim, hidden_size] if x.dim() == 3 else [batch, q_headdim, hidden_size]
    out_k_size = [0]
    out_v_size = [0]
    if kv_output:
        out_k_size = [batch, seqlen, k_headdim, hidden_size] if x.dim() == 3 else [batch, k_headdim, hidden_size]
        out_v_size = [batch, seqlen, k_headdim, hidden_size] if x.dim() == 3 else [batch, k_headdim, hidden_size]
    return (torch.empty(out_q_size, dtype=cos.dtype, device=x.device),
            torch.empty(out_k_size, dtype=cos.dtype, device=x.device),
            torch.empty(out_v_size, dtype=cos.dtype, device=x.device),
            k_cache, v_cache)


@impl(m, "npu_swiglu_quant")
def npu_swiglu_quant_meta(x, smooth_scales=None, offsets=None, group_index=None, activate_left=False, quant_mode=0,
                          group_list_type=0, dst_type=torch.int8):
    y_size = []
    scale_size = []
    for i in range(x.dim() - 1):
        y_size.append(x.size(i))
        scale_size.append(x.size(i))
    y_size.append(math.floor(x.size(x.dim() - 1) / 2))
    return (torch.empty(y_size, dtype=dst_type, device=x.device),
            torch.empty(scale_size, dtype=torch.float32, device=x.device))


@impl(m, "npu_dequant_swiglu_quant")
def npu_dequant_swiglu_quant_meta(x, weight_scale=None, activation_scale=None, bias=None, quant_scale=None,
                                  quant_offset=None, group_index=None, activate_left=False, quant_mode=0,
                                  dst_type=None, round_mode=None, activate_dim=None, swiglu_mode=0, clamp_limit=7.0,
                                  glu_alpha=1.702, glu_bias=1.0):
    y_size = []
    scale_size = []
    dst_type = dst_type if dst_type is not None else 1
    round_mode = round_mode if round_mode is not None else 0
    activate_dim = activate_dim if activate_dim is not None else -1
    select_dim = activate_dim if activate_dim >= 0 else activate_dim + x.dim()
    for i in range(x.dim()):
        if i == select_dim:
            y_size.append(x.size(i) // 2)
        else:
            y_size.append(x.size(i))

    for i in range(x.dim() - 1):
        if i == select_dim:
            scale_size.append(x.size(i) // 2)
        else:
            scale_size.append(x.size(i))

    dst_torch_dtype = TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP.get(dst_type, torch.int8)

    # fp4
    if dst_torch_dtype == torch.uint8:
        y_size[-1] = y_size[-1] // 2

    return (torch.empty(y_size, dtype=dst_torch_dtype, device=x.device),
            torch.empty(scale_size, dtype=torch.float32, device=x.device))


@impl(m, "npu_clipped_swiglu")
def npu_clipped_swiglu_meta(x, group_index=None, dim=-1, alpha=1.702, limit=7.0, bias=1.0, interleaved=True):
    output_size = []
    for i in range(x.dim()):
        output_size.append(x.size(i))
    output_size[dim] = math.floor(output_size[dim] / 2)
    return torch.empty(output_size, dtype=x.dtype, device=x.device)


@impl(m, "npu_dequant_rope_quant_kvcache")
def npu_dequant_rope_quant_kvcache_meta(x, cos, sin, k_cache, v_cache, indices, scale_k, scale_v, size_splits, *,
                                        offset_k=None, offset_v=None, weight_scale=None, activation_scale=None,
                                        bias=None, quant_mode=0, input_layout="BSND", kv_output=False,
                                        cache_mode="contiguous"):
    torch._check(
        x.dtype == torch.int32,
        lambda: f"The x's dtype should be Int32, but got {x.dtype}.",
        )
    return rope_quant_kvcache(x, cos, k_cache, v_cache, size_splits, kv_output=kv_output)


@impl(m, "npu_rope_quant_kvcache")
def npu_rope_quant_kvcache_meta(x, cos, sin, k_cache, v_cache, indices, scale_k, scale_v, size_splits, *, offset_k=None,
                                offset_v=None, quant_mode=0, input_layout="BSND", kv_output=False, cache_mode="contiguous"):
    return rope_quant_kvcache(x, cos, k_cache, v_cache, size_splits, kv_output=kv_output)


@impl(m, "npu_dequant_bias")
def npu_dequant_bias_meta(x, weight_scale, activation_scale, bias, output_dtype=None):
    if output_dtype is None:
        output_dtype = torch.float16
    if output_dtype != torch.float16 and output_dtype != torch.bfloat16:
        raise RuntimeError("Only supported output_dtype is float16 and bfloat16" + ops_error(ErrCode.NOT_SUPPORT))
    return torch.empty_like(x, dtype=output_dtype)


@impl(m, "npu_interleave_rope")
def npu_interleave_rope_meta(x, cos, sin):
    return torch.empty_like(x)


@impl(m, "npu_batch_gather_matmul")
def npu_batch_gather_matmul_meta(self, x, weight_b, indices, weight_a=None,
                                 layer_idx=0, scale=1e-3, y_offset=0, y_slice_size=-1):
    return torch.empty_like(self, dtype=self.dtype)


@impl(m, "npu_batch_gather_matmul_")
def npu_batch_gather_matmul__meta(self, x, weight_b, indices, weight_a=None,
                                 layer_idx=0, scale=1e-3, y_offset=0, y_slice_size=-1):
    return self


@impl(m, "npu_gather_backward")
def npu_gather_backward__meta(grad, self_size, dim, index, sparse_grad):
    return torch.empty(self_size, dtype=grad.dtype, device=grad.device)


@impl(m, "npu_moe_token_permute_with_routing_map")
def npu_moe_token_permute_with_routing_map_meta(tokens, routing_map, *, probs=None, num_out_tokens=None, drop_and_pad=False):
    if num_out_tokens is None:
        num_out_tokens = tokens.size(0)
    dim = 1 if drop_and_pad else 0
    out_token = num_out_tokens // routing_map.size(dim) * routing_map.size(dim)

    output_size_0 = (out_token, tokens.size(1))
    output_size_1 = (out_token,)
    output_dtype_0 = tokens.dtype
    output_dtype_1 = torch.int32
    out1 = torch.empty(output_size_0, dtype=output_dtype_0, device=tokens.device)
    out3 = torch.empty(output_size_1, dtype=output_dtype_1, device=tokens.device)
    out2 = None
    if probs is not None:
        out2 = torch.empty(output_size_1, dtype=probs.dtype, device=tokens.device)

    return out1, out2, out3


@impl(m, "npu_moe_token_permute_with_routing_map_grad")
def npu_moe_token_permute_with_routing_map_grad_meta(permuted_token_out_grad, probs_grad, sorted_indices, routing_map, experts_num, tokens_num, drop_and_pad):

    output_size_0 = (tokens_num, permuted_token_out_grad.size(1))
    output_size_1 = (tokens_num, experts_num)
    output_dtype_0 = permuted_token_out_grad.dtype
    out1 = torch.empty(output_size_0, dtype=output_dtype_0, device=permuted_token_out_grad.device)
    out2 = None
    if probs_grad is not None:
        out2 = torch.empty(output_size_1, dtype=probs_grad.dtype, device=permuted_token_out_grad.device)
    return out1, out2


@impl(m, "npu_moe_re_routing")
def npu_moe_re_routing_meta(tokens, expert_token_num_per_rank, per_token_scales=None, expert_token_num_type=1, idx_type=0):
    permute_tokens_size = []
    permute_per_token_scales_size = []
    permute_token_idx_size = []
    expert_token_num_size = []
    for i in range(tokens.dim()):
        permute_tokens_size.append(tokens.size(i))
    if per_token_scales is None:
        permute_per_token_scales_size.append(tokens.size(0))
        permute_per_token_scales_dtype = torch.float32
    else:
        for i in range(per_token_scales.dim()):
            permute_per_token_scales_size.append(per_token_scales.size(i))
        permute_per_token_scales_dtype = per_token_scales.dtype
    permute_token_idx_size.append(tokens.size(0))
    expert_token_num_size.append(expert_token_num_per_rank.size(1))
    return (torch.empty(permute_tokens_size, dtype=tokens.dtype, device=tokens.device),
            torch.empty(permute_per_token_scales_size, dtype=permute_per_token_scales_dtype, device=tokens.device),
            torch.empty(permute_token_idx_size, dtype=torch.int32, device=tokens.device),
            torch.empty(expert_token_num_size, dtype=expert_token_num_per_rank.dtype, device=tokens.device))


@impl(m, "npu_attention_worker_combine")
def npu_attention_worker_combine(schedule_context, expert_scales, layer_id, hidden_size, token_dtype=0, need_schedule=0):
    y_size = []
    next_layer_id_size = []
    y_size.append(expert_scales.size(0))
    y_size.append(hidden_size)
    next_layer_id_size.append(layer_id.size(0))
    y_dtype = torch.half
    if token_dtype == 1:
        y_dtype = torch.bfloat16
    return (torch.empty(y_size, dtype=y_dtype, device=schedule_context.device),
            torch.empty(next_layer_id_size, dtype=torch.int32, device=schedule_context.device))


@impl(m, "npu_add_rms_norm_quant")
def npu_add_rms_norm_quant(x1, x2, gamma, scales1, zero_points1=None, beta=None, scales2=None, zero_points2=None, axis=-1, epsilon=1e-06, div_mode=True, dst_type=None):
    torch._check(
        scales2 is None,
        lambda: f"scales2 should be None, but got {scales2}.",
        )
    torch._check(
        zero_points2 is None,
        lambda: f"zero_points2 should be None, but got {zero_points2}.",
        )
    torch._check(
        axis == -1,
        lambda: f"axis should be -1, but got {axis}.",
        )
    torch._check(
        div_mode is True,
        lambda: f"div_mode should be True, but got {div_mode}.",
        )
    dst_type = dst_type if dst_type is not None else 1
    dst_torch_dtype = TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP.get(dst_type, torch.int8)
    return (torch.empty(x1.size(), dtype=dst_torch_dtype, device=x1.device),
            torch.empty(x1.size(), dtype=dst_torch_dtype, device=x1.device),
            torch.empty(x1.size(), dtype=x1.dtype, device=x1.device))


@impl(m, "npu_attention_update")
def npu_attention_update_meta(lse, local_out, update_type):
    ref = local_out[0]
    ref_lse = lse[0]
    sp = len(lse)
    return (torch.empty(ref.size(), dtype=ref.dtype, device=ref.device),
            torch.empty(ref_lse.size(), dtype=ref_lse.dtype, device=ref_lse.device))


@impl(m, "npu_mrope")
def npu_mrope_meta(positions, query, key, cos_sin_cache, head_size, *, mrope_section=None, rotary_mode='half'):
    return (torch.empty_like(query), torch.empty_like(key))


@impl(m, "npu_gather_sparse_index")
def npu_gather_sparse_index(inputs, index):
    output_dim = inputs.dim() + index.dim() - 1
    torch._check(
        output_dim <= NPU_TENSOR_DIM_LIMIT,
        lambda: f"input.dim() + index.dim() - 1 must not greater than 8, but got {output_dim}.",
        )

    output_size = []
    input_dim = inputs.dim()
    input_size = inputs.size()
    if input_dim == 0:
        output_size = input_size

        return torch.empty(output_size, dtype=inputs.dtype, device=inputs.device)

    index_dim = index.dim()
    index_size = index.size()
    for i in range(index_dim):
        output_size.append(index_size[i])

    for i in range(1, input_dim):
        output_size.append(input_size[i])

    return torch.empty(output_size, dtype=inputs.dtype, device=inputs.device)


@impl(m, "npu_top_k_top_p")
def npu_top_k_top_p_meta(logits, p, k):
    return torch.empty_like(logits, dtype=logits.dtype)


@impl(m, "npu_moe_token_permute")
def npu_moe_token_permute_meta(tokens, indices, num_out_tokens=None, padded_mode=False):
    torch._check(tokens.dim() == 2, lambda: f"The dims of input tokens should be 2 dimensional, but got {tokens.dim()}-dimensional.")
    torch._check(indices.dim() == 1 or indices.dim() == 2, lambda: f"The dims of input indices should be 2 or 1 dimensional, but got {indices.dim()}-dimensional.")

    num_out_tokens_value = 0 if num_out_tokens is None else num_out_tokens
    flatten_size = indices.numel()

    if num_out_tokens_value > 0:
        actual_num_out_tokens = min(num_out_tokens_value, flatten_size)
    else:
        actual_num_out_tokens = num_out_tokens_value + flatten_size
    output_shape = (actual_num_out_tokens, tokens.size(1))

    return torch.empty(output_shape, dtype=tokens.dtype, device=tokens.device), torch.empty(indices.numel(), dtype=torch.int32, device=tokens.device)


@impl(m, "npu_moe_token_unpermute")
def npu_moe_token_unpermute_meta(permuted_tokens, sorted_indices, probs=None, padded_mode=False, restore_shape=None):
    DEFAULT_TOPK = 1

    if probs is not None:
        torch._check(probs.dim() == 2, lambda: f"The dims of input probs should be 2 dimensional, but got {probs.value().dim()}-dimensional.")

    torch._check(permuted_tokens.dim() == 2, lambda: f"The dims of input permuted_tokens should be 2 dimensional, but got {permuted_tokens.dim()}-dimensional.")
    torch._check(sorted_indices.dim() == 1, lambda: f"The dims of input sorted_indices should be 1 dimensional, but got {sorted_indices.dim()}-dimensional.")

    topk = DEFAULT_TOPK if probs is None else probs.size(1)

    output_shape = (sorted_indices.size(0) // topk, permuted_tokens.size(-1))

    return torch.empty(output_shape, dtype=permuted_tokens.dtype, device=permuted_tokens.device)


@impl(m, "npu_moe_token_permute_grad")
def npu_moe_token_permute_grad_meta(tokens, grad_permuted_tokens, indices, sorted_indices, padded_mode=False):
    torch._check(tokens.dim() == 2, lambda: f"The dims of input tokens should be 2 dimensional, but got {tokens.dim()}-dimensional.")
    torch._check(grad_permuted_tokens.dim() == 2, lambda: f"The dims of input grad_permuted_tokens should be 2 dimensional, but got {grad_permuted_tokens.dim()}-dimensional.")
    torch._check(indices.dim() == 1 or indices.dim() == 2, lambda: f"The dims of input indices should be 2 or 1 dimensional, but got {indices.dim()}-dimensional.")
    torch._check(sorted_indices.dim() == 1, lambda: f"The dims of input sorted_indices should be 1 dimensional, but got {sorted_indices.dim()}-dimensional.")

    N, D = tokens.shape
    return torch.empty((N, D), dtype=tokens.dtype, device=tokens.device)


@impl(m, "npu_moe_token_unpermute_grad")
def npu_moe_token_unpermute_grad_meta(permuted_tokens, grad_unpermuted_tokens, sorted_indices, probs=None, padded_mode=False, restore_shape=None):
    torch._check(permuted_tokens.dim() == 2, lambda: f"The dims of input permuted_tokens should be 2 dimensional, but got {permuted_tokens.dim()}-dimensional.")
    torch._check(grad_unpermuted_tokens.dim() == 2, lambda: f"The dims of input grad_unpermuted_tokens should be 2 dimensional, but got {grad_unpermuted_tokens.dim()}-dimensional.")
    torch._check(sorted_indices.dim() == 1, lambda: f"The dims of input sorted_indices should be 1 dimensional, but got {sorted_indices.dim()}-dimensional.")

    grad_permuted_tokens = torch.empty_like(permuted_tokens)
    grad_probs = torch.empty_like(probs, dtype=grad_unpermuted_tokens.dtype) if probs is not None else None

    return grad_permuted_tokens, grad_probs


@impl(m, "npu_grouped_matmul_swiglu_quant")
def npu_grouped_matmul_swiglu_quant_meta(x, weight, group_list, weight_scale, x_scale, *, bias=None, offset=None):
    batch_size = x.size(0)
    n = weight.size(2)
    output_shape = torch.empty([batch_size, n // 2], dtype=torch.int8, device=x.device)
    output_scale_shape = torch.empty([batch_size], dtype=torch.float32, device=x.device)
    output_offset_shape = torch.empty([], dtype=torch.float32, device=x.device)
    return output_shape, output_scale_shape, output_offset_shape


@impl(m, "npu_grouped_matmul_swiglu_quant_v2")
def npu_grouped_matmul_swiglu_quant_v2_meta(x, weight, weight_scale, x_scale, group_list, *, smooth_scale=None,
    weight_assist_matrix=None, bias=None, dequant_mode=0, dequant_dtype=0, quant_mode=0, quant_dtype=1,
    group_list_type=0, tuning_config=None, x_dtype=None, weight_dtype=None, weight_scale_dtype=None, x_scale_dtype=None):

    if x_dtype is not None:
        torch._check(
            x_dtype == torch_npu.float4_e1m2fn_x2 or x_dtype == torch_npu.float4_e2m1fn_x2,
            lambda: "The optional parameter x_dtype only supports mxfp4 or None, but the actual value is " + npu_dtype_to_str(x_dtype),
        )

    if weight_dtype is not None:
        torch._check(
            weight_dtype == torch_npu.float4_e1m2fn_x2 or weight_dtype == torch_npu.float4_e2m1fn_x2,
            lambda: "The optional parameter weight_dtype only supports mxfp4 or None, but the actual value is " + npu_dtype_to_str(weight_dtype),
        )

    if weight_scale_dtype is not None:
        torch._check(
            weight_scale_dtype == torch_npu.float8_e8m0fnu,
            lambda: "The weight_scale_dtype only supports float8_e8m0fnu for now, but the actual value is " + npu_dtype_to_str(weight_scale_dtype),
        )

        torch._check(x.dtype == torch.float8_e5m2 or x.dtype == torch.float8_e4m3fn or x_dtype == torch_npu.float4_e1m2fn_x2 or x_dtype == torch_npu.float4_e2m1fn_x2,
            lambda: "The x only supports mxfp8 or mxfp4 for now, but the actual value is " + npu_dtype_to_str(x.dtype),
        )
        torch._check(weight[0].dtype == torch.float8_e5m2 or weight[0].dtype == torch.float8_e4m3fn or weight_dtype == torch_npu.float4_e1m2fn_x2 or weight_dtype == torch_npu.float4_e2m1fn_x2,
            lambda: "The weight only supports mxfp8 or mxfp4 for now, but the actual value is " + npu_dtype_to_str(weight[0].dtype),
        )
    if x_scale_dtype is not None:
        torch._check(
            x_scale_dtype == torch_npu.float8_e8m0fnu,
            lambda: "The x_scale_dtype only supports float8_e8m0fnu for now, but the actual value is " + npu_dtype_to_str(x_scale_dtype),
        )
    torch._check(quant_dtype == 1 or quant_dtype == TORCH_DTYPE_MAP[torch.float8_e5m2] or quant_dtype == TORCH_DTYPE_MAP[torch.float8_e4m3fn]
        or quant_dtype == torch_npu.float4_e1m2fn_x2 or quant_dtype == torch_npu.float4_e2m1fn_x2,
        lambda: "quant_dtype only supports 1 or torch.float8_e5m2, torch.float8_e4m3fn, torch_npu.float4_e1m2fn_x2, torch_npu.float4_e2m1fn_x2 for now, but it is " + npu_dtype_to_str(quant_dtype),
        )

    batch_size = x.size(0)
    dim_n = 2
    n = weight[0].size(dim_n)
    is_a8w8_input = (x.dtype == torch.float8_e5m2 or x.dtype == torch.float8_e4m3fn) and \
                   (weight[0].dtype == torch.float8_e5m2 or weight[0].dtype == torch.float8_e4m3fn)
    is_a4w4_input = False
    if x_dtype is not None and weight_dtype is not None:
        is_a4w4_input = (x_dtype == torch_npu.float4_e1m2fn_x2 or x_dtype == torch_npu.float4_e2m1fn_x2) and \
                   (weight_dtype == torch_npu.float4_e1m2fn_x2 or weight_dtype == torch_npu.float4_e2m1fn_x2)

    FP4_IN_INT8 = 2
    weight_trans = (x.size(-1) == weight[0].size(-2))
    mxfp_multi_base_size = 2
    mxfp_divisor_size = 64
    output_n = n // mxfp_multi_base_size
    output_scale_n = n // mxfp_multi_base_size / mxfp_divisor_size
    output_n_new = ((n // mxfp_multi_base_size) * FP4_IN_INT8)
    output_scale_n_new = (math.ceil(n * FP4_IN_INT8 // mxfp_multi_base_size / mxfp_divisor_size))
    if weight[0].dtype == torch.int8:
        output_shape = torch.empty([batch_size, n // mxfp_multi_base_size], dtype=torch.int8, device=x.device)
        output_scale_shape = torch.empty([batch_size], dtype=torch.float32, device=x.device)
    if quant_dtype == TORCH_DTYPE_MAP[torch.float8_e5m2]:
        if is_a8w8_input:
            output_shape = torch.empty([batch_size, output_n], dtype=torch.float8_e5m2, device=x.device)
            output_scale_shape = torch.empty([batch_size, math.ceil(output_scale_n), mxfp_multi_base_size], dtype=torch.uint8, device=x.device)
        elif is_a4w4_input:
            if not weight_trans:
                output_shape = torch.empty([batch_size, output_n_new], dtype=torch.float8_e5m2, device=x.device)
                output_scale_shape = torch.empty([batch_size, output_scale_n_new, mxfp_multi_base_size], dtype=torch.uint8, device=x.device)
            else:
                output_shape = torch.empty([batch_size, output_n], dtype=torch.float8_e5m2, device=x.device)
                output_scale_shape = torch.empty([batch_size, math.ceil(output_scale_n), mxfp_multi_base_size], dtype=torch.uint8, device=x.device)
    elif quant_dtype == TORCH_DTYPE_MAP[torch.float8_e4m3fn]:
        if is_a8w8_input:
            output_shape = torch.empty([batch_size, output_n], dtype=torch.float8_e4m3fn, device=x.device)
            output_scale_shape = torch.empty([batch_size, math.ceil(output_scale_n), mxfp_multi_base_size], dtype=torch.uint8, device=x.device)
        elif is_a4w4_input:
            if not weight_trans:
                output_shape = torch.empty([batch_size, output_n_new], dtype=torch.float8_e4m3fn, device=x.device)
                output_scale_shape = torch.empty([batch_size, output_scale_n_new, mxfp_multi_base_size], dtype=torch.uint8, device=x.device)
            else:
                output_shape = torch.empty([batch_size, output_n], dtype=torch.float8_e4m3fn, device=x.device)
                output_scale_shape = torch.empty([batch_size, math.ceil(output_scale_n), mxfp_multi_base_size], dtype=torch.uint8, device=x.device)
    elif quant_dtype == torch_npu.float4_e1m2fn_x2 or quant_dtype == torch_npu.float4_e2m1fn_x2:
        if is_a4w4_input:
            if not weight_trans:
                output_shape = torch.empty([batch_size, output_n], dtype=torch.uint8, device=x.device)
                output_scale_shape = torch.empty([batch_size, math.ceil(output_scale_n_new), mxfp_multi_base_size], dtype=torch.uint8, device=x.device)
            else:
                output_shape = torch.empty([batch_size, output_n // FP4_IN_INT8], dtype=torch.uint8, device=x.device)
                output_scale_shape = torch.empty([batch_size, math.ceil(output_scale_n), mxfp_multi_base_size], dtype=torch.uint8, device=x.device)
    return output_shape, output_scale_shape


@impl(m, "npu_recurrent_gated_delta_rule")
def npu_recurrent_gated_delta_rule_meta(query, key, value, state, *, beta=None, scale=None, actual_seq_lengths=None, ssm_state_indices=None, num_accepted_tokens=None, g=None, gk=None):
    torch._check(value.dim() == 3, lambda: f"valueTensor dim must be 3, but got {value.dim()}.")
    out_shape = (value.size(0), value.size(1), value.size(2))

    out = torch.empty(out_shape, dtype=torch.bfloat16, device=value.device)
    return out


@impl(m, "npu_recurrent_gated_delta_rule_functional")
def npu_recurrent_gated_delta_rule_functional_meta(query, key, value, state, *, beta=None, scale=None, actual_seq_lengths=None, ssm_state_indices=None, num_accepted_tokens=None, g=None, gk=None):
    torch._check(state.dim() == 4, lambda: f"state dim must be 4, but got {state.dim()}.")
    torch._check(value.dim() == 3, lambda: f"valueTensor dim must be 3, but got {value.dim()}.")

    state_shape = (state.size(0), state.size(1), state.size(2), state.size(3))
    out_shape = (value.size(0), value.size(1), value.size(2))

    finalState = torch.empty(state_shape, dtype=torch.bfloat16, device=state.device)
    out = torch.empty(out_shape, dtype=torch.bfloat16, device=value.device)

    return out, finalState


@impl(m, "npu_moe_token_unpermute_with_routing_map")
def npu_moe_token_unpermute_with_routing_map(permuted_tokens, sorted_indices, restore_shape, *, probs=None, routing_map=None, drop_and_pad=False):
    unpermuted_tokens = torch.empty([restore_shape[0], restore_shape[1]], dtype=permuted_tokens.dtype, device=permuted_tokens.device)
    return unpermuted_tokens


@impl(m, "_npu_moe_token_unpermute_with_routing_map")
def _npu_moe_token_unpermute_with_routing_map(permuted_tokens, sorted_indices, restore_shape, *, probs=None, routing_map=None, drop_and_pad=False):
    unpermuted_tokens = torch.empty([restore_shape[0], restore_shape[1]], dtype=permuted_tokens.dtype, device=permuted_tokens.device)
    out_index = torch.empty(sorted_indices.shape, dtype=sorted_indices.dtype, device=sorted_indices.device)
    permuted_token_id = torch.empty(sorted_indices.shape, dtype=sorted_indices.dtype, device=sorted_indices.device)
    permute_probs = None
    if probs is not None:
        permute_probs = torch.empty(sorted_indices.shape, dtype=probs.dtype, device=probs.device)
    return unpermuted_tokens, out_index, permuted_token_id, permute_probs


@impl(m, "npu_moe_token_unpermute_with_routing_map_grad")
# pylint:disable = huawei-too-many-arguments
def npu_moe_token_unpermute_with_routing_map_grad(unpermuted_tokens_grad, out_index, permuted_token_id, routing_map, permuted_tokens, probs, drop_and_pad, restore_shape):
    permuted_tokens_grad_out = torch.empty([out_index.shape[0], unpermuted_tokens_grad.shape[1]], dtype=unpermuted_tokens_grad.dtype, device=unpermuted_tokens_grad.device)
    if probs is not None:
        probs_grad_out = torch.empty(probs.shape, dtype=unpermuted_tokens_grad.dtype, device=unpermuted_tokens_grad.device)
        return permuted_tokens_grad_out, probs_grad_out
    else:
        return permuted_tokens_grad_out, None


@impl(m, "npu_dynamic_block_quant")
# pylint:disable = huawei-too-many-arguments
def npu_dynamic_block_quant_meta(x, *, min_scale=0.0, round_mode="rint", dst_type=1, row_block_size=1, col_block_size=128):
    # dst_type only support torch.int8 in 910B/C
    dtype = TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP.get(dst_type, torch.float8_e5m2)
    y = torch.empty(x.shape, dtype=dtype, device=x.device)
    scale_shape = list(x.shape)

    if len(scale_shape) == 2:
        scale_shape[0] = math.ceil(scale_shape[0] / row_block_size)
        scale_shape[1] = math.ceil(scale_shape[1] / col_block_size)
    elif len(scale_shape) == 3:
        scale_shape[1] = math.ceil(scale_shape[1] / row_block_size)
        scale_shape[2] = math.ceil(scale_shape[2] / col_block_size)
    else:
        raise RuntimeError(f"Expected x to have 2 or 3 dimensions, but got {x.dim()}.")

    scale_shape = torch.Size(scale_shape)

    scale = torch.empty(scale_shape, dtype=torch.float32, device=x.device)
    return y, scale


@impl(m, "npu_gather_pa_kv_cache_functional")
def npu_gather_pa_kv_cache_functional_meta(key_cache, value_cache, block_tables, seq_lens, key, value, *, seq_offset=None, is_seq_lens_cumsum=False):
    key_out = key.new_empty(key.shape, dtype=key.dtype, device='meta')
    value_out = value.new_empty(value.shape, dtype=value.dtype, device='meta')
    return (key_out, value_out)


@impl(m, "npu_sim_exponential_")
def npu_sim_exponential__meta(self, lambd=1, generator=None):
    return torch.empty_like(self)


@impl(m, "npu_grouped_matmul_add")
def npu_grouped_matmul_add_meta(
    y,
    x,
    weight,
    group_list,
    *,
    transpose_x=True,
    transpose_weight=False,
    group_type=2,
):
    torch._check(
        group_type == 2,
        lambda: f"group_type only supports 2, but got {group_type} {ops_error(ErrCode.VALUE)}",
    )
    return y


@impl(m, "npu_cross_entropy_loss")
def npu_cross_entropy_loss_meta(
    input_,
    target,
    weight=None,
    reduction="mean",
    ignore_index=-100,
    label_smoothing=0.0,
    lse_square_scale_for_zloss=0.0,
    return_zloss=False,
):
    input_shape = input_.shape
    loss_out_shape = [
        input_shape[0],
    ]
    if reduction != "none":
        loss_out_shape = [
            1,
        ]
    log_prob_shape = input_shape
    zloss_shape = loss_out_shape
    lse_for_zloss_shape = [
        input_shape[0],
    ]
    return (
        torch.empty(loss_out_shape, dtype=input_.dtype, device=input_.device),
        torch.empty(log_prob_shape, dtype=input_.dtype, device=input_.device),
        torch.empty(zloss_shape, dtype=input_.dtype, device=input_.device),
        torch.empty(lse_for_zloss_shape, dtype=input_.dtype, device=input_.device),
    )


@impl(m, "npu_cross_entropy_loss_backward")
def npu_cross_entropy_loss_backward_meta(
    grad_loss,
    log_prob,
    target,
    weight=None,
    grad_zloss=None,
    lse_for_zloss=None,
    reduction='mean',
    ignore_index=-100,
    label_smoothing=0.0,
    lse_square_scale_for_zloss=0.0
):
    result = torch.empty_like(log_prob)

    return result


@impl(m, "npu_apply_adam_w.out")
def npu_apply_adam_w_meta(
    beta1_power,
    beta2_power,
    lr,
    weight_decay,
    beta1,
    beta2,
    epsilon,
    grad,
    max_grad_norm,
    amsgrad,
    maximize,
    *,
    out,
):
    return out[0], out[1], out[2]


@impl(m, "npu_conv2d")
def npu_conv2d_meta(input_, weight, bias, strides, pads, dilations, groups):

    input_shape = input_.size()
    weight_shape = weight.size()

    nout = input_shape[0]
    cout = weight_shape[0]
    hout = (
        input_shape[2] + pads[0] * 2 - dilations[0] * (weight_shape[2] - 1) - 1
    ) // strides[0] + 1
    wout = (
        input_shape[3] + pads[1] * 2 - dilations[1] * (weight_shape[3] - 1) - 1
    ) // strides[1] + 1

    torch._check(
        hout > 0 and wout > 0,
        lambda: "ho, wo should larger than 0, but now ho is "
        + str(hout)
        + ", and wo is "
        + str(wout)
        + ops_error(ErrCode.VALUE),
    )

    output_dim_list = [nout, cout, hout, wout]

    return torch.empty(tuple(output_dim_list), dtype=input_.dtype, device=input_.device)


@impl(m, "npu_conv2d_backward")
def npu_conv2d_backward_meta(x, grad_output, weight, stride, padding, dilation, groups, output_mask):
    Co = weight.size(0)

    result3_shape = (Co,)

    result_1 = torch.empty(x.size(), dtype=x.dtype, device='meta')
    result_2 = torch.empty(weight.size(), dtype=weight.dtype, device='meta')
    if output_mask[2]:
        result_3 = torch.empty(result3_shape, dtype=x.dtype, device='meta')
    else:
        result_3 = None

    return (result_1, result_2, result_3)


has_side_effect(torch.ops.npu.npu_attention_to_ffn.default)


@impl(m, "npu_attention_to_ffn")
def npu_attention_to_ffn_meta(x, session_id, micro_batch_id, layer_id, expert_ids, expert_rank_table, group, world_size,
                              ffn_token_info_table_shape, ffn_token_data_shape, attn_token_info_table_shape, moe_expert_num,
                              scales=None, active_mask=None, quant_mode=0, sync_flag=0, ffn_start_rank_id=0):
    return


has_side_effect(torch.ops.npu.npu_ffn_to_attention.default)


@impl(m, "npu_ffn_to_attention")
def npu_ffn_to_attention_meta(x, session_ids, micro_batch_ids, token_ids, expert_offsets, actual_token_num, group, world_size,
                              token_info_table_shape, token_data_shape, attn_rank_table=None):
    return


@impl(m, "repeat_interleave_backward_int")
def npu_repeat_interleave_backward_int_meta(grad, x, repeats, dim=None):
    result = torch.empty_like(x)

    return result


@impl(m, "npu_dynamic_mx_quant_with_dual_axis")
def npu_dynamic_mx_quant_with_dual_axis(input_dummy, *, round_mode="rint", dst_type=296, scale_alg=0):
    dim_num = input_dummy.dim()
    mxscale1_shape = []
    mxscale2_shape = []
    if scale_alg != 0:
        raise RuntimeError("Invalid scale_alg value: {0}. Expected 0.".format(scale_alg) +
                            ops_error(ErrCode.PARAM))
    last_axis = -1
    second_to_last_axis = -2
    last_axis_change = last_axis + dim_num
    second_to_last_axis_change = second_to_last_axis + dim_num
    for dim in range(dim_num):
        mxscale1_shape.append(input_dummy.size(dim))
        mxscale2_shape.append(input_dummy.size(dim))
    mxscale1_shape.append(2)
    mxscale2_shape.append(2)

    block_size = 32
    last_dim_size = int(math.ceil(mxscale1_shape[last_axis_change] / block_size))
    last_dim_size = (last_dim_size + 2 - 1) // 2
    second_to_last_dim_size = int(math.ceil(mxscale2_shape[second_to_last_axis_change] / block_size))
    second_to_last_dim_size = (second_to_last_dim_size + 2 - 1) // 2
    mxscale1_shape[last_axis_change] = last_dim_size
    mxscale2_shape[second_to_last_axis_change] = second_to_last_dim_size

    torch_dtype = TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP.get(dst_type, torch.int8)
    if torch_dtype == torch.float8_e5m2 or dst_type == torch_npu.float8_e5m2:
        y1 = torch.empty_like(input_dummy, dtype=torch.float8_e5m2)
        y2 = torch.empty_like(input_dummy, dtype=torch.float8_e5m2)
    elif torch_dtype == torch.float8_e4m3fn or dst_type == torch_npu.float8_e4m3fn:
        y1 = torch.empty_like(input_dummy, dtype=torch.float8_e4m3fn)
        y2 = torch.empty_like(input_dummy, dtype=torch.float8_e4m3fn)
    else: # float4_e2m1, float4_e1m2
        if input_dummy.size(dim_num - 1) % 2:
            raise RuntimeError("If output dtype is float4_e2m1 or float4_e1m2, " \
                                "the last dim of input must be divisible by 2, " +
                               ops_error(ErrCode.PARAM))
        y1_shape = []
        y2_shape = []
        for dim in range(dim_num - 1):
            y1_shape.append(input_dummy.size(dim))
            y2_shape.append(input_dummy.size(dim))
        y1_shape.append(input_dummy.size(dim_num - 1) // 2)
        y1 = input_dummy.new_empty(y1_shape, dtype=torch.uint8)
        y2_shape.append(input_dummy.size(dim_num - 1) // 2)
        y2 = input_dummy.new_empty(y2_shape, dtype=torch.uint8)
    mxscale1 = input_dummy.new_empty(mxscale1_shape, dtype=torch.uint8)
    mxscale2 = input_dummy.new_empty(mxscale2_shape, dtype=torch.uint8)
    return (y1, mxscale1, y2, mxscale2)


has_side_effect(torch.ops.npu.save_npugraph_tensor.default)


@impl(m, "save_npugraph_tensor")
def save_npugraph_tensor_meta(self, *, save_path=None):
    return
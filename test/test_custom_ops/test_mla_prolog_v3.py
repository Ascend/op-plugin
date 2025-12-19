import math
import unittest
import copy
import random
import torch
import numpy as np
import torch_npu
import torch_npu.npu.utils as utils
import re
from ml_dtypes import float8_e4m3fn

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


def s8_saturation(inputdata):
    inputdata = torch.where(inputdata > 127, 127, inputdata)
    inputdata = torch.where(inputdata < -128, -128, inputdata)
    return inputdata.to(torch.int8)


def s9_saturation(inputdata):
    inputdata = torch.where(inputdata > 255, 255, inputdata)
    inputdata = torch.where(inputdata < -256, -256, inputdata)
    return inputdata


def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.concatenate((-x2, x1), dim=-1)


def dynamic_quant(inputdata, smooth_scale=None):
    T = inputdata.size(0)
    H = inputdata.size(1)
    y = torch.zeros(T, H).to(torch.int32)
    scale = torch.zeros(T).to(torch.float32)

    inputdata = inputdata.reshape(T, H).to(torch.float32)
    smooth_scale = smooth_scale.to(torch.float32)
    for bs_index in range(T):
        abs_bs_tensor = torch.abs(inputdata[bs_index, :] * smooth_scale[0, :])
        scale_bs = abs_bs_tensor.max() / 127
        scale[bs_index] = scale_bs
        y[bs_index:] = torch.round(inputdata[bs_index:] * smooth_scale[0, :] / scale_bs)
    return y, scale


def dynamic_quant_without_smooth_scale(inputdata, out_deqq_shape_shape):
    T = inputdata.size(0)
    N = inputdata.size(1)
    H = inputdata.size(2)
    quant_loops = inputdata.size(0)
    eles_with_one_scale = inputdata.size(1) * inputdata.size(2)
    if len(out_deqq_shape_shape) == 3:  # [BS, N, 1], per_token per_head
        quant_loops = inputdata.size(0) * inputdata.size(1)
        eles_with_one_scale = inputdata.size(2)

    y = torch.zeros(quant_loops, eles_with_one_scale).to(torch.int32)
    scale = torch.zeros(quant_loops).to(torch.float32)
    inputdata = inputdata.reshape(quant_loops, eles_with_one_scale).to(torch.float32)

    max_values, _ = torch.max(torch.abs(inputdata), dim=-1, keepdim=True)
    scale = max_values / 127
    y = torch.round(inputdata / scale)
    y = s8_saturation(y)

    if len(out_deqq_shape_shape) == 2:  # [BS, 1], per_token
        print(f"[INFO]dynamic_quant_without_smooth_scale in per_token mode")
        return y.reshape(T, N, H), scale.reshape(quant_loops, 1).to(torch.float64)
    else:
        print(f"[INFO]dynamic_quant_without_smooth_scale in per_head mode")
        return y.reshape(T, N, H), scale.reshape(T, N, 1).to(torch.float64)


def dequant(inputdata, deq_scale_q_nope, quant_scale_ckv):
    org_shape = inputdata.size()
    quant_loops = inputdata.size(0)
    eles_with_one_scale = inputdata.size(1) * inputdata.size(2)
    if len(deq_scale_q_nope.size()) == 3:  # per_token_head
        quant_loops = inputdata.size(0) * inputdata.size(1)
        eles_with_one_scale = inputdata.size(2)
    inputdata = inputdata.reshape(quant_loops, eles_with_one_scale)
    deq_scale_q_nope = deq_scale_q_nope.reshape(quant_loops, 1)
    for quant_idx in range(quant_loops):
        inputdata[quant_idx, :] = inputdata[quant_idx, :] * quant_scale_ckv / (deq_scale_q_nope[quant_idx, 0])
    return inputdata.reshape(org_shape)


def quant(x, qscale):
    # 使用广播机制来避免显式的循环
    scaled_values = (x * qscale).round().to(torch.float32)
    s9_res = s9_saturation(scaled_values)
    s8_res_cal = s8_saturation(s9_res)

    return s8_res_cal

def ieee_754_conversion(sign, exponent_raw, mantissa, exp_len=8, mant_len=7):
    """Convert binary data into the floating point value"""
    sign_mult = -1 if sign == 1 else 1
    exponent = exponent_raw - (2 ** (exp_len - 1) - 1)
    mant_mult = 1
    for b in range(mant_len - 1, -1, -1):
        if mantissa & (2 ** b):
            mant_mult += 1 / (2 ** (mant_len - b))
    return sign_mult * (2 ** exponent) * mant_mult

def trans_torch_fp8_e8m0_to_bf16(array):
    new_array = torch.zeros_like(array).to(torch.bfloat16)
    for i in range(array.size(0)):
        for j in range(array.size(1)):
            new_value = ieee_754_conversion(0, int(array[i][j]), 0)
            new_array[i][j] = new_value
    return new_array

def get_dtype_range(dt):
    if "bfloat16" in str(dt):
        return -float.fromhex("0x1.FEp127"), float.fromhex("0x1.FEp127")
    if "uint4" in str(dt):
        return 0, 15
    if "int4" in str(dt):
        return -8, 7
    if "bool" in str(dt):
        return 0, 1
    if "float4_e2m1" in str(dt):
        return -float.fromhex("0x1.8p2"), float.fromhex("0x1.8p2")
    if "float4_e1m2" in str(dt):
        return -float.fromhex("0x1.Cp0"), float.fromhex("0x1.Cp0")
    if "float8_e8m0" in str(dt):
        return float.fromhex("0x1.p-127"), float.fromhex("0x1.p127")
    if "float8_e5m2" in str(dt):
        return -float.fromhex("0x1.Cp15"), float.fromhex("0x1.Cp15")
    if "float8_e4m3fn" in str(dt):
        return -float.fromhex("0x1.Cp8"), float.fromhex("0x1.Cp8")
    if "hifloat8" in str(dt):
        return -float.fromhex("0x1.p15"), float.fromhex("0x1.p15")
    if "complex32" in str(dt):
        dt = "float16"
    numpy_dtype = np.dtype(dt)
    if numpy_dtype.kind in "iu":
        numpy_info = np.iinfo(numpy_dtype)
    else:
        numpy_info = np.finfo(numpy_dtype)
    return numpy_info.min, numpy_info.max

def _mx_reshape_to_blocks(fp_array: np.ndarray, axis: int, block_size: int):
    fp_array = np.expand_dims(fp_array, axis=axis + 1)
    orig_shape = fp_array.shape
    pad = [[0, 0] for _ in range(len(orig_shape))]
    pad_size = orig_shape[axis] % block_size
    pad[axis][1] = block_size - pad_size
    if pad_size > 0:
        fp_array = np.pad(fp_array, pad, 'constant')
    padded_shape = fp_array.shape
    reshape = list(padded_shape)
    reshape[axis + 1] = block_size
    reshape[axis] = reshape[axis] // block_size
    fp_array = fp_array.reshape(reshape)
    return fp_array, orig_shape, padded_shape

def _mx_calculate_share_exp(fp_array: np.ndarray, scale_axis: int, mx_ele_dtype: str):
    FP32_EXPONENT_BIAS = 127
    FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)
    max_norm = get_dtype_range(mx_ele_dtype)[1]
    ele_emax = int(np.log2(max_norm))
    fp_abs_max = np.max(np.abs(fp_array), axis=scale_axis, keepdim=True)
    res = np.floor(
        np.log2(fp_abs_max.astype(np.float32) + FP32_MIN_NORMAL * (fp_abs_max == 0))
    ) - ele_emax
    res = res + FP32_EXPONENT_BIAS
    res[fp_abs_max == 0] = -float("inf")
    return res

def _mx_round_mantissa(fp_array: np.ndarray, round_mode: str):
    if round_mode in ("rint", "even"):  # tie to even
        fp_array = np.rint(fp_array)
    elif round_mode in ("round", "nearest"):  # tie away from zero
        sign = np.signbit(fp_array)
        rounded_abs = np.floor(np.abs(fp_array) + np.array([0.5], dtype=fp_array.dtype))
        fp_array = np.where(sign, -round_abs, rounded_abs)
    elif round_mode == "floor":  # round to minus infinity
        fp_array = np.floor(fp_array)
    elif round_mode == "ceil":  # round to positive infinity
        fp_array = np.ceil(fp_array)
    elif round_mode == "trunc":  # round to zero
        fp_array = np.trunc(fp_array)
    else:
        raise Exception(f"Unrecognized round method {round_mode}")
    return fp_array

def _mx_quantize_to_element_format(fp_array: np.ndarray, share_exp: np.ndarray,
                                   mx_ele_dtype: str, round_mode: str):
    mx_dtype = str(mx_ele_dtype)
    match = re.search(r'e(\d+)m(\d+)', mx_dtype)
    if match:
        exp_bits = int(match.group(1))
        mantissa_bits = int(match.group(2))
    else:
        raise ValueError(f'mx element dtype [{mx_ele_dtype}] is not recognized.')
    
    ret = fp_array / (2 ** (share_exp - 127))
    private_exp = np.floor(np.log2(np.abs(ret.astype(np.float32)) + (ret == 0))).astype(fp_array.dtype, copy=False)
    # The minimum representable exponent
    min_exp = 0 if "float4_e1m2" in mx_dtype else -(2 ** (exp_bits - 1)) + 2
    private_exp = private_exp.clip(min=min_exp)
    # Scale up so appropriate number of bits are in the integer portion of the number
    ret = ret / (2 ** private_exp) * (2 ** mantissa_bits)
    ret = _mx_round_mantissa(ret, round_mode)
    # Undo scaling
    ret = ret / (2 ** mantissa_bits) * (2 ** private_exp)
    # Set values > max_norm to Inf if desired, else clamp them
    max_norm = get_dtype_range(mx_dtype)[1]
    np.clip(ret, a_min=-max_norm, a_max=max_norm, out=ret)
    return ret

def pad_to_even(tensor: np.ndarray, axis: int) ->np.ndarray:
    if not isinstance(tensor, np.ndarray):
        raise ValueError("Input must be a numpy ndarray.")
    if axis < 0 or axis >= tensor.ndim:
        raise ValueError(f"Axis {axis} is out of bounds for tensor with {tensor.ndim} dimensions.")
    
    shape = tensor.shape
    length = shape[axis]

    if length % 2 == 0:
        return tensor
    
    pad_width = [(0, 0)] * tensor.ndim
    pad_width[axis] = (0, 1)

    padded_tensor = np.pad(tensor, pad_width, mode='constant', constant_values=2 ** -127)
    return padded_tensor

def interleave(tensor:np.ndarray, axis: int, n_group: int = 2)-> np.ndarray:
    if not isinstance(tensor, np.ndarray):
        raise ValueError("Input must be a numpy ndarray")
    if axis < 0 or axis >= tensor.ndim:
        raise ValueError(f"Axis {axis} is out of bounds for tensor with {tensor.ndim} dimensions.")
    length = tensor.shape[axis]
    if length % n_group != 0:
        raise ValueError(f"Axis length ({length}) must be divisible by n_group ({n_group})")
    group_length = length // n_group
    shape = list(tensor.shape)

    new_shape = (
        shape[:axis] +
        [group_length, 2] +
        shape[axis + 1:])
    reshaped = tensor.reshape(new_shape)

    transpose_order = (
        list(range(0, axis + 1)) + 
        list(range(axis + 2, len(new_shape))) + 
        [axis + 1,])
    
    transposed = reshaped.transpose(transpose_order)

    return transposed

def _mx_undo_reshape_to_blocks(fp_array: np.ndarray, axis: int, orig_shape: tuple, padded_shape: tuple):
    # Undo tile reshaping
    fp_array = fp_array.reshape(padded_shape)
    # Undo padding
    if tuple(padded_shape) != tuple(orig_shape):
        slices = [slice(0, x) for x in orig_shape]
        fp_array = fp_array[tuple(slices)]
    # Remove extra dimension
    fp_array = np.squeeze(fp_array, axis=axis + 1)
    return fp_array

def numpy_float8_e4m3fn():
    try:
        return float8_e4m3fn
    except ModuleNotFoundError:
        raise RuntimeError("ml_dtypes is needed to support float8_e4m3fn dtype!!! "
                        "Please install with pip3 install ml-dtypes")

def dynamic_mx_quant_cq(fp_array: np.ndarray, mx_ele_dtype: str = "float4_e2m1", 
                        axis: int = -1, block_size: int = 32, round_mode: str = "rint") -> tuple:
    if not isinstance(fp_array, np.ndarray):
        raise RuntimeError(f"Input tensor to be quantized should be numpy array. But got {type(fp_array)}")
    if fp_array.dtype.name not in ("bfloat16", "float16", "float32"):
        raise RuntimeError(f"Dtype of input tensor to be quantized is not supported: {fp_array.dtype.name}")
    if mx_ele_dtype not in ("float4_e2m1", "float4_e1m2", "float8_e4m3fn", "float8_e5m2"):
        raise NotImplementedError(f"Not support {mx_ele_dtype} yet!")
    axis = len(fp_array.shape) + axis if axis < 0 else axis
    fp_array, orig_shape, padded_shape = _mx_reshape_to_blocks(fp_array, axis, block_size)

    share_exp = _mx_calculate_share_exp(fp_array, scale_axis=axis + 1, mx_ele_dtype=mx_ele_dtype)
    scale_emax = 2 ** 7 - 1  # 8 for E8M0

    share_exp[(share_exp - 127) > scale_emax] = float("NaN")
    share_exp[(share_exp - 127) < -scale_emax] = -scale_emax
    ele_array = _mx_quantize_to_element_format(fp_array, share_exp, mx_ele_dtype, round_mode)
    # Undo reshape
    ele_array =_mx_undo_reshape_to_blocks(ele_array, axis, orig_shape, padded_shape)
    share_exp = np.squeeze(share_exp, axis=axis + 1)
    scale_array = share_exp
    if ele_array.dtype.name == "bfloat16":
        ele_array = ele_array.astype("float32", copy=False)
    # NPU will cast NaN (with or without sign) to positive ZERO (sign is dropped)
    ele_array = np.nan_to_num(ele_array, nan=0.0, copy=False)
    ele_array = ele_array.astype(numpy_float8_e4m3fn(), copy=False)
    # Cube only supports even scales, need to pad zero
    scale_array_pad = pad_to_even(scale_array, axis=axis)
    result_shape = copy.deepcopy(list(scale_array_pad.shape))
    result_shape.append(2)

    result_shape[axis] = scale_array_pad.shape[axis] // 2
    # When axis is -1, do not need interleave
    if axis != (len(fp_array.shape) - 1):
        scale_array_pad = interleave(scale_array_pad, axis=axis)
    
    scale_array_pad = scale_array_pad.reshape(result_shape)
    scale_array = scale_array_pad.astype("uint8", copy=False)
    return scale_array, ele_array

def quant_ckv_per_tensor(input, quant_scale_ckv):
    scaled_value = input * quant_scale_ckv
    scaled_value = np.round(scaled_value, 8)
    scaled_value = scaled_value.astype(numpy_float8_e4m3fn(), copy=False)
    return scaled_value

def dynamic_mx_quant_qn(x):
    scale_max = np.float32(448.0)
    x = x.astype("float32")
    input_mul = x
    input_abs = np.abs(input_mul)
    input_max = np.max(input_abs, axis=-1, keepdims=True)
    scale = input_max * (np.float32(1.0) / scale_max)
    input_scaled = input_mul / scale
    output_data = input_scaled.astype("float8_e4m3fn", copy=False)
    return scale, output_data
class TestPromptFlashAttetion(TestCase):
    def baseline(self, token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache, cache_index, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv,
            cache_mode, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uqqr, dequant_scale_w_dkvkr, quant_scale_ckv,
            smooth_scale_cq, query_norm_flag, weight_quant_mode, kv_cache_quant_mode,
            query_quant_mode, ckvkr_repo_mode, quant_scale_repo_mode, tile_size, k_nope_clip_alpha, mla_param, qc_qr_scale=1.0, kc_scale=1.0):

        B = mla_param['B']
        S1 = mla_param['S1']
        S2 = mla_param['S2']
        D = mla_param['D']
        Dr = mla_param['Dr']
        N1 = mla_param['N1']
        N2 = mla_param['N2']
        He = mla_param['He']
        Hckv = mla_param['Hckv']
        Hcq = mla_param['Hcq']
        BlockNum = mla_param['BlockNum']
        BlockSize = mla_param['BlockSize']
        T = mla_param['T']
        out_deqq_shape_shape = mla_param["out_deqq_shape_shape"]
        index_table = cache_index

        cos = rope_cos
        sin = rope_sin

        dequant_scale_qcqr = None
        dequant_scale_q_nope = None

        if not mla_param["t_flag"]:
            T = B * S1
            token_x = token_x.reshape(T, He)
            cos = cos.reshape(T, Dr)
            sin = sin.reshape(T, Dr)
            index_table = index_table.reshape(T)

        # Matmul1 预处理
        token_x = token_x.to(torch.int32)
        w_dq = weight_dq.to(torch.int32)

        # matmul1 : token_x(B*S1,He) * w_dq (He,Hcq) -> matmul1_res(B*S1,Hcq)
        matmul1_res = torch.matmul(token_x, w_dq).to(torch.int32)

        # matmul1后处理
        matmul1_res = matmul1_res.to(torch.float32)
        for t_index in range(T):
            matmul1_res[t_index, :] = matmul1_res[t_index, :] * dequant_scale_x[t_index, 0]
        for h_index in range(Hcq):
            matmul1_res[:, h_index] = matmul1_res[:, h_index] * dequant_scale_w_dq[0, h_index]

        # rmsnorm1 : matmul1_res(B*S1,Hcq) * gamma_cq(Hcq) -> norm1_res(B*S1,Hcq)
        ep1 = float(rmsnorm_epsilon_cq)
        gamma1 = rmsnorm_gamma_cq
        norm1_res = matmul1_res / torch.sqrt(torch.mean(matmul1_res ** 2, dim=-1, keepdim=True) + ep1)
        norm1_res *= gamma1
        norm1_res *= qc_qr_scale

        # matmul2 预处理
        weight_uq_qr = weight_uq_qr.to(torch.int32)
        norm1_res, dequant_scale_qcqr = dynamic_quant(norm1_res, smooth_scale_cq)

        # matmul2 : norm1_res(B*S1,Hcq) * w_uq_qr(Hcq,N*(D+Dr)) -> matmul2_res(B*S1,N,(D+Dr))
        w_uq_qr = weight_uq_qr
        matmul2_res = torch.matmul(norm1_res, w_uq_qr).to(torch.int32)

        # matmul2 后处理
        matmul2_res = matmul2_res.to(torch.float32)
        for t_index in range(T):
            matmul2_res[t_index, :] = matmul2_res[t_index, :] * dequant_scale_qcqr[t_index]
        for nddr_index in range(matmul2_res.shape[1]):
            matmul2_res[:, nddr_index] = matmul2_res[:, nddr_index] * dequant_scale_w_uqqr[0, nddr_index]

        matmul2_res = matmul2_res.reshape(T, N1, D + Dr)

        # splitD1 : matmul2_res(B*S1,N,D+Dr) -> splitd1_res1(B*S1,N,D) & splitd1_res2(B*S1,N,Dr)
        splitd1_res1 = matmul2_res[:, :, :D]  # 取前 D 维度
        splitd1_res2 = matmul2_res[:, :, D:]  # 取剩余的 Dr 维度

        # matmul3 : -> splitd1_res1(B*S1,N,D) * w_uk(N,D,Hckv) -> query_mla(B,S1,N,Hckv)
        w_uk = weight_uk.to(torch.float32)
        splitd1_res1 = splitd1_res1.transpose(0, 1)
        splitd1_res1 = splitd1_res1.to(torch.bfloat16).to(torch.float32)
        query_mla = torch.zeros((N1, T, Hckv))
        for n1_index in range(N1):
            query_mla[n1_index, :, :] = torch.matmul(splitd1_res1[n1_index, :, :], w_uk[n1_index, :, :]).to(torch.float32)
        query_mla = query_mla.transpose(0, 1)
        query_mla = query_mla.to(torch.bfloat16).to(torch.float32)

        # matmul3 后处理：dynamic quant
        query_mla, dequant_scale_q_nope = dynamic_quant_without_smooth_scale(query_mla, out_deqq_shape_shape)
        query_mla = query_mla if mla_param["t_flag"] else query_mla.reshape(B, S1, N1, Hckv)

        # rotary1 : -> splitd1_res2(B*S1,N,Dr) * cos(B*S1,Dr) * sin(B*S1,Dr) -> query_rope_mla(B,S1,N,Dr)
        expanded_cos = cos.unsqueeze(1).repeat(1, N1, 1)
        expanded_sin = sin.unsqueeze(1).repeat(1, N1, 1)
        q = splitd1_res2.reshape(T, N1, int(Dr / 2), 2).transpose(3, 2).reshape(T, N1, Dr)
        query_rope_mla = (q * expanded_cos) + (rotate_half(q) * expanded_sin)
        query_rope_mla = query_rope_mla.to(torch.bfloat16).to(torch.float32)

        # rotary1 后处理：dequant
        query_rope_mla = dequant(query_rope_mla, dequant_scale_q_nope, quant_scale_ckv)

        query_rope_mla = query_rope_mla if mla_param["t_flag"] else query_rope_mla.reshape(B, S1, N1, Dr)

        # matmul4 : token_x(B*S1,He) * w_kv_kr(He,Hckv+Dr) -> matmul4_res(B*S1,Hckv+Dr)
        w_kv_kr = weight_dkv_kr.to(torch.int32)
        matmul4_res = torch.matmul(token_x, w_kv_kr).to(torch.int32).to(torch.float32)

        # matmul4 后处理
        for t_index in range(T):
            matmul4_res[t_index, :] = matmul4_res[t_index, :] * dequant_scale_x[t_index, 0]
        for h_index in range(Hckv + Dr):
            matmul4_res[:, h_index] = matmul4_res[:, h_index] * dequant_scale_w_dkvkr[0, h_index]

        # splitD2 : matmul4_res(B*S1,Hckv+Dr) -> splitd2_res1(B*S1,Hckv) & splitd2_res2(B*S1,Dr)
        splitd2_res1 = matmul4_res[:, :Hckv]  # 取前 Hckv 维度
        splitd2_res2 = matmul4_res[:, Hckv:]  # 取剩余的 Dr 维度

        # rmsnorm2 : splitd2_res1(B*S1,Hckv) * gamma_ckv(Hckv) -> norm2_res(B*S1,Hckv)
        ep2 = float(rmsnorm_epsilon_ckv)
        gamma2 = rmsnorm_gamma_ckv
        norm2_res = splitd2_res1 / torch.sqrt(torch.mean(splitd2_res1 ** 2, dim=-1, keepdim=True) + ep2)
        norm2_res *= gamma2
        norm2_res *= kc_scale

        # rmsnorm2 后处理
        norm2_res = quant(norm2_res, quant_scale_ckv)

        # scatter1 : norm2_res(B*S1,Hckv) * kv_cache(B,N2,S2,Hckv/B,B,N2,Hckv) -> kv_cache_out_mla(B,N2,S2,Hckv/B,B,N2,Hckv)
        kv_cache = copy.deepcopy(kv_cache)
        kv_cache_out_mla_shape = kv_cache.shape
        kv_cache = kv_cache.reshape(BlockNum * BlockSize, N2, Hckv)
        for i in range(T):
            for j in range(N2):
                kv_cache[index_table[i], j, :] = norm2_res[i, :]
        kv_cache_out_mla = kv_cache.reshape(kv_cache_out_mla_shape)

        # rotary2 : splitd2_res2(B*S1,Dr) * cos(B*S1,Dr) * sin(B*S1,Dr) -> rotary2_res(B*S1,Dr)
        k = splitd2_res2.reshape(T, 1, int(Dr / 2), 2).transpose(3, 2).reshape(T, Dr)
        rotary2_res = (k * cos) + (rotate_half(k) * sin)

        # scatter2 : rotary2_res(B*S1,Dr) * kr_cache(B,N2,S2,Dr/B,B,N2,Dr) -> kr_cache_out_mla(B,N2,S2,Dr/B,B,N2,Dr)
        kr_cache = copy.deepcopy(kr_cache)
        kr_cache_out_mla_shape = kr_cache.shape
        kr_cache = kr_cache.reshape(BlockNum * BlockSize, N2, Dr)

        for i in range(T):
            for j in range(N2):
                kr_cache[index_table[i], j, :] = rotary2_res[i, :]
        kr_cache_out_mla = kr_cache.reshape(kr_cache_out_mla_shape)

        return query_mla, query_rope_mla, kv_cache_out_mla, kr_cache_out_mla, dequant_scale_q_nope
    
    def baseline_mxfp8(self, token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache, cache_index, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv,
            cache_mode, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uqqr, dequant_scale_w_dkvkr, quant_scale_ckv,
            smooth_scale_cq, query_norm_flag, weight_quant_mode, kv_cache_quant_mode,
            query_quant_mode, ckvkr_repo_mode, quant_scale_repo_mode, tile_size, k_nope_clip_alpha, mla_param, qc_qr_scale=1.0, kc_scale=1.0):

        B = mla_param['B']
        S1 = mla_param['S1']
        S2 = mla_param['S2']
        D = mla_param['D']
        Dr = mla_param['Dr']
        N1 = mla_param['N1']
        N2 = mla_param['N2']
        He = mla_param['He']
        Hckv = mla_param['Hckv']
        Hcq = mla_param['Hcq']
        BlockNum = mla_param['BlockNum']
        BlockSize = mla_param['BlockSize']
        T = mla_param['T']
        out_deqq_shape_shape = mla_param["out_deqq_shape_shape"]
        index_table = cache_index

        cos = rope_cos
        sin = rope_sin

        dequant_scale_qcqr = None
        dequant_scale_q_nope = None

        if not mla_param["t_flag"]:
            T = B * S1
            token_x = token_x.reshape(T, He)
            cos = cos.reshape(T, Dr)
            sin = sin.reshape(T, Dr)
            index_table = index_table.reshape(T)

        # Matmul1 预处理
        token_x_new = token_x
        token_x_new = token_x_new.to(torch.bfloat16)
        token_x = token_x.to(torch.bfloat16)
        dequant_scale_x = trans_torch_fp8_e8m0_to_bf16(dequant_scale_x)
        xs0 = dequant_scale_x.shape[0]
        xs1 = dequant_scale_x.shape[1]
        grp_size = 32
        for xs0_idx in range(xs0):
            for xs1_idx in range(xs1):
                # copy from
                token_cur = token_x_new[xs0_idx : xs0_idx + 1, xs1_idx * grp_size : (xs1_idx + 1) * grp_size]
                # broadcast
                scale_x = dequant_scale_x[xs0_idx : xs0_idx + 1, xs1_idx : xs1_idx + 1]
                scale_x = torch.full((1, grp_size), scale_x.item())
                # mul
                token_cur = token_cur * scale_x
                # copy to
                token_x_new[xs0_idx : xs0_idx + 1, xs1_idx * grp_size : (xs1_idx + 1) * grp_size] = token_cur
        # weight_dq 反量化
        w_dq = weight_dq.to(torch.bfloat16)
        # dequant_scale_w_dq 类型转换
        deq_scale_w_dq = trans_torch_fp8_e8m0_to_bf16(dequant_scale_w_dq)
        dqs0 = deq_scale_w_dq.shape[0]
        dqs1 = deq_scale_w_dq.shape[1]
        for dqs0_idx in range(dqs0):
            for dqs1_idx in range(dqs1):
                scale_w_dq = deq_scale_w_dq[dqs0_idx : dqs0_idx + 1, dqs1_idx : dqs1_idx + 1]
                w_dq[dqs1_idx * grp_size: (dqs1_idx + 1) * grp_size, dqs0_idx : dqs0_idx + 1] *= scale_w_dq

        token_x_new = token_x_new.to(torch.float32)
        w_dq = w_dq.to(torch.float32)
        # matmul1 : token_x(B*S1,He) * w_dq (He,Hcq) -> matmul1_res(B*S1,Hcq)
        matmul1_res = torch.matmul(token_x_new, w_dq).to(torch.float32)

        # matmul1 后处理
        matmul1_res = matmul1_res.to(torch.float32)

        # rmsnorm1 : matmul1_res(B*S1,Hcq) * gamma_cq(Hcq) -> norm1_res(B*S1,Hcq)
        ep1 = float(rmsnorm_epsilon_cq)
        gamma1 = rmsnorm_gamma_cq
        norm1_res = matmul1_res / torch.sqrt(torch.mean(matmul1_res ** 2, dim=-1, keepdim=True) + ep1)
        norm1_res *= gamma1

        # matmul2 预处理
        # weight_uq_qr 类型转换
        w_uq_qr = weight_uq_qr.to(torch.bfloat16)
        # dequant_scale_w_uqqr 类型转换
        deq_scale_uqqr = trans_torch_fp8_e8m0_to_bf16(dequant_scale_w_uqqr)
        uqqrs0 = deq_scale_uqqr.shape[0]
        uqqrs1 = deq_scale_uqqr.shape[1]
        for uqqrs0_idx in range(uqqrs0):
            for uqqrs1_idx in range(uqqrs1):
                scale_uqqr = deq_scale_uqqr[uqqrs0_idx : uqqrs0_idx + 1, uqqrs1_idx : uqqrs1_idx + 1]
                w_uq_qr[uqqrs1_idx * grp_size: (uqqrs1_idx + 1) * grp_size, uqqrs0_idx : uqqrs0_idx + 1] *= scale_uqqr
        norm1_res = norm1_res.to(torch.bfloat16)
        deq_scale_qcqr_np, norm1_res_np = dynamic_mx_quant_cq(norm1_res.float().numpy(), "float8_e4m3fn")
        norm1_res = torch.tensor(norm1_res_np.astype(np.float32)).to(torch.float8_e4m3fn)
        deq_scale_qcqr = torch.from_numpy(deq_scale_qcqr_np)
        deq_scale_qcqr = deq_scale_qcqr.reshape(deq_scale_qcqr.shape[0], deq_scale_qcqr.shape[1] * deq_scale_qcqr.shape[2])
        norm1_res = norm1_res.to(torch.bfloat16)

        # deq_scale_qcqr 类型转换
        deq_scale_qcqr = trans_torch_fp8_e8m0_to_bf16(deq_scale_qcqr)
        qcqrs0 = deq_scale_qcqr.shape[0]
        qcqrs1 = deq_scale_qcqr.shape[1]
        for qcqrs0_idx in range(qcqrs0):
            for qcqrs1_idx in range(qcqrs1):
                # copy from
                normal_cur = norm1_res[qcqrs0_idx : qcqrs0_idx + 1, qcqrs1_idx * grp_size : (qcqrs1_idx + 1) * grp_size]
                # broadcast
                scale_qcqr = deq_scale_qcqr[qcqrs0_idx : qcqrs0_idx + 1, qcqrs1_idx : qcqrs1_idx + 1]
                scale_qcqr = torch.full((1, grp_size), scale_qcqr.item())
                # mul
                normal_cur *= scale_qcqr
                # copy to
                norm1_res[qcqrs0_idx : qcqrs0_idx + 1, qcqrs1_idx * grp_size : (qcqrs1_idx + 1) * grp_size] = normal_cur
        # matmul2 : norm1_res(B*S1,Hcq) * w_uq_qr(Hcq,N*(D+Dr)) -> matmul2_res(B*S1,N,(D+Dr))
        norm1_res = norm1_res.to(torch.float32)
        w_uq_qr = w_uq_qr.to(torch.float32)
        matmul2_res = torch.matmul(norm1_res, w_uq_qr).to(torch.float32)

        # matmul2 后处理
        matmul2_res = matmul2_res.reshape(T, N1, D + Dr)

        # splitD1 : matmul2_res(B*S1,N,D+Dr) -> splitd1_res1(B*S1,N,D) & splitd1_res2(B*S1,N,Dr)
        splitd1_res1 = matmul2_res[:, :, :D]  # 取前 D 维度
        splitd1_res2 = matmul2_res[:, :, D:]  # 取剩余的 Dr 维度

        # matmul3 : -> splitd1_res1(B*S1,N,D) * w_uk(N,D,Hckv) -> query_mla(B,S1,N,Hckv)
        w_uk = weight_uk.to(torch.bfloat16)
        splitd1_res1 = splitd1_res1.transpose(0, 1)
        splitd1_res1 = splitd1_res1.to(torch.bfloat16)
        query_mla = torch.zeros((N1, T, Hckv))
        for n1_index in range(N1):
            query_mla[n1_index, :, :] = torch.matmul(splitd1_res1[n1_index, :, :].to(torch.float32), w_uk[n1_index, :, :].to(torch.float32)).to(torch.float32)
        query_mla = query_mla.transpose(0, 1)
        query_mla = query_mla.to(torch.bfloat16).to(torch.float32)

        # matmul3 后处理：dynamic quant
        deq_scale_q_nope_np, out_np = dynamic_mx_quant_qn(query_mla.numpy())
        query_mla = torch.tensor(out_np.astype(np.float32)).to(torch.float8_e4m3fn)
        dequant_scale_q_nope = torch.from_numpy(deq_scale_q_nope_np)
        query_mla = query_mla if mla_param["t_flag"] else query_mla.reshape(B, S1, N1, Hckv)
        
        # rotary1 : -> splitd1_res2(B*S1,N,Dr) * cos(B*S1,Dr) * sin(B*S1,Dr) -> query_rope_mla(B,S1,N,Dr)
        expanded_cos = cos.unsqueeze(1).repeat(1, N1, 1)
        expanded_sin = sin.unsqueeze(1).repeat(1, N1, 1)
        q = splitd1_res2.reshape(T, N1, int(Dr / 2), 2).transpose(3, 2).reshape(T, N1, Dr)
        query_rope_mla = (q * expanded_cos) + (rotate_half(q) * expanded_sin)
        query_rope_mla = query_rope_mla.to(torch.bfloat16).to(torch.float32)

        # rotary1 后处理：dequant
        query_rope_mla = dequant(query_rope_mla, dequant_scale_q_nope, quant_scale_ckv)

        query_rope_mla = query_rope_mla if mla_param["t_flag"] else query_rope_mla.reshape(B, S1, N1, Dr)

        # matmul4 : token_x(B*S1,He) * w_kv_kr(He,Hckv+Dr) -> matmul4_res(B*S1,Hckv+Dr)
        w_kv_kr = weight_dkv_kr.to(torch.bfloat16)
        # dequant_scale_w_dkvkr 类型转换
        deq_scale_dkvkr = trans_torch_fp8_e8m0_to_bf16(dequant_scale_w_dkvkr)
        dkvkrs0 = deq_scale_dkvkr.shape[0]
        dkvkrs1 = deq_scale_dkvkr.shape[1]
        for dkvkrs0_idx in range(dkvkrs0):
            for dkvkrs1_idx in range(dkvkrs1):
                scale_dkvkr = deq_scale_dkvkr[dkvkrs0_idx : dkvkrs0_idx + 1, dkvkrs1_idx : dkvkrs1_idx + 1]
                w_kv_kr[dkvkrs1_idx * grp_size: (dkvkrs1_idx + 1) * grp_size, dkvkrs0_idx : dkvkrs0_idx + 1] *= scale_dkvkr

        # matmul4 后处理
        matmul4_res = torch.matmul(token_x_new.to(torch.float32), w_kv_kr.to(torch.float32)).to(torch.float32)

        # splitD2 : matmul4_res(B*S1,Hckv+Dr) -> splitd2_res1(B*S1,Hckv) & splitd2_res2(B*S1,Dr)
        splitd2_res1 = matmul4_res[:, :Hckv]  # 取前 Hckv 维度
        splitd2_res2 = matmul4_res[:, Hckv:]  # 取剩余的 Dr 维度

        # rmsnorm2 : splitd2_res1(B*S1,Hckv) * gamma_ckv(Hckv) -> norm2_res(B*S1,Hckv)
        ep2 = float(rmsnorm_epsilon_ckv)
        gamma2 = rmsnorm_gamma_ckv
        norm2_res = splitd2_res1 / torch.sqrt(torch.mean(splitd2_res1 ** 2, dim=-1, keepdim=True) + ep2)
        norm2_res *= gamma2

        # rmsnorm2 后处理
        norm2_res_np = quant_ckv_per_tensor(norm2_res.numpy(), quant_scale_ckv.numpy())
        norm2_res = torch.tensor(norm2_res_np.astype(np.float32)).to(torch.float8_e4m3fn)

        # scatter1 : norm2_res(B*S1,Hckv) * kv_cache(B,N2,S2,Hckv/B,B,N2,Hckv) -> kv_cache_out_mla(B,N2,S2,Hckv/B,B,N2,Hckv)
        kv_cache = copy.deepcopy(kv_cache)
        kv_cache_out_mla_shape = kv_cache.shape
        kv_cache = kv_cache.reshape(BlockNum * BlockSize, N2, Hckv)
        for i in range(T):
            for j in range(N2):
                kv_cache[index_table.reshape(T)[i], j, :] = norm2_res[i, :]
        kv_cache_out_mla = kv_cache.reshape(kv_cache_out_mla_shape)

        # rotary2 : splitd2_res2(B*S1,Dr) * cos(B*S1,Dr) * sin(B*S1,Dr) -> rotary2_res(B*S1,Dr)
        k = splitd2_res2.reshape(T, 1, int(Dr / 2), 2).transpose(3, 2).reshape(T, Dr)
        rotary2_res = (k * cos) + (rotate_half(k) * sin)

        # scatter2 : rotary2_res(B*S1,Dr) * kr_cache(B,N2,S2,Dr/B,B,N2,Dr) -> kr_cache_out_mla(B,N2,S2,Dr/B,B,N2,Dr)
        kr_cache = copy.deepcopy(kr_cache)
        kr_cache_out_mla_shape = kr_cache.shape
        kr_cache = kr_cache.reshape(BlockNum * BlockSize, N2, Dr)

        for i in range(T):
            for j in range(N2):
                kr_cache[index_table.reshape(T)[i], j, :] = rotary2_res[i, :]
        kr_cache_out_mla = kr_cache.reshape(kr_cache_out_mla_shape)

        return query_mla, query_rope_mla, kv_cache_out_mla, kr_cache_out_mla, dequant_scale_q_nope

    def mla_prolog_npu_v3(self, token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache, cache_index, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv,
            cache_mode, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uqqr, dequant_scale_w_dkvkr,
            quant_scale_ckv, smooth_scale_cq, query_norm_flag, weight_quant_mode, kv_cache_quant_mode,
            query_quant_mode, ckvkr_repo_mode, quant_scale_repo_mode, tile_size, k_nope_clip_alpha, qc_qr_scale=1.0, kc_scale=1.0):

        return torch_npu.npu_mla_prolog_v3(
            token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq, rmsnorm_gamma_ckv,
            rope_sin, rope_cos, kv_cache, kr_cache, cache_index=cache_index, rmsnorm_epsilon_cq=rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv,
            cache_mode=cache_mode, dequant_scale_x=dequant_scale_x, dequant_scale_w_dq=dequant_scale_w_dq,
            dequant_scale_w_uq_qr=dequant_scale_w_uqqr, dequant_scale_w_dkv_kr=dequant_scale_w_dkvkr,
            quant_scale_ckv=quant_scale_ckv, smooth_scales_cq=smooth_scale_cq, k_nope_clip_alpha=k_nope_clip_alpha, query_norm_flag=query_norm_flag,
            weight_quant_mode=weight_quant_mode, kv_cache_quant_mode=kv_cache_quant_mode, query_quant_mode=query_quant_mode, ckvkr_repo_mode=ckvkr_repo_mode,
            quant_scale_repo_mode=quant_scale_repo_mode, tile_size=tile_size, qc_qr_scale=qc_qr_scale, kc_scale=kc_scale)

    def npu_mla_prolog_v3_functional(self, token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache, cache_index, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv,
            cache_mode, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uqqr, dequant_scale_w_dkvkr,
            quant_scale_ckv, smooth_scale_cq, query_norm_flag, weight_quant_mode, kv_cache_quant_mode,
            query_quant_mode, ckvkr_repo_mode, quant_scale_repo_mode, tile_size, k_nope_clip_alpha, qc_qr_scale=1.0, kc_scale=1.0):

        return torch_npu.npu_mla_prolog_v3_functional(
            token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq, rmsnorm_gamma_ckv,
            rope_sin, rope_cos, kv_cache, kr_cache, cache_index=cache_index, rmsnorm_epsilon_cq=rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv,
            cache_mode=cache_mode, dequant_scale_x=dequant_scale_x, dequant_scale_w_dq=dequant_scale_w_dq,
            dequant_scale_w_uq_qr=dequant_scale_w_uqqr, dequant_scale_w_dkv_kr=dequant_scale_w_dkvkr,
            quant_scale_ckv=quant_scale_ckv, smooth_scales_cq=smooth_scale_cq, k_nope_clip_alpha=k_nope_clip_alpha, query_norm_flag=query_norm_flag,
            weight_quant_mode=weight_quant_mode, kv_cache_quant_mode=kv_cache_quant_mode, query_quant_mode=query_quant_mode, ckvkr_repo_mode=ckvkr_repo_mode,
            quant_scale_repo_mode=quant_scale_repo_mode, tile_size=tile_size, qc_qr_scale=qc_qr_scale, kc_scale=kc_scale)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_op_exec_mla_prolog_npu_v3(self):
        B = 2
        He = 7168
        Hcq = 1536
        Hckv = 512
        N = 32
        D = 128
        Dr = 64
        Skv = 6144
        S = 1
        Nkv = 1
        BlockSize = 128
        BlockNum = math.ceil(B * Skv / BlockSize)
        T = 8
        token_x = torch.randint(-100, 100, (B, S, He), dtype=torch.int8).npu()
        w_dq = torch.randint(-100, 100, (He, Hcq), dtype=torch.int8).npu()
        w_dq_cast = torch_npu.npu_format_cast(w_dq.contiguous(), 29)
        w_uq_qr = torch.randint(-100, 100, (Hcq, N * (D + Dr)), dtype=torch.int8).npu()
        w_uq_qr_cast = torch_npu.npu_format_cast(w_uq_qr.contiguous(), 29)
        w_uk = torch.rand(N, D, Hckv, dtype=torch.bfloat16).npu()
        w_dkv_kr = torch.randint(-100, 100, (He, Hckv + Dr), dtype=torch.int8).npu()
        w_dkv_kr_cast = torch_npu.npu_format_cast(w_dkv_kr.contiguous(), 29)
        rmsnorm_gamma_cq = torch.rand(Hcq, dtype=torch.bfloat16).npu()
        rmsnorm_gamma_ckv = torch.rand(Hckv, dtype=torch.bfloat16).npu()
        rope_sin = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
        rope_cos = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
        cache_index = torch.randint(0, B * S, (B, S), dtype=torch.int64).npu()
        kv_cache = torch.randint(-100, 100, (1, BlockNum * BlockSize * Nkv * Hckv), dtype=torch.int8).npu()
        kv_cache = kv_cache.view(BlockNum, BlockSize, Nkv, Hckv)
        kr_cache = torch.rand(1, BlockNum * BlockSize * Nkv * Dr, dtype=torch.bfloat16).npu()
        kr_cache = kr_cache.view(BlockNum, BlockSize, Nkv, Dr)
        rmsnorm_epsilon_cq = 1.0e-5
        rmsnorm_epsilon_ckv = 1.0e-5
        cache_mode = "PA_BSND"
        qc_qr_scale = 10.0
        kc_scale = 10.0
        dequant_scale_x = torch.rand(B * S, 1, dtype=torch.float32).npu()
        dequant_scale_w_dq = torch.rand(1, Hcq, dtype=torch.float32).npu()
        dequant_scale_w_uqqr = torch.rand(1, N * (D + Dr), dtype=torch.float32).npu()
        dequant_scale_w_dkvkr = torch.rand(1, Hckv + Dr, dtype=torch.float32).npu()
        quant_scale_ckv = torch.rand(1, dtype=torch.float32).npu()
        smooth_scale_cq = torch.ones(1, Hcq, dtype=torch.float32).npu()

        mla_param = {
            'B': B,
            'He': He,
            'Hcq': Hcq,
            'Hckv': Hckv,
            'N1': N,
            'D': D,
            'Dr': Dr,
            'S2': Skv,
            'S1': S,
            'N2': Nkv,
            'BlockNum': BlockNum,
            'BlockSize': BlockSize,
            't_flag': False,
            'T': T,
            "out_deqq_shape_shape": [B * S, N, 1]
        }

        kv_cache_copy = kv_cache.clone()
        kr_cache_copy = kr_cache.clone()
        query_mla, query_rope_mla, dequant_scale_q_nope_mla, query_norm_mla, dequant_scale_q_norm_mla = self.mla_prolog_npu_v3(token_x, w_dq_cast, w_uq_qr_cast, w_uk, w_dkv_kr_cast, rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache_copy, kr_cache_copy, cache_index, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv,
            cache_mode, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uqqr, dequant_scale_w_dkvkr, quant_scale_ckv, smooth_scale_cq, query_norm_flag=True, weight_quant_mode=2, kv_cache_quant_mode=1,
            query_quant_mode=1, ckvkr_repo_mode=0, quant_scale_repo_mode=0, tile_size=128, k_nope_clip_alpha=None, qc_qr_scale=qc_qr_scale, kc_scale=kc_scale)
        query_mla = query_mla.cpu()
        query_rope_mla = query_rope_mla.cpu()
        kv_cache_copy = kv_cache_copy.cpu()
        kr_cache_copy = kr_cache_copy.cpu()
        dequant_scale_q_nope_mla = dequant_scale_q_nope_mla.cpu()

        token_x = token_x.cpu()
        w_dq = w_dq.cpu()
        w_dq_cast = w_dq_cast.cpu()
        w_uq_qr_cast = w_uq_qr_cast.cpu()
        w_uk = w_uk.cpu()
        w_dkv_kr_cast = w_dkv_kr_cast.cpu()
        rmsnorm_gamma_cq = rmsnorm_gamma_cq.cpu()
        rmsnorm_gamma_ckv = rmsnorm_gamma_ckv.cpu()
        rope_sin = rope_sin.cpu()
        rope_cos = rope_cos.cpu()
        cache_index = cache_index.cpu()
        kv_cache = kv_cache.cpu()
        kr_cache = kr_cache.cpu()
        dequant_scale_x = dequant_scale_x.cpu()
        dequant_scale_w_dq = dequant_scale_w_dq.cpu()
        dequant_scale_w_uqqr = dequant_scale_w_uqqr.cpu()
        dequant_scale_w_dkvkr = dequant_scale_w_dkvkr.cpu()
        quant_scale_ckv = quant_scale_ckv.cpu()
        smooth_scale_cq = smooth_scale_cq.cpu()

        query, query_rope, kv_cache_out, kr_cache_out, dequant_scale_q_nope = self.baseline(token_x, w_dq_cast, w_uq_qr_cast, w_uk, w_dkv_kr_cast, rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache, cache_index, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv,
            cache_mode, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uqqr, dequant_scale_w_dkvkr,
            quant_scale_ckv, smooth_scale_cq, query_norm_flag=True, weight_quant_mode=2, kv_cache_quant_mode=1,
            query_quant_mode=1, ckvkr_repo_mode=0, quant_scale_repo_mode=0, tile_size=128, k_nope_clip_alpha=None, mla_param=mla_param, qc_qr_scale=qc_qr_scale, kc_scale=kc_scale)

        # query为int8类型，允许误差为1
        self.assertRtolEqual(query_mla, query, prec=1, prec16=1)
        self.assertRtolEqual(query_rope_mla.to(torch.float32), query_rope.to(torch.float32), prec=0.005, prec16=0.005)
        self.assertRtolEqual(kv_cache_copy.to(torch.float32), kv_cache_out.to(torch.float32), prec=0.005, prec16=0.005)
        self.assertRtolEqual(kr_cache_copy.to(torch.float32), kr_cache_out.to(torch.float32), prec=0.005, prec16=0.005)
        self.assertRtolEqual(dequant_scale_q_nope_mla.to(torch.float32), dequant_scale_q_nope.to(torch.float32), prec=0.005, prec16=0.005)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_op_exec_mla_prolog_npu_v3_functional(self):
        B = 2
        He = 7168
        Hcq = 1536
        Hckv = 512
        N = 32
        D = 128
        Dr = 64
        Skv = 6144
        S = 1
        Nkv = 1
        BlockSize = 128
        BlockNum = math.ceil(B * Skv / BlockSize)
        T = 8
        token_x = torch.randint(-100, 100, (B, S, He), dtype=torch.int8).npu()
        w_dq = torch.randint(-100, 100, (He, Hcq), dtype=torch.int8).npu()
        w_dq_cast = torch_npu.npu_format_cast(w_dq.contiguous(), 29)
        w_uq_qr = torch.randint(-100, 100, (Hcq, N * (D + Dr)), dtype=torch.int8).npu()
        w_uq_qr_cast = torch_npu.npu_format_cast(w_uq_qr.contiguous(), 29)
        w_uk = torch.rand(N, D, Hckv, dtype=torch.bfloat16).npu()
        w_dkv_kr = torch.randint(-100, 100, (He, Hckv + Dr), dtype=torch.int8).npu()
        w_dkv_kr_cast = torch_npu.npu_format_cast(w_dkv_kr.contiguous(), 29)
        rmsnorm_gamma_cq = torch.rand(Hcq, dtype=torch.bfloat16).npu()
        rmsnorm_gamma_ckv = torch.rand(Hckv, dtype=torch.bfloat16).npu()
        rope_sin = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
        rope_cos = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
        cache_index = torch.randint(0, B * S, (B, S), dtype=torch.int64).npu()
        kv_cache = torch.randint(-100, 100, (1, BlockNum * BlockSize * Nkv * Hckv), dtype=torch.int8).npu()
        kv_cache = kv_cache.view(BlockNum, BlockSize, Nkv, Hckv)
        kr_cache = torch.rand(1, BlockNum * BlockSize * Nkv * Dr, dtype=torch.bfloat16).npu()
        kr_cache = kr_cache.view(BlockNum, BlockSize, Nkv, Dr)
        rmsnorm_epsilon_cq = 1.0e-5
        rmsnorm_epsilon_ckv = 1.0e-5
        cache_mode = "PA_BSND"
        qc_qr_scale = 10.0
        kc_scale = 10.0
        dequant_scale_x = torch.rand(B * S, 1, dtype=torch.float32).npu()
        dequant_scale_w_dq = torch.rand(1, Hcq, dtype=torch.float32).npu()
        dequant_scale_w_uqqr = torch.rand(1, N * (D + Dr), dtype=torch.float32).npu()
        dequant_scale_w_dkvkr = torch.rand(1, Hckv + Dr, dtype=torch.float32).npu()
        quant_scale_ckv = torch.rand(1, dtype=torch.float32).npu()
        smooth_scale_cq = torch.ones(1, Hcq, dtype=torch.float32).npu()

        mla_param = {
            'B': B,
            'He': He,
            'Hcq': Hcq,
            'Hckv': Hckv,
            'N1': N,
            'D': D,
            'Dr': Dr,
            'S2': Skv,
            'S1': S,
            'N2': Nkv,
            'BlockNum': BlockNum,
            'BlockSize': BlockSize,
            't_flag': False,
            'T': T,
            "out_deqq_shape_shape": [B * S, N, 1]
        }

        query_mla, query_rope_mla, dequant_scale_q_nope_mla, _, _, kv_cache_mla, kr_cache_mla = self.npu_mla_prolog_v3_functional(token_x, w_dq_cast, w_uq_qr_cast, w_uk, w_dkv_kr_cast, rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache, cache_index, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv,
            cache_mode, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uqqr, dequant_scale_w_dkvkr, quant_scale_ckv, smooth_scale_cq, query_norm_flag=True, weight_quant_mode=2, kv_cache_quant_mode=1,
            query_quant_mode=1, ckvkr_repo_mode=0, quant_scale_repo_mode=0, tile_size=128, k_nope_clip_alpha=None, qc_qr_scale=qc_qr_scale, kc_scale=kc_scale)
        query_mla = query_mla.cpu()
        query_rope_mla = query_rope_mla.cpu()
        kv_cache_mla = kv_cache_mla.cpu()
        kr_cache_mla = kr_cache_mla.cpu()
        dequant_scale_q_nope_mla = dequant_scale_q_nope_mla.cpu()

        token_x = token_x.cpu()
        w_dq = w_dq.cpu()
        w_dq_cast = w_dq_cast.cpu()
        w_uq_qr_cast = w_uq_qr_cast.cpu()
        w_uk = w_uk.cpu()
        w_dkv_kr_cast = w_dkv_kr_cast.cpu()
        rmsnorm_gamma_cq = rmsnorm_gamma_cq.cpu()
        rmsnorm_gamma_ckv = rmsnorm_gamma_ckv.cpu()
        rope_sin = rope_sin.cpu()
        rope_cos = rope_cos.cpu()
        cache_index = cache_index.cpu()
        kv_cache = kv_cache.cpu()
        kr_cache = kr_cache.cpu()
        dequant_scale_x = dequant_scale_x.cpu()
        dequant_scale_w_dq = dequant_scale_w_dq.cpu()
        dequant_scale_w_uqqr = dequant_scale_w_uqqr.cpu()
        dequant_scale_w_dkvkr = dequant_scale_w_dkvkr.cpu()
        quant_scale_ckv = quant_scale_ckv.cpu()
        smooth_scale_cq = smooth_scale_cq.cpu()

        query, query_rope, kv_cache_out, kr_cache_out, dequant_scale_q_nope = self.baseline(token_x, w_dq_cast, w_uq_qr_cast, w_uk, w_dkv_kr_cast, rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache, cache_index, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv,
            cache_mode, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uqqr, dequant_scale_w_dkvkr,
            quant_scale_ckv, smooth_scale_cq, query_norm_flag=True, weight_quant_mode=2, kv_cache_quant_mode=1,
            query_quant_mode=1, ckvkr_repo_mode=0, quant_scale_repo_mode=0, tile_size=128, k_nope_clip_alpha=None, mla_param=mla_param, qc_qr_scale=qc_qr_scale, kc_scale=kc_scale)

        # query为int8类型，允许误差为1
        self.assertRtolEqual(query_mla, query, prec=1, prec16=1)
        self.assertRtolEqual(query_rope_mla.to(torch.float32), query_rope.to(torch.float32), prec=0.005, prec16=0.005)
        self.assertRtolEqual(kv_cache_mla.to(torch.float32), kv_cache_out.to(torch.float32), prec=0.005, prec16=0.005)
        self.assertRtolEqual(kr_cache_mla.to(torch.float32), kr_cache_out.to(torch.float32), prec=0.005, prec16=0.005)
        self.assertRtolEqual(dequant_scale_q_nope_mla.to(torch.float32), dequant_scale_q_nope.to(torch.float32), prec=0.005, prec16=0.005)
    
    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910_95'])
    def test_op_exec_mla_prolog_npu_v3_mxfp8(self):
        B = 2
        He = 7168
        Hcq = 1536
        Hckv = 512
        N = 32
        D = 128
        Dr = 64
        Skv = 6144
        S = 1
        Nkv = 1
        BlockSize = 128
        BlockNum = math.ceil(B * Skv / BlockSize)
        T = 8
        token_x = torch.rand(B, S, He).to(torch.float8_e4m3fn).npu()
        w_dq = torch.rand(He, Hcq).to(torch.float8_e4m3fn).npu()
        w_dq_cast = torch_npu.npu_format_cast(w_dq.contiguous(), 29)
        w_uq_qr = torch.rand(Hcq, N * (D + Dr)).to(torch.float8_e4m3fn).npu()
        w_uq_qr_cast = torch_npu.npu_format_cast(w_uq_qr.contiguous(), 29)
        w_uk = torch.rand(N, D, Hckv, dtype=torch.bfloat16).npu()
        w_dkv_kr = torch.rand(He, Hckv + Dr).to(torch.float8_e4m3fn).npu()
        w_dkv_kr_cast = torch_npu.npu_format_cast(w_dkv_kr.contiguous(), 29)
        rmsnorm_gamma_cq = torch.rand(Hcq, dtype=torch.bfloat16).npu()
        rmsnorm_gamma_ckv = torch.rand(Hckv, dtype=torch.bfloat16).npu()
        rope_sin = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
        rope_cos = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
        cache_index = torch.randint(0, B * S, (B, S), dtype=torch.int64).npu()
        kv_cache = torch.rand(1, BlockNum * BlockSize * Nkv * Hckv).to(torch.float8_e4m3fn).npu()
        kv_cache = kv_cache.view(BlockNum, BlockSize, Nkv, Hckv)
        kr_cache = torch.rand(1, BlockNum * BlockSize * Nkv * Dr, dtype=torch.bfloat16).npu()
        kr_cache = kr_cache.view(BlockNum, BlockSize, Nkv, Dr)
        rmsnorm_epsilon_cq = 1.0e-5
        rmsnorm_epsilon_ckv = 1.0e-5
        cache_mode = "PA_BSND"
        qc_qr_scale = 10.0
        kc_scale = 10.0
        dequant_scale_x = torch.randint(0, 100, (B * S, He // 32)).to(torch.int8).npu()
        dequant_scale_w_dq = torch.randint(0, 100, (Hcq, He // 32)).to(torch.int8).npu()
        dequant_scale_w_uqqr = torch.randint(0, 100, (N * (D + Dr), Hcq // 32)).to(torch.int8).npu()
        dequant_scale_w_dkvkr = torch.randint(0, 100, (Hckv + Dr, He // 32)).to(torch.int8).npu()
        quant_scale_ckv = torch.rand(1, dtype=torch.float32).npu()
        smooth_scale_cq = None

        mla_param = {
            'B': B,
            'He': He,
            'Hcq': Hcq,
            'Hckv': Hckv,
            'N1': N,
            'D': D,
            'Dr': Dr,
            'S2': Skv,
            'S1': S,
            'N2': Nkv,
            'BlockNum': BlockNum,
            'BlockSize': BlockSize,
            't_flag': False,
            'T': T,
            "out_deqq_shape_shape": [B * S, N, 1]
        }

        kv_cache_copy = kv_cache.clone()
        kr_cache_copy = kr_cache.clone()
        query_mla, query_rope_mla, dequant_scale_q_nope_mla, query_norm_mla, dequant_scale_q_norm_mla = self.mla_prolog_npu_v3(token_x, w_dq_cast, w_uq_qr_cast, w_uk, w_dkv_kr_cast, rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache_copy, kr_cache_copy, cache_index, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv,
            cache_mode, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uqqr, dequant_scale_w_dkvkr, quant_scale_ckv, smooth_scale_cq, query_norm_flag=False, weight_quant_mode=3, kv_cache_quant_mode=1,
            query_quant_mode=1, ckvkr_repo_mode=0, quant_scale_repo_mode=0, tile_size=128, k_nope_clip_alpha=None, qc_qr_scale=qc_qr_scale, kc_scale=kc_scale)
        query_mla = query_mla.cpu()
        query_rope_mla = query_rope_mla.cpu()
        kv_cache_copy = kv_cache_copy.cpu()
        kr_cache_copy = kr_cache_copy.cpu()
        dequant_scale_q_nope_mla = dequant_scale_q_nope_mla.cpu()

        token_x = token_x.cpu()
        w_dq = w_dq.cpu()
        w_dq_cast = w_dq_cast.cpu()
        w_uq_qr_cast = w_uq_qr_cast.cpu()
        w_uk = w_uk.cpu()
        w_dkv_kr_cast = w_dkv_kr_cast.cpu()
        rmsnorm_gamma_cq = rmsnorm_gamma_cq.cpu()
        rmsnorm_gamma_ckv = rmsnorm_gamma_ckv.cpu()
        rope_sin = rope_sin.cpu()
        rope_cos = rope_cos.cpu()
        cache_index = cache_index.cpu()
        kv_cache = kv_cache.cpu()
        kr_cache = kr_cache.cpu()
        dequant_scale_x = dequant_scale_x.cpu()
        dequant_scale_w_dq = dequant_scale_w_dq.cpu()
        dequant_scale_w_uqqr = dequant_scale_w_uqqr.cpu()
        dequant_scale_w_dkvkr = dequant_scale_w_dkvkr.cpu()
        quant_scale_ckv = quant_scale_ckv.cpu()
        smooth_scale_cq = None

        query, query_rope, kv_cache_out, kr_cache_out, dequant_scale_q_nope = self.baseline_mxfp8(token_x, w_dq, w_uq_qr, w_uk, w_dkv_kr, rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache, cache_index, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv,
            cache_mode, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uqqr, dequant_scale_w_dkvkr,
            quant_scale_ckv, smooth_scale_cq, query_norm_flag=False, weight_quant_mode=3, kv_cache_quant_mode=1,
            query_quant_mode=1, ckvkr_repo_mode=0, quant_scale_repo_mode=0, tile_size=128, k_nope_clip_alpha=None, mla_param=mla_param, qc_qr_scale=qc_qr_scale, kc_scale=kc_scale)

        self.assertRtolEqual(query_mla.to(torch.float32), query.to(torch.float32), prec=0.005, prec16=0.005)
        self.assertRtolEqual(query_rope_mla.to(torch.float32), query_rope.to(torch.float32), prec=0.005, prec16=0.005)
        self.assertRtolEqual(kv_cache_copy.to(torch.float32), kv_cache_out.to(torch.float32), prec=0.005, prec16=0.005)
        self.assertRtolEqual(kr_cache_copy.to(torch.float32), kr_cache_out.to(torch.float32), prec=0.005, prec16=0.005)
        self.assertRtolEqual(dequant_scale_q_nope_mla.to(torch.float32), dequant_scale_q_nope.to(torch.float32), prec=0.005, prec16=0.005)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910_95'])
    def test_op_exec_mla_prolog_npu_v3_functional_mxfp8(self):
        B = 2
        He = 7168
        Hcq = 1536
        Hckv = 512
        N = 32
        D = 128
        Dr = 64
        Skv = 6144
        S = 1
        Nkv = 1
        BlockSize = 128
        BlockNum = math.ceil(B * Skv / BlockSize)
        T = 8
        token_x = torch.rand(B, S, He).to(torch.float8_e4m3fn).npu()
        w_dq = torch.rand(He, Hcq).to(torch.float8_e4m3fn).npu()
        w_dq_cast = torch_npu.npu_format_cast(w_dq.contiguous(), 29)
        w_uq_qr = torch.rand(Hcq, N * (D + Dr)).to(torch.float8_e4m3fn).npu()
        w_uq_qr_cast = torch_npu.npu_format_cast(w_uq_qr.contiguous(), 29)
        w_uk = torch.rand(N, D, Hckv, dtype=torch.bfloat16).npu()
        w_dkv_kr = torch.rand(He, Hckv + Dr).to(torch.float8_e4m3fn).npu()
        w_dkv_kr_cast = torch_npu.npu_format_cast(w_dkv_kr.contiguous(), 29)
        rmsnorm_gamma_cq = torch.rand(Hcq, dtype=torch.bfloat16).npu()
        rmsnorm_gamma_ckv = torch.rand(Hckv, dtype=torch.bfloat16).npu()
        rope_sin = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
        rope_cos = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
        cache_index = torch.randint(0, B * S, (B, S), dtype=torch.int64).npu()
        kv_cache = torch.rand(1, BlockNum * BlockSize * Nkv * Hckv).to(torch.float8_e4m3fn).npu()
        kv_cache = kv_cache.view(BlockNum, BlockSize, Nkv, Hckv)
        kr_cache = torch.rand(1, BlockNum * BlockSize * Nkv * Dr, dtype=torch.bfloat16).npu()
        kr_cache = kr_cache.view(BlockNum, BlockSize, Nkv, Dr)
        rmsnorm_epsilon_cq = 1.0e-5
        rmsnorm_epsilon_ckv = 1.0e-5
        cache_mode = "PA_BSND"
        qc_qr_scale = 10.0
        kc_scale = 10.0
        dequant_scale_x = torch.randint(0, 100, (B * S, He // 32)).to(torch.int8).npu()
        dequant_scale_w_dq = torch.randint(0, 100, (Hcq, He // 32)).to(torch.int8).npu()
        dequant_scale_w_uqqr = torch.randint(0, 100, (N * (D + Dr), Hcq // 32)).to(torch.int8).npu()
        dequant_scale_w_dkvkr = torch.randint(0, 100, (Hckv + Dr, He // 32)).to(torch.int8).npu()
        quant_scale_ckv = torch.rand(1, dtype=torch.float32).npu()
        smooth_scale_cq = None

        mla_param = {
            'B': B,
            'He': He,
            'Hcq': Hcq,
            'Hckv': Hckv,
            'N1': N,
            'D': D,
            'Dr': Dr,
            'S2': Skv,
            'S1': S,
            'N2': Nkv,
            'BlockNum': BlockNum,
            'BlockSize': BlockSize,
            't_flag': False,
            'T': T,
            "out_deqq_shape_shape": [B * S, N, 1]
        }

        query_mla, query_rope_mla, dequant_scale_q_nope_mla, _, _, kv_cache_mla, kr_cache_mla = self.npu_mla_prolog_v3_functional(token_x, w_dq_cast, w_uq_qr_cast, w_uk, w_dkv_kr_cast, rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache, cache_index, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv,
            cache_mode, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uqqr, dequant_scale_w_dkvkr, quant_scale_ckv, smooth_scale_cq, query_norm_flag=False, weight_quant_mode=3, kv_cache_quant_mode=1,
            query_quant_mode=1, ckvkr_repo_mode=0, quant_scale_repo_mode=0, tile_size=128, k_nope_clip_alpha=None, qc_qr_scale=qc_qr_scale, kc_scale=kc_scale)
        query_mla = query_mla.cpu()
        query_rope_mla = query_rope_mla.cpu()
        kv_cache_mla = kv_cache_mla.cpu()
        kr_cache_mla = kr_cache_mla.cpu()
        dequant_scale_q_nope_mla = dequant_scale_q_nope_mla.cpu()

        token_x = token_x.cpu()
        w_dq = w_dq.cpu()
        w_dq_cast = w_dq_cast.cpu()
        w_uq_qr_cast = w_uq_qr_cast.cpu()
        w_uk = w_uk.cpu()
        w_dkv_kr_cast = w_dkv_kr_cast.cpu()
        rmsnorm_gamma_cq = rmsnorm_gamma_cq.cpu()
        rmsnorm_gamma_ckv = rmsnorm_gamma_ckv.cpu()
        rope_sin = rope_sin.cpu()
        rope_cos = rope_cos.cpu()
        cache_index = cache_index.cpu()
        kv_cache = kv_cache.cpu()
        kr_cache = kr_cache.cpu()
        dequant_scale_x = dequant_scale_x.cpu()
        dequant_scale_w_dq = dequant_scale_w_dq.cpu()
        dequant_scale_w_uqqr = dequant_scale_w_uqqr.cpu()
        dequant_scale_w_dkvkr = dequant_scale_w_dkvkr.cpu()
        smooth_scale_cq = None

        query, query_rope, kv_cache_out, kr_cache_out, dequant_scale_q_nope = self.baseline_mxfp8(token_x, w_dq, w_uq_qr, w_uk, w_dkv_kr, rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache, cache_index, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv,
            cache_mode, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uqqr, dequant_scale_w_dkvkr,
            quant_scale_ckv, smooth_scale_cq, query_norm_flag=False, weight_quant_mode=3, kv_cache_quant_mode=1,
            query_quant_mode=1, ckvkr_repo_mode=0, quant_scale_repo_mode=0, tile_size=128, k_nope_clip_alpha=None, mla_param=mla_param, qc_qr_scale=qc_qr_scale, kc_scale=kc_scale)

        self.assertRtolEqual(query_mla.to(torch.float32), query.to(torch.float32), prec=0.005, prec16=0.005)
        self.assertRtolEqual(query_rope_mla.to(torch.float32), query_rope.to(torch.float32), prec=0.005, prec16=0.005)
        self.assertRtolEqual(kv_cache_mla.to(torch.float32), kv_cache_out.to(torch.float32), prec=0.005, prec16=0.005)
        self.assertRtolEqual(kr_cache_mla.to(torch.float32), kr_cache_out.to(torch.float32), prec=0.005, prec16=0.005)
        self.assertRtolEqual(dequant_scale_q_nope_mla.to(torch.float32), dequant_scale_q_nope.to(torch.float32), prec=0.005, prec16=0.005)


if __name__ == "__main__":
    run_tests()
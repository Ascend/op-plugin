import math
import unittest
import copy
import random
import torch
import numpy as np
import torch_npu
import torch_npu.npu.utils as utils

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


class TestPromptFlashAttetion(TestCase):
    def baseline(self, token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv,
            cache_mode, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uqqr, dequant_scale_w_dkvkr, quant_scale_ckv,
            smooth_scale_cq, mla_param, qc_qr_scale=1.0, kc_scale=1.0):

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
        quant_scale_ckv = quant_scale_ckv[:, :1]

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

    def mla_prolog_npu_v3(self, token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv,
            cache_mode, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uqqr, dequant_scale_w_dkvkr,
            quant_scale_ckv, smooth_scale_cq, qc_qr_scale=1.0, kc_scale=1.0):

        return torch_npu.npu_mla_prolog_v3(
            token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq, rmsnorm_gamma_ckv,
            rope_sin, rope_cos, kv_cache, kr_cache, rmsnorm_epsilon_cq=rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv,
            cache_mode=cache_mode, qc_qr_scale=qc_qr_scale, kc_scale=kc_scale, cache_index=cache_index, dequant_scale_x=dequant_scale_x, dequant_scale_w_dq=dequant_scale_w_dq,
            dequant_scale_w_uq_qr=dequant_scale_w_uqqr, dequant_scale_w_dkv_kr=dequant_scale_w_dkvkr,
            quant_scale_ckv=quant_scale_ckv, smooth_scales_cq=smooth_scale_cq)

    def npu_mla_prolog_v3_functional(self, token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv,
            cache_mode, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uqqr, dequant_scale_w_dkvkr,
            quant_scale_ckv, smooth_scale_cq, qc_qr_scale=1.0, kc_scale=1.0):

        return torch_npu.npu_mla_prolog_v3_functional(
            token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq, rmsnorm_gamma_ckv,
            rope_sin, rope_cos, kv_cache, kr_cache, rmsnorm_epsilon_cq=rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv,
            cache_mode=cache_mode, qc_qr_scale=qc_qr_scale, kc_scale=kc_scale, cache_index=cache_index, dequant_scale_x=dequant_scale_x, dequant_scale_w_dq=dequant_scale_w_dq,
            dequant_scale_w_uq_qr=dequant_scale_w_uqqr, dequant_scale_w_dkv_kr=dequant_scale_w_dkvkr,
            quant_scale_ckv=quant_scale_ckv, smooth_scales_cq=smooth_scale_cq)

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
        quant_scale_ckv = torch.rand(1, Hckv, dtype=torch.float32).npu()
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
            rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache_copy, kr_cache_copy, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv,
            cache_mode, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uqqr, dequant_scale_w_dkvkr,
            quant_scale_ckv, smooth_scale_cq, qc_qr_scale, kc_scale)
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
            rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv,
            cache_mode, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uqqr, dequant_scale_w_dkvkr,
            quant_scale_ckv, smooth_scale_cq, mla_param, qc_qr_scale, kc_scale)

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
        quant_scale_ckv = torch.rand(1, Hckv, dtype=torch.float32).npu()
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
            rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv,
            cache_mode, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uqqr, dequant_scale_w_dkvkr,
            quant_scale_ckv, smooth_scale_cq, qc_qr_scale, kc_scale)
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
            rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv,
            cache_mode, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uqqr, dequant_scale_w_dkvkr,
            quant_scale_ckv, smooth_scale_cq, mla_param, qc_qr_scale, kc_scale)

        # query为int8类型，允许误差为1
        self.assertRtolEqual(query_mla, query, prec=1, prec16=1)
        self.assertRtolEqual(query_rope_mla.to(torch.float32), query_rope.to(torch.float32), prec=0.005, prec16=0.005)
        self.assertRtolEqual(kv_cache_mla.to(torch.float32), kv_cache_out.to(torch.float32), prec=0.005, prec16=0.005)
        self.assertRtolEqual(kr_cache_mla.to(torch.float32), kr_cache_out.to(torch.float32), prec=0.005, prec16=0.005)
        self.assertRtolEqual(dequant_scale_q_nope_mla.to(torch.float32), dequant_scale_q_nope.to(torch.float32), prec=0.005, prec16=0.005)


if __name__ == "__main__":
    run_tests()
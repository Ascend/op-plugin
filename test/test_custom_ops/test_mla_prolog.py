import math
import unittest
import copy
import torch
import numpy as np
import torch_npu
import torch_npu.npu.utils as utils

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestPromptFlashAttetion(TestCase):
    def baseline(self, token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv,
            cache_mode, mla_param):
        def rotate_half(x):
            x1 = x[..., :x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return torch.concatenate((-x2, x1), dim=-1)

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
        index_table = cache_index

        cos = rope_cos.to(torch.float32)
        sin = rope_sin.to(torch.float32)

        if not mla_param["t_flag"]:
            T = B * S1
            token_x = token_x.reshape(T, He).to(torch.float32)
            cos = cos.reshape(T, Dr)
            sin = sin.reshape(T, Dr)
            index_table = index_table.reshape(T)

        # matmul1 : token_x(B*S1,He) * w_dq (He,Hcq) -> matmul1_res(B*S1,Hcq)
        w_dq = weight_dq.to(torch.float32)
        matmul1_res = torch.matmul(token_x, w_dq).to(torch.float32)
        matmul1_res = matmul1_res.to(torch.bfloat16).to(torch.float32)

        # rmsnorm1 : matmul1_res(B*S1,Hcq) * gamma_cq(Hcq) -> norm1_res(B*S1,Hcq)
        ep1 = float(rmsnorm_epsilon_cq)
        gamma1 = rmsnorm_gamma_cq.to(torch.float32)
        norm1_res = matmul1_res / torch.sqrt(torch.mean(matmul1_res ** 2, dim=-1, keepdim=True) + ep1)
        norm1_res *= gamma1

        # matmul2 : norm1_res(B*S1,Hcq) * w_uq_qr(Hcq,N*(D+Dr)) -> matmul2_res(B*S1,N,(D+Dr))
        norm1_res = norm1_res.to(torch.bfloat16).to(torch.float32)
        w_uq_qr = weight_uq_qr.to(torch.float32)
        matmul2_res = torch.matmul(norm1_res, w_uq_qr).to(torch.float32)
        matmul2_res = matmul2_res.reshape(T, N1, D + Dr)
        matmul2_res = matmul2_res.to(torch.bfloat16).to(torch.float32)

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
        query_mla = query_mla if mla_param["t_flag"] else query_mla.reshape(B, S1, N1, Hckv)
        query_mla = query_mla.to(torch.bfloat16).to(torch.float32)

        # rotary1 : -> splitd1_res2(B*S1,N,Dr) * cos(B*S1,Dr) * sin(B*S1,Dr) -> query_rope_mla(B,S1,N,Dr)
        expanded_cos = cos.unsqueeze(1).repeat(1, N1, 1)
        expanded_sin = sin.unsqueeze(1).repeat(1, N1, 1)
        q = splitd1_res2.reshape(T, N1, int(Dr / 2), 2).transpose(3, 2).reshape(T, N1, Dr)
        query_rope_mla = (q * expanded_cos) + (rotate_half(q) * expanded_sin)
        query_rope_mla = query_rope_mla if mla_param["t_flag"] else query_rope_mla.reshape(B, S1, N1, Dr)
        query_rope_mla = query_rope_mla.to(torch.bfloat16).to(torch.float32)

        # matmul4 : token_x(B*S1,He) * w_kv_kr(He,Hckv+Dr) -> matmul4_res(B*S1,Hckv+Dr)
        w_kv_kr = weight_dkv_kr.to(torch.float32)
        matmul4_res = torch.matmul(token_x, w_kv_kr).to(torch.float32)

        # splitD2 : matmul4_res(B*S1,Hckv+Dr) -> splitd2_res1(B*S1,Hckv) & splitd2_res2(B*S1,Dr)
        splitd2_res1 = matmul4_res[:, :Hckv]  # 取前 Hckv 维度
        splitd2_res2 = matmul4_res[:, Hckv:]  # 取剩余的 Dr 维度

        # rmsnorm2 : splitd2_res1(B*S1,Hckv) * gamma_ckv(Hckv) -> norm2_res(B*S1,Hckv)
        ep2 = float(rmsnorm_epsilon_ckv)
        gamma2 = rmsnorm_gamma_ckv
        norm2_res = splitd2_res1 / torch.sqrt(torch.mean(splitd2_res1 ** 2, dim=-1, keepdim=True) + ep2)
        norm2_res *= gamma2

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

        return query_mla, query_rope_mla, kv_cache_out_mla, kr_cache_out_mla

    def mla_prolog_npu(self, token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv,
            cache_mode):

        return torch_npu.npu_mla_prolog(
            token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, rmsnorm_epsilon_cq=rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv, cache_mode=cache_mode)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_op_exec(self):
        B = 8
        He = 7168
        Hcq = 1536
        Hckv = 512
        N = 32
        D = 128
        Dr = 64
        Skv = 1024
        S = 1
        Nkv = 1
        BlockSize = 128
        BlockNum = math.ceil(B * Skv / BlockSize)
        T = 8
        token_x = torch.rand(B, S, He, dtype=torch.bfloat16).npu()
        w_dq = torch.rand(He, Hcq, dtype=torch.bfloat16).npu()
        w_dq_cast = torch_npu.npu_format_cast(w_dq.contiguous(), 29)
        w_uq_qr = torch.rand(Hcq, N * (D + Dr), dtype=torch.bfloat16).npu()
        w_uq_qr_cast = torch_npu.npu_format_cast(w_uq_qr.contiguous(), 29)
        w_uk = torch.rand(N, D, Hckv, dtype=torch.bfloat16).npu()
        w_dkv_kr = torch.rand(He, Hckv + Dr, dtype=torch.bfloat16).npu()
        w_dkv_kr_cast = torch_npu.npu_format_cast(w_dkv_kr.contiguous(), 29)
        rmsnorm_gamma_cq = torch.rand(Hcq, dtype=torch.bfloat16).npu()
        rmsnorm_gamma_ckv = torch.rand(Hckv, dtype=torch.bfloat16).npu()
        rope_sin = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
        rope_cos = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
        cache_index = torch.rand(B, S).to(torch.int64).npu()
        kv_cache = torch.rand(1, BlockNum * BlockSize * Nkv * Hckv, dtype=torch.bfloat16).npu()
        kv_cache = kv_cache.view(BlockNum, BlockSize, Nkv, Hckv)
        kr_cache = torch.rand(1, BlockNum * BlockSize * Nkv * Dr, dtype=torch.bfloat16).npu()
        kr_cache = kr_cache.view(BlockNum, BlockSize, Nkv, Dr)
        rmsnorm_epsilon_cq = 1.0e-5
        rmsnorm_epsilon_ckv = 1.0e-5
        cache_mode = "PA_BSND"

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
            'T': T
        }

        query_mla, query_rope_mla, kv_cache_out_mla, kr_cache_out_mla = self.mla_prolog_npu(token_x, w_dq_cast, w_uq_qr_cast, w_uk, w_dkv_kr_cast, rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv,
            cache_mode)
        print("mla_prolog output", query_mla, query_mla.shape)

        query, query_rope, kv_cache_out, kr_cache_out = self.baseline(token_x, w_dq_cast, w_uq_qr_cast, w_uk, w_dkv_kr_cast, rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv,
            cache_mode, mla_param)
        print("baseline output", query, query.shape)

        self.assertRtolEqual(query_mla.to(torch.float32), query.to(torch.float32), prec=0.005, prec16=0.005)
        self.assertRtolEqual(query_rope_mla.to(torch.float32), query_rope.to(torch.float32), prec=0.005, prec16=0.005)
        self.assertRtolEqual(kv_cache_out_mla.to(torch.float32), kv_cache_out.to(torch.float32), prec=0.005, prec16=0.005)
        self.assertRtolEqual(kr_cache_out_mla.to(torch.float32), kr_cache_out.to(torch.float32), prec=0.005, prec16=0.005)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_op_exec_tnd(self):
        B = 8
        He = 7168
        Hcq = 1536
        Hckv = 512
        N = 32
        D = 128
        Dr = 64
        Skv = 1024
        S = 1
        Nkv = 1
        BlockSize = 128
        BlockNum = math.ceil(B * Skv / BlockSize)
        T = 8
        token_x = torch.rand(T, He, dtype=torch.bfloat16).npu()
        w_dq = torch.rand(He, Hcq, dtype=torch.bfloat16).npu()
        w_dq_cast = torch_npu.npu_format_cast(w_dq.contiguous(), 29)
        w_uq_qr = torch.rand(Hcq, N * (D + Dr), dtype=torch.bfloat16).npu()
        w_uq_qr_cast = torch_npu.npu_format_cast(w_uq_qr.contiguous(), 29)
        w_uk = torch.rand(N, D, Hckv, dtype=torch.bfloat16).npu()
        w_dkv_kr = torch.rand(He, Hckv + Dr, dtype=torch.bfloat16).npu()
        w_dkv_kr_cast = torch_npu.npu_format_cast(w_dkv_kr.contiguous(), 29)
        rmsnorm_gamma_cq = torch.rand(Hcq, dtype=torch.bfloat16).npu()
        rmsnorm_gamma_ckv = torch.rand(Hckv, dtype=torch.bfloat16).npu()
        rope_sin = torch.rand(T, Dr, dtype=torch.bfloat16).npu()
        rope_cos = torch.rand(T, Dr, dtype=torch.bfloat16).npu()
        cache_index = torch.rand(T).to(torch.int64).npu()
        kv_cache = torch.rand(1, BlockNum * BlockSize * Nkv * Hckv, dtype=torch.bfloat16).npu()
        kv_cache = kv_cache.view(BlockNum, BlockSize, Nkv, Hckv)
        kr_cache = torch.rand(1, BlockNum * BlockSize * Nkv * Dr, dtype=torch.bfloat16).npu()
        kr_cache = kr_cache.view(BlockNum, BlockSize, Nkv, Dr)
        rmsnorm_epsilon_cq = 1.0e-5
        rmsnorm_epsilon_ckv = 1.0e-5
        cache_mode = "PA_BSND"

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
            't_flag': True,
            'T': T
        }

        query_mla, query_rope_mla, kv_cache_out_mla, kr_cache_out_mla = self.mla_prolog_npu(token_x, w_dq_cast, w_uq_qr_cast, w_uk, w_dkv_kr_cast, rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv,
            cache_mode)
        print("mla_prolog output", query_mla, query_mla.shape)

        query, query_rope, kv_cache_out, kr_cache_out = self.baseline(token_x, w_dq_cast, w_uq_qr_cast, w_uk, w_dkv_kr_cast, rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv,
            cache_mode, mla_param)
        print("baseline output", query, query.shape)

        self.assertRtolEqual(query_mla.to(torch.float32), query.to(torch.float32), prec=0.005, prec16=0.005)
        self.assertRtolEqual(query_rope_mla.to(torch.float32), query_rope.to(torch.float32), prec=0.005, prec16=0.005)
        self.assertRtolEqual(kv_cache_out_mla.to(torch.float32), kv_cache_out.to(torch.float32), prec=0.005, prec16=0.005)
        self.assertRtolEqual(kr_cache_out_mla.to(torch.float32), kr_cache_out.to(torch.float32), prec=0.005, prec16=0.005)

if __name__ == "__main__":
    run_tests()

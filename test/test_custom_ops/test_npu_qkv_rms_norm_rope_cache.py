import random
import copy
import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import get_npu_device, SupportedDevices
from einops import rearrange

class TestNPUQkvRmsNormRopeCache(TestCase):

    def generate_inputs(self,
                        batch_size,
                        seq_len,
                        Nq,
                        Nk,
                        Nv,
                        dim,
                        block_num,
                        block_size,
                        quant_mode,
                        cache_mode,
                        output_mode,
                        input_dtype):
        # generate inputs
        Nqkv = Nq + Nk + Nv
        qkv = torch.randn(batch_size * seq_len, Nqkv * dim, dtype=input_dtype) 
        q_gamma = torch.randn(dim, dtype=input_dtype)
        k_gamma = torch.randn(dim, dtype=input_dtype)
        cos = torch.randn(batch_size * seq_len, dim, dtype=input_dtype)
        sin = torch.randn(batch_size * seq_len, dim, dtype=input_dtype)
        
        if cache_mode == "PA_NZ":
            q_out = torch.ones(batch_size * seq_len, Nq * dim, dtype=input_dtype) * 9
            k_cache = torch.ones(block_num, Nk * dim // 32, block_size, 32, dtype=input_dtype) * 9
            v_cache = torch.ones(block_num, Nv * dim // 32, block_size, 32, dtype=input_dtype) * 9
            index_shape = ((batch_size * seq_len),)
            data = list(range(-1, block_num * block_size))
            if batch_size * seq_len > block_num * block_size:
                sampled_data = random.sample(data, block_num * block_size)
                for _ in range(batch_size * seq_len - block_num * block_size):
                    sampled_data.append(-1)
            else:
                sampled_data = random.sample(data, batch_size * seq_len)
            index = torch.Tensor(sampled_data)
            index = index.to(torch.int64)
        if quant_mode == 1:
            # q_out = q_out.to(torch.int8)
            k_cache = k_cache.to(torch.int8)
            v_cache = v_cache.to(torch.int8)
            k_scale = torch.randn(Nk, dim, dtype=torch.float32)
            v_scale = torch.randn(Nv, dim, dtype=torch.float32)
        else:
            k_scale = None
            v_scale = None

        # call golden
        qkv = 200 * qkv - 100
        q_gamma = 200 * q_gamma - 100
        k_gamma = 200 * k_gamma - 100
        sin = 2 * sin - 1
        cos = 2 * cos - 1
        qkv_size = [batch_size, seq_len, Nqkv, dim]
        num_heads = [Nq, Nk, Nv]

        return qkv, q_gamma, k_gamma, cos, sin, index, q_out, k_cache, v_cache, qkv_size, num_heads, k_scale, v_scale, cache_mode, output_mode, input_dtype

    def supported_op_exec(self,
                          qkv,
                          q_gamma,
                          k_gamma,
                          cos,
                          sin,
                          index,
                          q_out,
                          k_cache,
                          v_cache,
                          qkv_size,
                          num_heads,
                          k_scale=None,
                          v_scale=None,
                          k_offset=None,
                          v_offset=None,
                          epsilon=1e-06,
                          cache_mode="PA_NZ",
                          is_output_qkv=False):

        # golden function
        def RmsNorm(process, eps, gamma):
            rms_res = process / torch.sqrt(torch.mean(process ** 2, dim=-1, keepdim=True) + eps)
            rms_res = rms_res * gamma
            return rms_res

        def Rope(rms_res, cos, sin, rope_range):
            B = rms_res.shape[0]
            S = rms_res.shape[1]
            Nqkv = rms_res.shape[2]

            rope_dim = rope_range[1] - rope_range[0]

            Srope = cos.shape[1]
            Nrope = cos.shape[2]

            if(Srope == 1):
                cos = cos.repeat(1, S, 1, 1)
                sin = sin.repeat(1, S, 1, 1)
            if(Nrope == 1):
                cos = cos.repeat(1, 1, Nqkv, 1)
                sin = sin.repeat(1, 1, Nqkv, 1)
            rope_in = rms_res[..., rope_range[0]:rope_range[1]]
            rope_tmp1 = rope_in[..., : rope_in.shape[-1] // 2]
            rope_tmp2 = rope_in[..., rope_in.shape[-1] // 2:]

            rotate_half = torch.cat((-rope_tmp2, rope_tmp1), dim=-1)
            rope_embed = (rope_in * cos) + (rotate_half * sin)

            out = torch.cat([rms_res[..., :rope_range[0]], rope_embed, rms_res[..., rope_range[1]:]], dim=-1)
            return out

        def Quant(out, scale, offset):
            if scale is not None:
                out = out / scale
            if offset is not None:
                out = out + offset
            if scale is not None:
                out = torch.round(out).clamp(-128, 127)
            return out

        def Scatter(k_embed, k_cache, cache_mode, index):
            if cache_mode == "PA_NZ":
                Bqkv, Sqkv, N, Dqkv = k_embed.shape
                k_cache_shape = k_cache.shape
                bn = k_cache_shape[0]
                block_size = k_cache_shape[2]

                dk0 = k_cache_shape[-1]
                dk1 = Dqkv // dk0
                num_head = N
                k_cache = k_cache.reshape(bn, num_head, dk1, block_size, dk0)
                k_embed = rearrange(k_embed, 'b s n d -> (b s) n d')

                for batch in range(len(index)):
                    index_value = index[batch]
                    if index_value < 0:
                        continue
                    bn_id = index_value // block_size
                    block_offset = index_value % block_size
                    for i in range(dk1):
                        k_cache[bn_id, :, i, block_offset, :] = k_embed[batch, :, i*dk0:(i+1)*dk0].to(k_cache.dtype)
                k_cache = k_cache.reshape(k_cache_shape)

            return k_cache

        batch_size, seq_len, Nqkv, dim = qkv_size
        Nq, Nk, Nv = num_heads
        if "PA" in cache_mode:
            block_num, block_size, _, _ = k_cache.shape

        # calc rmsnorm and rope
        qkv = qkv.to(torch.float32)
        q_gamma = q_gamma.to(torch.float32)
        k_gamma = k_gamma.to(torch.float32)
        sin = sin.to(torch.float32)
        cos = cos.to(torch.float32)
        q, k, v = qkv.split([Nq * dim, Nk * dim, Nv * dim], dim=-1)
        q_4 = q.view(batch_size, seq_len, Nq, dim)
        k_4 = k.view(batch_size, seq_len, Nk, dim)
        v_4 = v.view(batch_size, seq_len, Nv, dim)

        Srope = cos.shape[0] // batch_size
        Nrope = cos.shape[1] // dim
        cos = cos.view(batch_size, Srope, Nrope, dim)    
        sin = sin.view(batch_size, Srope, Nrope, dim) 

        # q计算
        rope_range = [0, dim]
        q_4 = RmsNorm(q_4, epsilon, q_gamma)
        q_4 = Rope(q_4, cos, sin, rope_range)
        q_out_res = rearrange(q_4, 'b s n d -> (b s)  (n d)')
        q_proto_out = q_out_res.clone()

        # k计算
        k_4 = RmsNorm(k_4, epsilon, k_gamma)
        k_4 = Rope(k_4, cos, sin, rope_range)
        k_out_before_quant = k_4.clone()
        k_out_before_quant = rearrange(k_out_before_quant, 'b s n d -> (b s)  (n d)')
        k_4 = Quant(k_4, k_scale, k_offset)
        k_cache_res = Scatter(k_4, k_cache, cache_mode, index)

        # v计算
        v_out_before_quant = v_4.clone()
        v_out_before_quant = rearrange(v_out_before_quant, 'b s n d -> (b s)  (n d)')
        v_4 = Quant(v_4, v_scale, v_offset)
        v_cache_res = Scatter(v_4, v_cache, cache_mode, index)

        # 输出
        q_out_res = q_out_res.to(torch.float16)
        k_cache_res = k_cache_res.to(torch.int8)
        v_cache_res = v_cache_res.to(torch.int8)
        q_proto_out = q_proto_out.to(torch.float16)
        k_out_before_quant = k_out_before_quant.to(torch.float16)
        v_out_before_quant = v_out_before_quant.to(torch.float16)

        if is_output_qkv:
            return q_out_res, k_cache_res, v_cache_res, q_proto_out, k_out_before_quant, v_out_before_quant
        return q_out_res, k_cache_res, v_cache_res, None, None, None
        

    def custom_op_exec(self, qkv, q_gamma, k_gamma, cos, sin, index, q_out, k_cache, v_cache,
                       qkv_size=None, num_heads=None,
                       k_scale=None, v_scale=None, k_offset=None, v_offset=None, 
                       epsilon=1e-06, cache_mode="PA_NZ", is_output_qkv=False):
        qkv = qkv.npu()
        q_gamma = q_gamma.npu()
        k_gamma = k_gamma.npu()
        cos = cos.npu()
        sin = sin.npu()
        index = index.npu()
        q_out = q_out.npu()
        k_cache = k_cache.npu()
        v_cache = v_cache.npu()
        if k_scale is not None:
            k_scale = k_scale.npu()
        if v_scale is not None:
            v_scale = v_scale.npu()
        q_out_before_quant_npu, k_out_before_quant_npu, v_out_before_quant_npu = torch_npu.npu_qkv_rms_norm_rope_cache(qkv, q_gamma, k_gamma,
                                                                                                    cos, sin, index, 
                                                                                                    q_out, k_cache, v_cache,
                                                                                                    qkv_size = qkv_size,
                                                                                                    head_nums = num_heads,
                                                                                                    k_scale=k_scale,
                                                                                                    v_scale=v_scale,
                                                                                                    k_offset=k_offset,
                                                                                                    v_offset=v_offset,
                                                                                                    epsilon=epsilon,
                                                                                                    cache_mode=cache_mode,
                                                                                                    is_output_qkv=is_output_qkv)
        q_out_cpu = q_out.cpu()
        k_cache_cpu = k_cache.cpu()
        v_cache_cpu = v_cache.cpu()
        q_out_before_quant_cpu = q_out_before_quant_npu.cpu()
        k_out_before_quant_cpu = k_out_before_quant_npu.cpu()
        v_out_before_quant_cpu = v_out_before_quant_npu.cpu()
        torch._dynamo.reset()
        return q_out_cpu, k_cache_cpu, v_cache_cpu, q_out_before_quant_cpu, k_out_before_quant_cpu, v_out_before_quant_cpu

    @unittest.skip("Skip test_npu_qkv_rmsnorm_rope_cache_PA_NZ now")
    @SupportedDevices(['Ascend910B'])
    def test_npu_qkv_rmsnorm_rope_cache_PA_NZ(self, device="npu"):
        qkv, q_gamma, k_gamma, cos, sin, index, q_out, k_cache, v_cache, qkv_size, num_heads, k_scale, v_scale, cache_mode, is_output_qkv, input_dtype = self.generate_inputs(
            72, 2, 16, 1, 1, 128, 11898, 128, 1, "PA_NZ", True, torch.float16)
        golden_out = self.supported_op_exec(qkv, q_gamma, k_gamma, cos, sin, index, q_out, k_cache, v_cache,
                                            qkv_size=qkv_size,
                                            num_heads=num_heads,
                                            k_scale=k_scale, v_scale=v_scale,
                                            k_offset=None, v_offset=None, epsilon=1e-06, cache_mode=cache_mode,   is_output_qkv=is_output_qkv)
        # call npu api
        npu_out = self.custom_op_exec(qkv, q_gamma, k_gamma, cos, sin, index, q_out, k_cache, v_cache, qkv_size=qkv_size, num_heads=num_heads,
                                    k_scale=k_scale, v_scale=v_scale,
                                    k_offset=None, v_offset=None, 
                                    epsilon=1e-06, cache_mode=cache_mode, is_output_qkv=is_output_qkv)

        q_out_cpu, k_cache_cpu, v_cache_cpu, q_out_before_quant_cpu, k_out_before_quant_cpu, v_out_before_quant_cpu = npu_out
        q_out_res, k_cache_res, v_cache_res, q_out_before_quant, k_out_before_quant, v_out_before_quant = golden_out

        # comparison
        if input_dtype == torch.float16:
            atol = 0.001
            rtol = 0.001
        elif input_dtype == torch.bfloat16:
            atol = 0.004
            rtol = 0.004
        else:
            atol = 1e-5
            rtol = 1e-5

        self.assertRtolEqual(q_out_cpu, q_out_res, prec=rtol)
        self.assertRtolEqual(k_cache_cpu, k_cache_res, prec=rtol)
        self.assertRtolEqual(v_cache_cpu, v_cache_res, prec=rtol)
    
        if is_output_qkv:
            self.assertEqual(q_out_before_quant_cpu.shape, q_out_before_quant.shape)
            self.assertEqual(k_out_before_quant_cpu.shape, k_out_before_quant.shape)
            self.assertEqual(v_out_before_quant_cpu.shape, v_out_before_quant.shape)
            self.assertEqual(q_out_before_quant_cpu.dtype, q_out_before_quant.dtype)
            self.assertEqual(k_out_before_quant_cpu.dtype, k_out_before_quant.dtype)
            self.assertEqual(v_out_before_quant_cpu.dtype, v_out_before_quant.dtype)

            try:
                self.assertRtolEqual(q_out_before_quant_cpu, q_out_before_quant, prec=rtol)
            except AssertionError:
                max_diff = torch.max(torch.abs(q_out_before_quant_cpu - q_out_before_quant))
                self.assertLessEqual(
                    max_diff, 1, f"Output mismatch. Max diff: {max_diff}")

            try:
                self.assertRtolEqual(k_out_before_quant_cpu, k_out_before_quant, prec=rtol)
            except AssertionError:
                max_diff = torch.max(torch.abs(k_out_before_quant_cpu - k_out_before_quant))
                self.assertLessEqual(
                    max_diff, 1, f"Output mismatch. Max diff: {max_diff}")

            try:
                self.assertRtolEqual(v_out_before_quant_cpu, v_out_before_quant, prec=rtol)
            except AssertionError:
                max_diff = torch.max(torch.abs(v_out_before_quant_cpu - v_out_before_quant))
                self.assertLessEqual(
                    max_diff, 1, f"Output mismatch. Max diff: {max_diff}")

if __name__ == "__main__":
    run_tests()

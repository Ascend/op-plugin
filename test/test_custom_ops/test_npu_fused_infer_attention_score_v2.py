import math
import unittest
import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices
from einops import rearrange


class TestFusedInferAttentionV2(TestCase):
    def softmax(self, x):
        x = x.cpu().numpy().astype(np.float32)
        x_max = x.max(axis=-1, keepdims=True)
        x_sub = x - x_max
        y = np.exp(x_sub)
        x_sum = y.sum(axis=-1, keepdims=True)
        ans = y
        return ans, x_sum, x_max

    def npSoftmax_new(self, x, sinks=None):
        if sinks is not None:
            sinks = sinks.view(1, -1, 1, 1)
            sinks = sinks.broadcast_to(1, sinks.shape[1], x.shape[2], 1)
            x = torch.cat([x, sinks], dim=-1)
        x_max = torch.max(x, dim=-1, keepdims=True)[0]
        x_sub = x.sub(x_max)
        y = torch.exp(x_sub)
        del x
        del x_sub
        x_max = x_max.cpu()
        x_sum = y.sum(dim=-1, keepdims=True)
        ans = y.div(x_sum)
        if sinks is not None:
            ans = ans[..., :-1]
        return ans, x_max, x_sum

    def supported_op_exec(self, query_states1, past_key, past_value, head_dim, B, N, S, return_softmax_lse):
        attn_weights1 = torch.matmul(query_states1, past_key.transpose(2, 3)) / 0.0078125
        if return_softmax_lse:
            softmax_res, softmax_sum, softmax_max = self.softmax(attn_weights1)
            lse = np.log(softmax_sum) + softmax_max
        else:
            lse = np.zeros([B, N, S, 1], np.float32)
            attn_weights1 = torch.max(attn_weights1, torch.full(
                (1, 1), torch.finfo(attn_weights1.dtype).min, device=attn_weights1.device))
            attn_weights1 = torch.nn.functional.softmax(attn_weights1, dim=-1, dtype=torch.float32).to(query_states1.dtype)

        attn_output1 = torch.matmul(attn_weights1, past_value)
        return attn_output1, lse
    
    def supported_op_exec_ntd(self, query_states1, past_key, past_value, head_dim, T, N):
        attn_weights1 = torch.matmul(query_states1, past_key.transpose(1, 2)) / 0.0078125

        attn_weights1 = torch.nn.functional.softmax(attn_weights1, dim=-1, dtype=torch.float32).to(query_states1.dtype)

        attn_output1 = torch.matmul(attn_weights1, past_value)
        attn_output1 = attn_output1.transpose(0, 1)
        return attn_output1

    def supported_op_exec_sink(self, query_states1, past_key, past_value, learnable_sink):
        q_tensor = rearrange(query_states1, 's n d -> 1 n s d').to(torch.float64)
        k_tensor = rearrange(past_key, 's n d -> 1 n s d').to(torch.float64)
        v_tensor = rearrange(past_value, 's n d -> 1 n s d').to(torch.float64)
    
        qkEleRes = torch.matmul(q_tensor, k_tensor.transpose(3, 2)) / 0.0078125
        softmax_res, x_max, x_sum = self.npSoftmax_new(qkEleRes, learnable_sink)
        y = torch.matmul(softmax_res, v_tensor)
        return rearrange(y, '1 n s d -> s n d')

    def custom_op_exec(self, query, key, value, head_dim, return_softmax_lse):
        softmax_scale = 1 / 0.0078125
        return torch_npu.npu_fused_infer_attention_score_v2(
            query, key, value, num_query_heads=32, input_layout="BNSD", softmax_scale=softmax_scale,
            pre_tokens=65535, next_tokens=65535, return_softmax_lse=return_softmax_lse)

    def custom_op_exec_tnd(self, query, key, value, head_dim, return_softmax_lse):
        softmax_scale = 1 / 0.0078125
        return torch_npu.npu_fused_infer_attention_score_v2(
            query, key, value, num_query_heads=32, input_layout="TND", softmax_scale=softmax_scale,
            pre_tokens=65535, next_tokens=65535, return_softmax_lse=return_softmax_lse)
    
    def custom_op_exec_tnd_pa(self, query, key, value, return_softmax_lse, block_table):
        softmax_scale = 1 / 0.0078125
        return torch_npu.npu_fused_infer_attention_score_v2(
            query, key, value, num_query_heads=1, input_layout="TND", softmax_scale=softmax_scale,
            pre_tokens=65535, next_tokens=65535, return_softmax_lse=return_softmax_lse, block_table=block_table)

    def custom_op_exec_ntd_tnd(self, query, key, value, head_dim, actseqlen, actseqlenkv, return_softmax_lse):
        softmax_scale = 1 / 0.0078125
        return torch_npu.npu_fused_infer_attention_score_v2(
            query, key, value, num_query_heads=2, input_layout="NTD_TND", softmax_scale=softmax_scale,
            pre_tokens=65535, next_tokens=65535, actual_seq_qlen=actseqlen,
            actual_seq_kvlen=actseqlenkv, return_softmax_lse=return_softmax_lse)

    def custom_op_exec_sinks(self, query, key, value, head_dim, actseqlen, actseqlenkv, return_softmax_lse, learnable_sink):
        softmax_scale = 1 / 0.0078125
        return torch_npu.npu_fused_infer_attention_score_v2(
            query, key, value, learnable_sink=learnable_sink, num_query_heads=8, input_layout="TND",
            softmax_scale=softmax_scale, pre_tokens=65535, next_tokens=65535, actual_seq_qlen=actseqlen,
            actual_seq_kvlen=actseqlenkv, return_softmax_lse=return_softmax_lse)
    
    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_fused_infer_attention_score_v2(self, device="npu"):
        query = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()
        key = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()
        value = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()

        head_dim = 128
        return_softmax_lse = False

        supported_output, lse_out = self.supported_op_exec(query, key, value, head_dim, 1, 32, 2048, return_softmax_lse)
        custom_output = self.custom_op_exec(query, key, value, head_dim, False)
        attention_output = custom_output[0]

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_fused_infer_attention_score_v2_pfa_return_lse(self, device="npu"):
        query = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()
        key = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()
        value = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()

        head_dim = 128
        return_softmax_lse = True

        supported_output, lse_out = self.supported_op_exec(query, key, value, head_dim, 1, 32, 2048, return_softmax_lse)
        custom_output = self.custom_op_exec(query, key, value, head_dim, True)
        attention_output = custom_output[0]
        softmaxlse_output = custom_output[1]

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_fused_infer_attention_score_v2_ifa_return_lse(self, device="npu"):
        query = torch.randn(1, 32, 1, 128, dtype=torch.float16).npu()
        key = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()
        value = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()

        head_dim = 128
        return_softmax_lse = True

        supported_output, lse_out = self.supported_op_exec(query, key, value, head_dim, 1, 32, 1, return_softmax_lse)
        custom_output = self.custom_op_exec(query, key, value, head_dim, True)
        attention_output = custom_output[0]
        softmaxlse_output = custom_output[1]

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_fused_infer_attention_score_v2_ntd_tnd(self, device="npu"):
        query = torch.full((2, 32, 192), 1, dtype=torch.bfloat16).npu()
        key = torch.full((2, 32, 192), 1, dtype=torch.bfloat16).npu()
        value = torch.full((2, 32, 128), 1, dtype=torch.bfloat16).npu()

        head_dim = 128
        return_softmax_lse = False

        actseqlen = [32]
        actseqlenkv = [32]

        golden_output = self.supported_op_exec_ntd(query, key, value, head_dim, 32, 2)
        custom_output = self.custom_op_exec_ntd_tnd(query, key, value, head_dim, actseqlen, actseqlenkv, return_softmax_lse)
        attention_out = custom_output[0]
        self.assertRtolEqual(golden_output, attention_out, prec=0.000001, prec16=0.000001)
    
    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_fused_infer_attention_score_v2(self, device="npu"):
        query = torch.full((128, 1, 128), 1, dtype=torch.bfloat16).npu()
        key = torch.full((128, 1, 128), 1, dtype=torch.bfloat16).npu()
        value = torch.full((128, 1, 128), 1, dtype=torch.bfloat16).npu()
        block_table = torch.randint(0, 10, (1, 1), dtype=torch.int32).npu()

        head_dim = 128
        return_softmax_lse = True

        supported_output = self.supported_op_exec(query, key, value, head_dim, 1, 1, 128, return_softmax_lse)
        key_cache = key.reshape(1, 1, 128, 8, 16).transpose(0, 1, 3, 2, 4)
        value_cache = value.reshape(1, 1, 128, 8, 16).transpose(0, 1, 3, 2, 4)

        custom_output = self.custom_op_exec_tnd_pa(query, key_cache, value_cache, return_softmax_lse, block_table)
        golden_output = supported_output[0]
        attention_output = custom_output[0]
        self.assertRtolEqual(golden_output, attention_out, prec=0.000001, prec16=0.000001)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_fused_infer_attention_score_v2_sink(self, device="npu"):
        query = torch.full((128, 8, 128), 1, dtype=torch.bfloat16).npu()
        key = torch.full((128, 8, 128), 1, dtype=torch.bfloat16).npu()
        value = torch.full((128, 8, 128), 1, dtype=torch.bfloat16).npu()
        learnable_sink = torch.full((8,), 1, dtype=torch.bfloat16).npu()

        head_dim = 128
        return_softmax_lse = True

        actseqlen = [128]
        actseqlenkv = [128]

        supported_output = self.supported_op_exec_sink(query, key, value, learnable_sink).to(torch.float64)
        custom_output = self.custom_op_exec_sinks(query, key, value, head_dim, actseqlen, actseqlenkv, return_softmax_lse, learnable_sink)

        attention_output = custom_output[0].to(torch.float64)
        self.assertRtolEqual(supported_output, attention_output, prec=0.000001, prec16=0.000001)

    @SupportedDevices(['Ascend910_95', 'Ascend950'])
    def test_npu_fused_infer_attention_score_v2_d_unequal(self, device="npu"):
        query = torch.ones(1, 2, 1024, 192, dtype=torch.float16).npu()
        key = torch.ones((1, 2, 1024, 128), dtype=torch.float16).npu()
        value = torch.ones((1, 2, 1024, 128), dtype=torch.float16).npu()

        return_softmax_lse = False
        custom_output = torch_npu.npu_fused_infer_attention_v2(
            query, key, value, num_query_heads=2, input_layout="BNSD", 
            pre_tokens=65535, next_tokens=65535, return_softmax_lse=return_softmax_lse)
        
        golden_output = torch.ones(1, 2, 1024, 128, dtype=torch.float16).npu()
        res = custom_output[0].equal(golden_output)
        self.assertRtolEqual(res, True)

    @SupportedDevices(['Ascend910_95', 'Ascend950'])
    def test_npu_fused_infer_attention_score_v2_page_attention(self, device="npu"):
        query = torch.ones(2, 1, 2048, 128, dtype=torch.float16).npu()
        key = torch.ones((32, 1, 128, 128), dtype=torch.float16).npu()
        value = torch.ones((32, 1, 128, 128), dtype=torch.float16).npu()
        block_table = torch.ones((2, 32), dtype=torch.int32).npu()
        actseqlen = [2048]
        actseqlenkv = [2048]

        return_softmax_lse = False
        custom_output = torch_npu.npu_fused_infer_attention_v2(
            query, key, value, num_query_heads=1, input_layout="BNSD", block_table=block_table, actual_seq_qlen=actseqlen, actual_seq_kvlen=actseqlenkv, 
            block_size=128, pre_tokens=65535, next_tokens=65535, return_softmax_lse=return_softmax_lse)
        
        golden_output = torch.ones(2, 1, 2048, 128, dtype=torch.float16).npu()
        res = custom_output[0].equal(golden_output)
        self.assertRtolEqual(res, True)

    @SupportedDevices(['Ascend910_95', 'Ascend950'])
    def test_npu_fused_infer_attention_score_v2_int32_antiquant(self, device="npu"):
        query = torch.ones(2, 1, 1, 128, dtype=torch.float16).npu()
        key = torch.ones((2, 1, 256, 16), dtype=torch.int32).npu()
        value = torch.full((2, 1, 256, 16), 286331153, dtype=torch.int32).npu()
        key_antiquant_scale = torch.ones((1, 128), dtype=torch.float16).npu()
        value_antiquant_scale = torch.ones((1, 128), dtype=torch.float16).npu()
        antiquant_mode = 0

        return_softmax_lse = False
        custom_output = torch_npu.npu_fused_infer_attention_v2(
            query, key, value, num_query_heads=1, input_layout="BNSD", dequant_scale_key=key_antiquant_scale, dequant_scale_value=value_antiquant_scale, 
            key_quant_mode=antiquant_mode, value_quant_mode=antiquant_mode, pre_tokens=65535, next_tokens=65535, return_softmax_lse=return_softmax_lse)
        
        golden_output = torch.ones(2, 1, 1, 128, dtype=torch.float16).npu()
        res = custom_output[0].equal(golden_output)
        self.assertRtolEqual(res, True)

    @SupportedDevices(['Ascend910_95', 'Ascend950'])
    def test_npu_fused_infer_attention_score_v2_fp4_antiquant(self, device="npu"):
        query = torch.ones(2, 1, 1, 128, dtype=torch.float16).npu()
        key = torch.ones((2, 1, 256, 64), dtype=torch.uint8).npu()
        value = torch.full((2, 1, 256, 64), 68, dtype=torch.uint8).npu()
        key_antiquant_scale = torch.ones((1, 2, 1, 256, 4), dtype=torch.uint8).npu()
        value_antiquant_scale = torch.ones((1, 2, 1, 256, 4), dtype=torch.uint8).npu()
        antiquant_mode = 6
        key_dtype = torch_npu.float4_e1m2
        value_dtype = torch_npu.float4_e1m2
        dequant_scale_key_dtype = torch_npu.float8_e8m0
        dequant_scale_value_dtype = torch_npu.float8_e8m0

        return_softmax_lse = False
        custom_output = torch_npu.npu_fused_infer_attention_v2(
            query, key, value, num_query_heads=1, input_layout="BNSD", dequant_scale_key=key_antiquant_scale, dequant_scale_value=value_antiquant_scale, 
            key_quant_mode=antiquant_mode, value_quant_mode=antiquant_mode, pre_tokens=65535, next_tokens=65535, return_softmax_lse=return_softmax_lse,
            key_dtype=key_dtype, value_dtype=value_dtype, dequant_scale_key_dtype=dequant_scale_key_dtype, dequant_scale_value_dtype=dequant_scale_value_dtype)
        
        golden_output = torch.zeros(2, 1, 1, 128, dtype=torch.float16).npu()
        res = custom_output[0].equal(golden_output)
        self.assertRtolEqual(res, True)
    
    @SupportedDevices(['Ascend910_95', 'Ascend950'])
    def test_npu_fused_infer_attention_score_v2_without_outdtype(self, device="npu"):
        query = torch.randn(32, 2, 8, 192, dtype=torch.float16).npu()
        key = torch.randn((32, 2048, 8, 192), dtype=torch.float16).npu()
        value = torch.randn((32, 2048, 8, 128), dtype=torch.float16).npu()
        quant_scale_out = torch.randn((1), dtype=torch.float32).npu()
        return_softmax_lse = False
        softmax_scale = 1 / 0.78127
        custom_output = torch_npu.npu_fused_infer_attention_score_v2(
            query, key, value, num_query_heads=8, input_layout="BSND", 
            quant_scale_out=quant_scale_out,
            softmax_scale=softmax_scale, pre_tokens=65535, next_tokens=65535, return_softmax_lse=return_softmax_lse)
        golden_output = torch.randn((32, 2, 8, 128), dtype=torch.int8).npu()
        res = custom_output[0].equal(golden_output)
        self.assertRtolEqual(res, True)

    @SupportedDevices(['Ascend910_95', 'Ascend950'])
    def test_npu_fused_infer_attention_score_v2_with_outdtype(self, device="npu"):
        query = torch.randn(32, 8, 2, 192, dtype=torch.float16).npu()
        key = torch.randn((32, 8, 2048, 192), dtype=torch.float16).npu()
        value = torch.randn((32, 8, 2048, 128), dtype=torch.float16).npu()
        quant_scale_out = torch.randn((1), dtype=torch.float32).npu()
        out_dtype = torch.float8_e5m2
        return_softmax_lse = False
        softmax_scale = 1 / 0.0078125
        custom_output = torch_npu.npu_fused_infer_attention_score_v2(
            query, key, value, num_query_heads=8, input_layout="BNSD", 
            quant_scale_out=quant_scale_out, out_dtype=out_dtype,
            softmax_scale=softmax_scale, pre_tokens=65535, next_tokens=65535, return_softmax_lse=return_softmax_lse)
        golden_output = torch.randn((32, 8, 2, 128), dtype=torch.float8_e5m2).npu()
        res = custom_output[0].equal(golden_output)
        self.assertRtolEqual(res, True)

    @SupportedDevices(['Ascend910_95', 'Ascend950'])
    def test_npu_fused_infer_attention_score_v2_bnsd_bsnd_d_unequal(self, device="npu"):
        query = torch.ones(32, 8, 1, 192, dtype=torch.float16).npu()
        key = torch.ones((32, 8, 2048, 192), dtype=torch.float16).npu()
        value = torch.ones((32, 8, 2048, 128), dtype=torch.float16).npu()

        return_softmax_lse = False
        softmax_scale = 1 / 0.0078125
        custom_output = torch_npu.npu_fused_infer_attention_score_v2(
            query, key, value, num_query_heads=8, input_layout="BNSD_BSND", 
            softmax_scale=softmax_scale, pre_tokens=65535, next_tokens=65535, return_softmax_lse=return_softmax_lse)
        
        golden_output = torch.ones(32, 1, 8, 128, dtype=torch.float16).npu()
        res = custom_output[0].equal(golden_output)
        self.assertRtolEqual(res, True)

    @SupportedDevices(['Ascend910_95', 'Ascend950'])
    def test_npu_fused_infer_attention_score_v2_bsnd_d_unequal(self, device="npu"):
        query = torch.ones(32, 1, 8, 192, dtype=torch.float16).npu()
        key = torch.ones((32, 2048, 8, 192), dtype=torch.float16).npu()
        value = torch.ones((32, 2048, 8, 128), dtype=torch.float16).npu()

        return_softmax_lse = False
        softmax_scale = 1 / 0.0078125
        custom_output = torch_npu.npu_fused_infer_attention_score_v2(
            query, key, value, num_query_heads=8, input_layout="BSND", 
            softmax_scale=softmax_scale, pre_tokens=65535, next_tokens=65535, return_softmax_lse=return_softmax_lse)
        
        golden_output = torch.ones(32, 1, 8, 128, dtype=torch.float16).npu()
        res = custom_output[0].equal(golden_output)
        self.assertRtolEqual(res, True)

    @SupportedDevices(['Ascend910_95', 'Ascend950'])
    def test_npu_fused_infer_attention_score_v2_bsh_d_unequal(self, device="npu"):
        query = torch.ones(32, 1, 1536, dtype=torch.float16).npu()
        key = torch.ones((32, 2048, 1536), dtype=torch.float16).npu()
        value = torch.ones((32, 2048, 1024), dtype=torch.float16).npu()

        return_softmax_lse = False
        softmax_scale = 1 / 0.0078125
        custom_output = torch_npu.npu_fused_infer_attention_score_v2(
            query, key, value, num_query_heads=8, input_layout="BSH", 
            softmax_scale=softmax_scale, pre_tokens=65535, next_tokens=65535, return_softmax_lse=return_softmax_lse)
        
        golden_output = torch.ones(32, 1, 1024, dtype=torch.float16).npu()
        res = custom_output[0].equal(golden_output)
        self.assertRtolEqual(res, True)        

    @SupportedDevices(['Ascend910_95', 'Ascend950'])
    def test_npu_fused_infer_attention_score_v2_nsd_d_unequal(self, device="npu"):
        query = torch.ones(8, 1, 192, dtype=torch.float16).npu()
        key = torch.ones((8, 2048, 192), dtype=torch.float16).npu()
        value = torch.ones((8, 2048, 128), dtype=torch.float16).npu()

        softmax_lse_flag = False
        scale = 1 / 0.0078125
        custom_output = torch_npu.npu_fused_infer_attention_score_v2(
            query, key, value, num_query_heads=8, input_layout="NSD", 
            softmax_scale=softmax_scale, pre_tokens=65535, next_tokens=65535, return_softmax_lse=return_softmax_lse)
        
        golden_output = torch.ones(8, 1, 128, dtype=torch.float16).npu()
        res = custom_output[0].equal(golden_output)
        self.assertRtolEqual(res, True)     

    @SupportedDevices(['Ascend910_95', 'Ascend950'])
    def test_npu_fused_infer_attention_score_v2_tnd_d_unequal(self, device="npu"):
        query = torch.full((32, 8, 192), 1, dtype=torch.bfloat16).npu()
        key = torch.full((32, 8, 192), 1, dtype=torch.bfloat16).npu()
        value = torch.full((32, 8, 128), 1, dtype=torch.bfloat16).npu()

        actseqlen = [32]
        actseqlenkv = [32]

        return_softmax_lse = False
        softmax_scale = 1 / 0.0078125
        custom_output = torch_npu.npu_fused_infer_attention_score_v2(
            query, key, value, num_query_heads=8, input_layout="TND",
            softmax_scale=softmax_scale, pre_tokens=65535, next_tokens=65535,
            actual_seq_qlen=actseqlen, actual_seq_kvlen=actseqlenkv, return_softmax_lse=return_softmax_lse)
        
        golden_output = torch.ones(32, 8, 128, dtype=torch.float16).npu()
        res = custom_output[0].equal(golden_output)
        self.assertRtolEqual(res, True)

if __name__ == "__main__":
    run_tests()

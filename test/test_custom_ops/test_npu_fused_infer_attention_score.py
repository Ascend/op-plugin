import math
import unittest
import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestFusedInferAttentionScore(TestCase):

    def generate_int_tensor_with_sum(self, B, T):
        if B <= 0 or T < 0:
            raise ValueError("B 必须大于 0, T 必须是非负整数")
        if B > T:
            raise ValueError(f"无法生成 B={B} 个非负整数且总和为 T={T}, 每个数至少为 1, 所以 B <= T")
        
        partition_points = np.sort(np.random.choice(range(1, T), B - 1, replace = False))
        partition_points = np.concatenate([[0], partition_points, [T]])
        tensor = torch.tensor(np.diff(partition_points), dtype = torch.int)
        return tensor

    def softmax(self, x):
        x = x.cpu().numpy().astype(np.float32)
        x_max = x.max(axis=-1, keepdims=True)
        x_sub = x - x_max
        y = np.exp(x_sub)
        x_sum = y.sum(axis=-1, keepdims=True)
        ans = y
        return ans, x_sum, x_max

    def compute_golden_output_cpu(self, query, key_cache, value_cache, num_heads, num_key_value_heads, head_dim, block_size, block_table, seq_lens, scale, input_layout = "TND"):
        B, H, D = query.shape
        assert H == num_heads, f"Query heads {H} != num_heads {num_heads}"
        assert D == head_dim, f"Query dim {D} != head_dim {head_dim}"

        Block_num, Block_size, total_dim = value_cache.shape
        value_head_dim = total_dim // num_key_value_heads

        block_num, block_size, _ = key_cache.shape
        T_kv = block_num * block_size

        key_cache_reshaped = key_cache.view(block_num, block_size, num_key_value_heads, head_dim)
        key_cache_reshaped = key_cache_reshaped.permute(2, 0, 1, 3)
        key_cache_reshaped = key_cache_reshaped.reshape(num_key_value_heads, T_kv, head_dim)

        value_cache_reshaped = value_cache.view(block_num, block_size, num_key_value_heads, value_head_dim)
        value_cache_reshaped = value_cache_reshaped.permute(2, 0, 1, 3)
        value_cache_reshaped = value_cache_reshaped.reshape(num_key_value_heads, T_kv, value_head_dim)

        if num_key_value_heads == 1 and num_heads > 1:
            key_cache_reshaped = key_cache_reshaped.expand(num_heads, T_kv, head_dim)
            value_cache_reshaped = value_cache_reshaped.expand(num_heads, T_kv, value_head_dim)
        elif num_key_value_heads == num_heads:
            pass
        else:
            raise NotImplementedError("不支持的 num_key_value_heads != num_heads 且 != 1 的情况")
        
        query_expanded = query.unsqueeze(-2)
        key_expanded = key_cache_reshaped.unsqueeze(0).expand(B, -1, -1, -1)
        value_expanded = value_cache_reshaped.unsqueeze(0).expand(B, -1, -1, -1)

        attn_scores = torch.matmul(query_expanded, key_expanded.transpose(-1, -2)) * scale
        mask = torch.arange(T_kv, device = query.device).unsqueeze(0) < seq_lens.unsqueeze(1)
        mask = mask.unsqueeze(1).unsqueeze(1)

        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        attention_out = torch.matmul(attn_weights, value_expanded)

        return attention_out

    
    def supported_op_exec(self, query_states1, past_key, past_value, head_dim, B, N, S, softmax_lse_flag):
        attn_weights1 = torch.matmul(query_states1, past_key.transpose(2, 3)) / 0.0078125
        if (softmax_lse_flag == True):
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

    def custom_op_exec(self, query, key, value, head_dim, softmax_lse_flag):
        scale = 1 / 0.0078125
        return torch_npu.npu_fused_infer_attention_score(
            query, key, value, num_heads=32, input_layout="BNSD", scale=scale, pre_tokens=65535, next_tokens=65535, softmax_lse_flag=softmax_lse_flag)

    def custom_op_exec_tnd(self, query, key, value, head_dim, softmax_lse_flag):
        scale = 1 / 0.0078125
        return torch_npu.npu_fused_infer_attention_score(
            query, key, value, num_heads=32, input_layout="TND", scale=scale, pre_tokens=65535, next_tokens=65535, softmax_lse_flag=softmax_lse_flag)

    def custom_op_exec_ntd_tnd(self, query, key, value, head_dim, actseqlen, actseqlenkv, softmax_lse_flag):
        scale = 1 / 0.0078125
        return torch_npu.npu_fused_infer_attention_score(
            query, key, value, num_heads=2, input_layout="NTD_TND", scale=scale, pre_tokens=65535, next_tokens=65535, actual_seq_lengths=actseqlen, 
            actual_seq_lengths_kv=actseqlenkv, softmax_lse_flag=softmax_lse_flag)
    
    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_fused_infer_attention_score(self, device="npu"):
        query = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()
        key = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()
        value = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()

        head_dim = 128
        softmax_lse_flag = False

        supported_output, lse_out = self.supported_op_exec(query, key, value, head_dim, 1, 32, 2048, softmax_lse_flag)
        custom_output = self.custom_op_exec(query, key, value, head_dim, False)
        attention_output = custom_output[0]

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_fused_infer_attention_score_pfa_return_lse(self, device="npu"):
        query = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()
        key = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()
        value = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()

        head_dim = 128
        softmax_lse_flag = True

        supported_output, lse_out = self.supported_op_exec(query, key, value, head_dim, 1, 32, 2048, softmax_lse_flag)
        custom_output = self.custom_op_exec(query, key, value, head_dim, True)
        attention_output = custom_output[0]
        softmaxlse_output = custom_output[1]
    
    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_fused_infer_attention_score_ifa_return_lse(self, device="npu"):
        query = torch.randn(1, 32, 1, 128, dtype=torch.float16).npu()
        key = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()
        value = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()

        head_dim = 128
        softmax_lse_flag = True

        supported_output, lse_out = self.supported_op_exec(query, key, value, head_dim, 1, 32, 1, softmax_lse_flag)
        custom_output = self.custom_op_exec(query, key, value, head_dim, True)
        attention_output = custom_output[0]
        softmaxlse_output = custom_output[1]

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_fused_infer_attention_score_tnd(self, device="npu"):
        query = torch.randn(1, 32, 1, 128, dtype=torch.float16).npu()
        key = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()
        value = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()

        head_dim = 128
        softmax_lse_flag = True

        supported_output, lse_out = self.supported_op_exec(query, key, value, head_dim, 1, 32, 1, softmax_lse_flag)
        custom_output = self.custom_op_exec_tnd(query, key, value, head_dim, True)
        attention_output = custom_output[0]
        softmaxlse_output = custom_output[1]

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_fused_infer_attention_score_v3_param(self, device="npu"):
        query = torch.randn(1, 32, 1, 128, dtype=torch.float16).npu()
        key = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()
        value = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()
        query_rope = torch.randn(1, 32, 1, 16, dtype=torch.float16).npu()
        key_rope = torch.randn(1, 32, 2048, 16, dtype=torch.float16).npu()
        key_rope_antiquant_scale = torch.randn(1, 1, 1, 16, dtype=torch.float16).npu()
        head_dim = 128
        softmax_lse_flag = True

        supported_output, lse_out = self.supported_op_exec(query, key, value, head_dim, 1, 32, 1, softmax_lse_flag)
        scale = 1 / 0.0078125
        custom_output = torch_npu.npu_fused_infer_attention_score(
            query, key, value, query_rope=query_rope, key_rope=key_rope, key_rope_antiquant_scale=key_rope_antiquant_scale,
            num_heads=32, input_layout="BNSD", scale=scale, pre_tokens=65535, next_tokens=65535, softmax_lse_flag=softmax_lse_flag)

        attention_output = custom_output[0]
        softmaxlse_output = custom_output[1]

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_fused_infer_attention_score_tnd_ntd(self, device="npu"):
        query = torch.randn(2, 32, 512, dtype=torch.float16).npu()
        key = torch.randn(1, 1, 2048, 512, dtype=torch.float16).npu()
        value = torch.randn(1, 1, 2048, 512, dtype=torch.float16).npu()

        head_dim = 512
        softmax_lse_flag = True
        scale = 1 / 0.0078125
        custom_output = torch_npu.npu_fused_infer_attention_score(
            query, key, value, num_heads=32, input_layout="TND_NTD", scale=scale, pre_tokens=65535,
            next_tokens=65535, softmax_lse_flag=softmax_lse_flag)

        attention_output = custom_output[0]
        softmaxlse_output = custom_output[1]

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_fused_infer_attention_score_ntd_tnd(self, device="npu"):
        query = torch.full((2, 32, 192), 1, dtype=torch.bfloat16).npu()
        key = torch.full((2, 32, 192), 1, dtype=torch.bfloat16).npu()
        value = torch.full((2, 32, 128), 1, dtype=torch.bfloat16).npu()

        head_dim = 128
        softmax_lse_flag = False

        actseqlen = [32]
        actseqlenkv = [32]

        golden_output = self.supported_op_exec_ntd(query, key, value, head_dim, 32, 2)
        custom_output = self.custom_op_exec_ntd_tnd(query, key, value, head_dim, actseqlen, actseqlenkv, softmax_lse_flag)
        attention_out = custom_output[0]
        self.assertRtolEqual(golden_output, attention_out, prec=0.000001, prec16=0.000001)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_fused_infer_attention_score_v3_fullquant(self, device="npu"):
        query = torch.randint(1, 2, (1, 32, 2, 128), dtype=torch.int8).npu()
        key = torch.randint(1, 2, (1, 32, 2048, 128), dtype=torch.int8).npu()
        value = torch.randint(1, 2, (1, 32, 2048, 128), dtype=torch.int8).npu()
        dequant_scale1 = torch.ones(1).npu()
        dequant_scale2 = torch.ones(1).npu()
        quant_scale1 = torch.ones(1).npu()
        quant_scale2 = torch.ones(1).npu()

        head_dim = 128
        softmax_lse_flag = False
        scale = 1 / 0.0078125
        custom_output = torch_npu.npu_fused_infer_attention_score(
            query, key, value, dequant_scale1=dequant_scale1, dequant_scale2=dequant_scale2, quant_scale1=quant_scale1,
            quant_scale2=quant_scale2, num_heads=32, input_layout="BNSD", scale=scale, pre_tokens=65535, next_tokens=65535,
            softmax_lse_flag=softmax_lse_flag)

        query_t = torch.ones(1, 32, 2, 128).npu()
        key_t = torch.ones(1, 32, 2048, 128).npu()
        value_t = torch.ones(1, 32, 2048, 128).npu()
        
        golden_output = self.supported_op_exec(query_t, key_t, value_t, head_dim, 1, 32, 1, softmax_lse_flag)
        res = custom_output[0].equal(golden_output[0])
        self.assertRtolEqual(res, True)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_fused_infer_attention_score_v3_antiquant(self, device="npu"):
        query = torch.ones(32, 8, 1, 128, dtype=torch.float16).npu()
        key = torch.full((32, 8, 2048, 16), 286331353, dtype=torch.int32).npu()
        value = torch.full((32, 8, 2048, 16), 286331353, dtype=torch.int32).npu()
        key_antiquant_scale = torch.ones(1, dtype=torch.float16).npu()
        value_antiquant_scale = torch.ones(1, dtype=torch.float16).npu()
        key_antiquant_mode = torch.ones(1).npu()
        value_antiquant_mode = torch.ones(1).npu()

        softmax_lse_flag = False
        scale = 1 / 0.0078125
        custom_output = torch_npu.npu_fused_infer_attention_score(
            query, key, value, key_antiquant_scale=key_antiquant_scale, value_antiquant_scale=value_antiquant_scale,
            key_antiquant_mode=key_antiquant_mode, value_antiquant_mode=value_antiquant_mode, num_heads=8, input_layout="BNSD", 
            scale=scale, pre_tokens=65535, next_tokens=65535, softmax_lse_flag=softmax_lse_flag)
        
        golden_output = torch.ones(32, 8, 1, 128, dtype=torch.float16).npu()
        res = custom_output[0].equal(golden_output)
        self.assertRtolEqual(res, True)


    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_infer_attention_score_PA(self, DEVICE = "npu"):
        B = 1
        T = 69
        head_dim = 192
        num_heads = 16
        num_kv_heads = 1
        block_num = 1
        block_size = 128
        block_table = torch.randint(0, 10, [B, 32], dtype = torch.int32).npu()
        query = torch.rand([B * 1, num_heads, head_dim], dtype=torch.float16).npu()
        key_cache = torch.rand([block_num, block_size, num_kv_heads * head_dim], dtype=torch.float16).npu()
        value_cache = torch.rand([block_num, block_size, num_kv_heads * 128], dtype=torch.float16).npu()
        scale = head_dim**-0.5

        seq_lens = self.generate_int_tensor_with_sum(B, T).to(torch.int32)
        query_lens = torch.ones(B, dtype = torch.int32).npu()

        attention_output = torch_npu.npu_fused_infer_attention_score(query, key_cache, value_cache, num_heads = num_heads, num_key_value_heads = num_kv_heads, input_layout = "TND",
                            scale = scale, block_table = block_table, block_size = block_size, actual_seq_lengths = query_lens, actual_seq_lengths_kv = seq_lens)[0]
        
        query_cpu = query.detach().cpu().to(torch.float32)
        key_cache_cpu = key_cache.detach().cpu().to(torch.float32)
        value_cache_cpu = value_cache.detach().cpu().to(torch.float32)
        block_table_cpu = block_table.detach().cpu()
        seq_lens_cpu = seq_lens.detach().cpu()

        golden_output = self.compute_golden_output_cpu(query = query_cpu, key_cache = key_cache_cpu, value_cache = value_cache_cpu, num_heads = num_heads, num_key_value_heads = num_kv_heads,
            head_dim = head_dim, block_size = block_size, block_table = block_table_cpu, seq_lens = seq_lens_cpu, scale = scale)

        golden_output_reshaped = golden_output.reshape(attention_output.shape)
        golden_output_reshaped = golden_output_reshaped.to(torch.float16)
        attention_out_cpu = attention_output.detach().cpu()

        self.assertRtolEqual(golden_output_reshaped, attention_out_cpu)

    @SupportedDevices(['Ascend950'])
    def test_npu_fused_infer_attention_score_v3_d_unequal(self, device="npu"):
        query = torch.ones(32, 8, 1, 192, dtype=torch.float16).npu()
        key = torch.ones((32, 8, 2048, 128), dtype=torch.float16).npu()
        value = torch.ones((32, 8, 2048, 128), dtype=torch.float16).npu()

        softmax_lse_flag = False
        scale = 1 / 0.0078125
        custom_output = torch_npu.npu_fused_infer_attention_score(
            query, key, value, num_heads=8, input_layout="BNSD", 
            scale=scale, pre_tokens=65535, next_tokens=65535, softmax_lse_flag=softmax_lse_flag)
        
        golden_output = torch.ones(32, 8, 1, 128, dtype=torch.float16).npu()
        res = custom_output[0].equal(golden_output)
        self.assertRtolEqual(res, True)

    @SupportedDevices(['Ascend950'])
    def test_npu_fused_infer_attention_score_v3_page_attention(self, device="npu"):
        query = torch.ones(2, 1, 2048, 128, dtype=torch.float16).npu()
        key = torch.ones((32, 1, 128, 128), dtype=torch.float16).npu()
        value = torch.ones((32, 1, 128, 128), dtype=torch.float16).npu()
        block_table = torch.ones((2, 32), dtype=torch.int32).npu()
        actseqlen = [2048]
        actseqlenkv = [2048]

        softmax_lse_flag = False
        scale = 1 / 0.0078125
        custom_output = torch_npu.npu_fused_infer_attention_score(
            query, key, value, num_heads=1, input_layout="BNSD", block_table=block_table, actual_seq_lengths=actseqlen, actual_seq_lengths_kv=actseqlenkv, 
            block_size=128, scale=scale, pre_tokens=65535, next_tokens=65535, softmax_lse_flag=softmax_lse_flag)
        
        golden_output = torch.ones(2, 1, 2048, 128, dtype=torch.float16).npu()
        res = custom_output[0].equal(golden_output)
        self.assertRtolEqual(res, True)

    @SupportedDevices(['Ascend950'])
    def test_npu_fused_infer_attention_score_v3_int32_antiquant(self, device="npu"):
        query = torch.ones(2, 1, 1, 128, dtype=torch.float16).npu()
        key = torch.ones((2, 1, 256, 16), dtype=torch.int32).npu()
        value = torch.full((2, 1, 256, 16), 286331153, dtype=torch.int32).npu()
        key_antiquant_scale = torch.ones((1, 128), dtype=torch.float16).npu()
        value_antiquant_scale = torch.ones((1, 128), dtype=torch.float16).npu()
        antiquant_mode = 0

        softmax_lse_flag = False
        scale = 1 / 0.0078125
        custom_output = torch_npu.npu_fused_infer_attention_score(
            query, key, value, num_heads=1, input_layout="BNSD", key_antiquant_scale=key_antiquant_scale, value_antiquant_scale=value_antiquant_scale, 
            antiquant_mode=antiquant_mode, scale=scale, pre_tokens=65535, next_tokens=65535, softmax_lse_flag=softmax_lse_flag)
        
        golden_output = torch.ones(2, 1, 1, 128, dtype=torch.float16).npu()
        res = custom_output[0].equal(golden_output)
        self.assertRtolEqual(res, True)

    @SupportedDevices(['Ascend950'])
    def test_npu_fused_infer_attention_score_bnsd_bsnd_d_unequal(self, device="npu"):
        query = torch.ones(32, 8, 1, 192, dtype=torch.float16).npu()
        key = torch.ones((32, 8, 2048, 192), dtype=torch.float16).npu()
        value = torch.ones((32, 8, 2048, 128), dtype=torch.float16).npu()

        softmax_lse_flag = False
        scale = 1 / 0.0078125
        custom_output = torch_npu.npu_fused_infer_attention_score(
            query, key, value, num_heads=8, input_layout="BNSD_BSND", 
            scale=scale, pre_tokens=65535, next_tokens=65535, softmax_lse_flag=softmax_lse_flag)
        
        golden_output = torch.ones(32, 1, 8, 128, dtype=torch.float16).npu()
        res = custom_output[0].equal(golden_output)
        self.assertRtolEqual(res, True)

    @SupportedDevices(['Ascend950'])
    def test_npu_fused_infer_attention_score_bsnd_d_unequal(self, device="npu"):
        query = torch.ones(32, 1, 8, 192, dtype=torch.float16).npu()
        key = torch.ones((32, 2048, 8, 192), dtype=torch.float16).npu()
        value = torch.ones((32, 2048, 8, 128), dtype=torch.float16).npu()

        softmax_lse_flag = False
        scale = 1 / 0.0078125
        custom_output = torch_npu.npu_fused_infer_attention_score(
            query, key, value, num_heads=8, input_layout="BSND", 
            scale=scale, pre_tokens=65535, next_tokens=65535, softmax_lse_flag=softmax_lse_flag)
        
        golden_output = torch.ones(32, 1, 8, 128, dtype=torch.float16).npu()
        res = custom_output[0].equal(golden_output)
        self.assertRtolEqual(res, True)

    @SupportedDevices(['Ascend950'])
    def test_npu_fused_infer_attention_score_bsh_d_unequal(self, device="npu"):
        query = torch.ones(32, 1, 1536, dtype=torch.float16).npu()
        key = torch.ones((32, 2048, 1536), dtype=torch.float16).npu()
        value = torch.ones((32, 2048, 1024), dtype=torch.float16).npu()

        softmax_lse_flag = False
        scale = 1 / 0.0078125
        custom_output = torch_npu.npu_fused_infer_attention_score(
            query, key, value, num_heads=8, input_layout="BSH", 
            scale=scale, pre_tokens=65535, next_tokens=65535, softmax_lse_flag=softmax_lse_flag)
        
        golden_output = torch.ones(32, 1, 1024, dtype=torch.float16).npu()
        res = custom_output[0].equal(golden_output)
        self.assertRtolEqual(res, True)        

    @SupportedDevices(['Ascend950'])
    def test_npu_fused_infer_attention_score_nsd_d_unequal(self, device="npu"):
        query = torch.ones(8, 1, 192, dtype=torch.float16).npu()
        key = torch.ones((8, 2048, 192), dtype=torch.float16).npu()
        value = torch.ones((8, 2048, 128), dtype=torch.float16).npu()

        softmax_lse_flag = False
        scale = 1 / 0.0078125
        custom_output = torch_npu.npu_fused_infer_attention_score(
            query, key, value, num_heads=8, input_layout="NSD", 
            scale=scale, pre_tokens=65535, next_tokens=65535, softmax_lse_flag=softmax_lse_flag)
        
        golden_output = torch.ones(8, 1, 128, dtype=torch.float16).npu()
        res = custom_output[0].equal(golden_output)
        self.assertRtolEqual(res, True)        

    @SupportedDevices(['Ascend950'])
    def test_npu_fused_infer_attention_score_tnd_d_unequal(self, device="npu"):
        query = torch.full((32, 8, 192), 1, dtype=torch.bfloat16).npu()
        key = torch.full((32, 8, 192), 1, dtype=torch.bfloat16).npu()
        value = torch.full((32, 8, 128), 1, dtype=torch.bfloat16).npu()

        actseqlen = [32]
        actseqlenkv = [32]

        softmax_lse_flag = False
        scale = 1 / 0.0078125
        custom_output = torch_npu.npu_fused_infer_attention_score(
            query, key, value, num_heads=8, input_layout="TND",
            scale=scale, pre_tokens=65535, next_tokens=65535, actual_seq_lengths=actseqlen, 
            actual_seq_lengths_kv=actseqlenkv, softmax_lse_flag=softmax_lse_flag)
        
        golden_output = torch.ones(32, 8, 128, dtype=torch.float16).npu()
        res = custom_output[0].equal(golden_output)
        self.assertRtolEqual(res, True)   
if __name__ == "__main__":
    run_tests()

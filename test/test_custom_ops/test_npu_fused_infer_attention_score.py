import math
import unittest
import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestFusedInferAttentionScore(TestCase):
    def softmax(self, x):
        x = x.cpu().numpy().astype(np.float32)
        x_max = x.max(axis=-1, keepdims=True)
        x_sub = x - x_max
        y = np.exp(x_sub)
        x_sum = y.sum(axis=-1, keepdims=True)
        ans = y
        return ans, x_sum, x_max

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

    def custom_op_exec(self, query, key, value, head_dim, softmax_lse_flag):
        scale = 1 / 0.0078125
        return torch_npu.npu_fused_infer_attention_score(
            query, key, value, num_heads=32, input_layout="BNSD", scale=scale, pre_tokens=65535, next_tokens=65535, softmax_lse_flag=softmax_lse_flag)

    def custom_op_exec_tnd(self, query, key, value, head_dim, softmax_lse_flag):
        scale = 1 / 0.0078125
        return torch_npu.npu_fused_infer_attention_score(
            query, key, value, num_heads=32, input_layout="TND", scale=scale, pre_tokens=65535, next_tokens=65535, softmax_lse_flag=softmax_lse_flag)
    
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
        query = torch.randn(2, 32, 128, dtype=torch.float16).npu()
        key = torch.randn(2, 32, 128, dtype=torch.float16).npu()
        value = torch.randn(2, 32, 128, dtype=torch.float16).npu()

        head_dim = 128
        softmax_lse_flag = True
        scale = 1 / 0.0078125
        custom_output = torch_npu.npu_fused_infer_attention_score(
            query, key, value, num_heads=32, input_layout="NTD_TND", scale=scale, pre_tokens=65535,
            next_tokens=65535, softmax_lse_flag=softmax_lse_flag)

        attention_output = custom_output[0]
        softmaxlse_output = custom_output[1]

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

if __name__ == "__main__":
    run_tests()

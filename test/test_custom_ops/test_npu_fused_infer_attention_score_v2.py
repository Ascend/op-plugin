import math
import unittest
import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestFusedInferAttentionV2(TestCase):
    def softmax(self, x):
        x = x.cpu().numpy().astype(np.float32)
        x_max = x.max(axis=-1, keepdims=True)
        x_sub = x - x_max
        y = np.exp(x_sub)
        x_sum = y.sum(axis=-1, keepdims=True)
        ans = y
        return ans, x_sum, x_max

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

    def custom_op_exec_ntd_tnd(self, query, key, value, head_dim, actseqlen, actseqlenkv, return_softmax_lse):
        softmax_scale = 1 / 0.0078125
        return torch_npu.npu_fused_infer_attention_score_v2(
            query, key, value, num_query_heads=2, input_layout="NTD_TND", softmax_scale=softmax_scale,
            pre_tokens=65535, next_tokens=65535, actual_seq_qlen=actseqlen,
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

if __name__ == "__main__":
    run_tests()

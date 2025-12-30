import unittest

import numpy as np
import torch
import torch_npu
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests


class TestFusedInferAttentionV2WithWorkspace(TestCase):
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
            attn_weights1 = torch.nn.functional.softmax(attn_weights1, dim=-1, dtype=torch.float32).to(
                query_states1.dtype)

        attn_output1 = torch.matmul(attn_weights1, past_value)
        return attn_output1, lse

    def custom_op_exec_with_workspace(self, query, key, value, head_dim, return_softmax_lse):
        softmax_scale = 1 / 0.0078125
        workspace = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
            query, key, value, num_query_heads=32, input_layout="BNSD", softmax_scale=softmax_scale,
            pre_tokens=65535, next_tokens=65535, return_softmax_lse=return_softmax_lse)

        attention_out, softmax_lse = torch_npu._npu_fused_infer_attention_score_v2_infer_output(
            query, value, num_query_heads=32, input_layout="BNSD", return_softmax_lse=return_softmax_lse)

        return torch_npu.npu_fused_infer_attention_score_v2.out(
            query, key, value, workspace=workspace, num_query_heads=32, input_layout="BNSD",
            softmax_scale=softmax_scale, pre_tokens=65535,
            next_tokens=65535, return_softmax_lse=return_softmax_lse, out=[attention_out, softmax_lse])

    def custom_op_exec(self, query, key, value, head_dim, return_softmax_lse):
        softmax_scale = 1 / 0.0078125
        workspace = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
            query, key, value, num_query_heads=32, input_layout="BNSD", softmax_scale=softmax_scale,
            pre_tokens=65535, next_tokens=65535, return_softmax_lse=return_softmax_lse)

        attention_out = torch.randn(query.size(), dtype=torch.float16).npu()
        if return_softmax_lse:
            softmax_lse = torch.randn(1, 32, 1, 1, dtype=torch.float32).npu()
        else:
            softmax_lse = torch.randn(1, dtype=torch.float32).npu()

        return torch_npu.npu_fused_infer_attention_score_v2.out(
            query, key, value, num_query_heads=32, input_layout="BNSD", softmax_scale=softmax_scale, pre_tokens=65535,
            next_tokens=65535, return_softmax_lse=return_softmax_lse, out=[attention_out, softmax_lse])

    @unittest.skip(
        "Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_fused_infer_attention_score_v2_ifa_return_lse_return_softmax_lse_true(self, device="npu"):
        query = torch.randn(1, 32, 1, 128, dtype=torch.float16).npu()
        key = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()
        value = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()

        head_dim = 128
        return_softmax_lse = True

        supported_output, lse_out = self.supported_op_exec(query, key, value, head_dim, 1, 32, 1, return_softmax_lse)
        custom_output = self.custom_op_exec(query, key, value, head_dim, True)
        attention_output = custom_output[0]
        softmaxlse_output = custom_output[1]

    @unittest.skip(
        "Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_fused_infer_attention_score_v2_ifa_return_lse_return_softmax_lse_false(self, device="npu"):
        query = torch.randn(1, 32, 1, 128, dtype=torch.float16).npu()
        key = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()
        value = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()

        head_dim = 128
        return_softmax_lse = False

        supported_output, lse_out = self.supported_op_exec(query, key, value, head_dim, 1, 32, 1, return_softmax_lse)
        custom_output = self.custom_op_exec(query, key, value, head_dim, False)
        attention_output = custom_output[0]
        softmaxlse_output = custom_output[1]

    @unittest.skip(
        "Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_fused_infer_attention_score_v2_ifa_return_lse_return_softmax_lse_true_with_workspace(self, device="npu"):
        query = torch.randn(1, 32, 1, 128, dtype=torch.float16).npu()
        key = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()
        value = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()

        head_dim = 128
        return_softmax_lse = True

        supported_output, lse_out = self.supported_op_exec(query, key, value, head_dim, 1, 32, 1, return_softmax_lse)
        custom_output = self.custom_op_exec_with_workspace(query, key, value, head_dim, True)
        attention_output = custom_output[0]
        softmaxlse_output = custom_output[1]


if __name__ == "__main__":
    run_tests()

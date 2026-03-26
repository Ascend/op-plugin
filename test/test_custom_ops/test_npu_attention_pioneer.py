import math
import unittest
import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestAttentionPioneer(TestCase):
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
            attn_weights1 = torch.from_numpy(softmax_res / softmax_sum).to(query_states1.dtype).to(query_states1.device)
        else:
            lse = np.zeros([B, N, S, 1], np.float32)
            attn_weights1 = torch.max(attn_weights1, torch.full(
                (1, 1), torch.finfo(attn_weights1.dtype).min, device=attn_weights1.device))
            attn_weights1 = torch.nn.functional.softmax(attn_weights1, dim=-1, dtype=torch.float32).to(query_states1.dtype)

        attn_output1 = torch.matmul(attn_weights1, past_value)
        return attn_output1, lse

    def custom_op_exec_tnd_pa(self, query, key, value, return_softmax_lse, block_table):
        softmax_scale = 1 / 0.0078125
        return torch_npu._npu_attention_pioneer(
            query, key, value, num_heads=1, input_layout="TND", scale=softmax_scale,
            pre_tokens=65535, next_tokens=65535, softmax_lse_flag=return_softmax_lse, block_table=block_table)
    
    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend950'])
    def test_npu_attention_pioneer(self, device="npu"):
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
        self.assertRtolEqual(golden_output, attention_output, prec=0.000001, prec16=0.000001)

if __name__ == "__main__":
    run_tests()

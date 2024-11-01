import math
import unittest
import copy
import struct
from struct import pack, unpack
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestQuantScatter(TestCase):
    def supported_op_exec(self, query, key, cos, sin):
        x1 = query[..., :64].cpu()
        x2 = query[..., 64:].cpu()
        concat = np.concatenate((-x2, x1), axis=-1)
        x2_mul = torch.from_numpy(concat).npu() * sin
        x1_mul = query * cos
        res0 = x2_mul + x1_mul

        k1 = key[..., :64].cpu()
        k2 = key[..., 64:].cpu()
        concatk = np.concatenate((-k2, k1), axis=-1)
        x1k_mul = torch.from_numpy(concatk).npu() * sin
        x2k_mul = key * cos
        res1 = x2k_mul + x1k_mul
        return [res0, res1]

    def custom_op_exec(self, query, key, cos, sin):
        return torch_npu.npu_apply_rotary_pos_emb(query, key, cos, sin, '1')

    @SupportedDevices(['Ascend910B'])
    def test_npu_apply_rotary_pos_emb(self, device="npu"):
        query_data = np.random.uniform(0, 1, [4, 1024, 16, 128]).astype(np.float16)
        query1 = torch.from_numpy(query_data).to(torch.float16).npu()
        query2 = query1.clone()

        key_data = np.random.uniform(0, 1, [4, 1024, 16, 128]).astype(np.float16)
        key1 = torch.from_numpy(key_data).to(torch.float16).npu()
        key2 = key1.clone()

        cos_data = np.random.uniform(0, 1, [4, 1024, 1, 128]).astype(np.float16)
        cos1 = torch.from_numpy(cos_data).to(torch.float16).npu()
        cos2 = cos1.clone()

        sin_data = np.random.uniform(0, 1, [4, 1024, 1, 128]).astype(np.float16)
        sin1 = torch.from_numpy(sin_data).to(torch.float16).npu()
        sin2 = sin1.clone()

        supported_output = self.supported_op_exec(query1, key1, cos1, sin1)
        custom_output = self.custom_op_exec(query2, key2, cos2, sin2)
        self.assertRtolEqual(supported_output, custom_output, 0.001)

if __name__ == "__main__":
    run_tests()
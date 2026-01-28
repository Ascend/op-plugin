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


class TestGroupedMatmul(TestCase):

    def single_matmul(self, y, x1, x2, x1_scale, x2_scale):
        output = np.matmul(x1, x2)
        output = output * x1_scale * x2_scale
        output = output + y
        return output

    @SupportedDevices(['Ascend910_95', 'Ascend950'])
    def test_npu_grouped_matmul_quant_950(self): # 量化 单单单
        torch.manual_seed(0)
        group_list = torch.tensor([16, ]).to(torch.int64).npu()

        x1 = torch.randint(2, 3, size=(16, 256), dtype=torch.float32)
        x2 = torch.randint(2, 3, size=(1, 256, 32), dtype=torch.float32)

        x1_clone = x1.clone().npu()
        x2_clone = x2.clone().npu()

        y = torch.randint(2, 3, size=(1, 16, 32), dtype=torch.float32)
        y1_clone = y.clone().npu()
        y2_clone = y.clone().npu()
        x2_scale = np.random.uniform(low=-5, high=5, size=(1, 32)).astype(np.float32)
        x1_scale = np.random.uniform(low=-5, high=5, size=(1,)).astype(np.float32)
        x2_scale_clone = x2_scale.clone.npu()
        x1_scale_clone = x1_scale.clone.npu()
        x1_dtype = torch_npu.float8_e8m0
        x2_dtype = torch_npu.float8_e8m0
        supported_output = self.single_matmul(y, x1, x2, x1_scale, x2_scale)
        custom_output1 = torch_npu.npu_add_quant_gmm_(y1, x1, x2, x2_scale, group_list, x1_scale=x1_scale,
                                                      group_list_type=0, group_sizes=None, x1_dtype=x1_dtype,
                                                      x2_dtype=x2_dtype)

        custom_output2 = torch_npu.npu_add_quant_gmm(y2, x1, x2, x2_scale, group_list, x1_scale=x1_scale,
                                                     group_list_type=0, group_sizes=None, x1_dtype=x1_dtype,
                                                     x2_dtype=x2_dtype)
        self.assertRtolEqual(supported_output, custom_output1.cpu().numpy(), 0.001)
        self.assertRtolEqual(supported_output, custom_output2.cpu().numpy(), 0.001)

if __name__ == "__main__":
    run_tests()
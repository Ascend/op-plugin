import math
import unittest
import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNPUWeightQuantBatchMatmul(TestCase):

    def supported_op_exec(self, x, weight, antiquant_scale, antiquant_offset):
        res = torch.matmul(x, (weight + antiquant_offset) * antiquant_scale)
        return res

    def custom_op_exec(self, x, weight, antiquant_scale, antiquant_offset):
        return torch_npu.npu_weight_quant_batchmatmul(x, weight, antiquant_scale, antiquant_offset)

    @SupportedDevices(['Ascend310P'])
    def test_npu_weight_quant_batchmatmul2(self, device="npu"):
        torch.manual_seed(0)
        x = torch.randn((4, 32, 1024, 128), dtype=torch.float16).npu()
        weight = torch.randn((4, 32, 128, 1024), dtype=torch.int8).npu()
        antiquant_scale = torch.randn((1, 1024), dtype=torch.float16).npu()
        antiquant_offset = torch.randn((1, 1024), dtype=torch.float16).npu()

        x_clone = x.clone()
        weight_clone = weight.clone()
        antiquant_scale_clone = antiquant_scale.clone()
        antiquant_offset_clone = antiquant_offset.clone()

        supported_output = self.supported_op_exec(
            x, weight, antiquant_scale, antiquant_offset)
        custom_output = self.custom_op_exec(
            x_clone, weight_clone, antiquant_scale_clone, antiquant_offset_clone)

        self.assertRtolEqual(supported_output, custom_output, 0.001)


if __name__ == "__main__":
    run_tests()

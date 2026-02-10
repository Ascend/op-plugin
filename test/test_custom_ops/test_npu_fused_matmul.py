import math
import unittest
import numpy as np
import torch

import torch_npu
import torch.nn as nn
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNPUFusedMatmul(TestCase):

    def supported_op_exec(self, x1, x2, bias, x3, fused_op_type):
        res = torch.matmul(x1, x2)
        if fused_op_type == "add":
            res = torch.add(res, x3)
        elif fused_op_type == "mul":
            res = torch.mul(res, x3)
        elif fused_op_type == "gelu_erf":
            m = nn.GELU()
            res = m(res)
        elif fused_op_type == "gelu_tanh":
            m = nn.GELU('tanh')
            res = m(res)
        return res

    def custom_op_exec(self, x1, x2, bias, x3, fused_op_type):
        return torch_npu.npu_fused_matmul(x1, x2, bias=None, x3=x3, fused_op_type=fused_op_type)

    @SupportedDevices(['Ascend950'])
    def test_npu_fused_matmul_add(self, device="npu"):
        torch.manual_seed(0)
        x1 = torch.randn((16, 48), dtype=torch.float16).npu()
        x2 = torch.randn((48, 16), dtype=torch.float16).npu()
        x3 = torch.randn((16, 16), dtype=torch.float16).npu()
        fused_op_type = "add"
        supported_output = self.supported_op_exec(
            x1, x2, None, x3, fused_op_type)
        custom_output = self.custom_op_exec(
            x1, x2, None, x3, fused_op_type)

        self.assertRtolEqual(supported_output, custom_output, 0.001)

    @SupportedDevices(['Ascend950'])
    def test_npu_fused_matmul_mul(self, device="npu"):
        torch.manual_seed(0)
        x1 = torch.randn((16, 48), dtype=torch.float16).npu()
        x2 = torch.randn((48, 16), dtype=torch.float16).npu()
        x3 = torch.randn((16, 16), dtype=torch.float16).npu()
        fused_op_type = "mul"
        supported_output = self.supported_op_exec(
            x1, x2, None, x3, fused_op_type)
        custom_output = self.custom_op_exec(
            x1, x2, None, x3, fused_op_type)

        self.assertRtolEqual(supported_output, custom_output, 0.001)

    @SupportedDevices(['Ascend950'])
    def test_npu_fused_matmul_gelu_erf(self, device="npu"):
        torch.manual_seed(0)
        x1 = torch.randn((16, 48), dtype=torch.float16).npu()
        x2 = torch.randn((48, 16), dtype=torch.float16).npu()
        fused_op_type = "gelu_erf"
        supported_output = self.supported_op_exec(
            x1, x2, None, None, fused_op_type)
        custom_output = self.custom_op_exec(
            x1, x2, None, None, fused_op_type)

        self.assertRtolEqual(supported_output, custom_output, 0.001)


if __name__ == "__main__":
    run_tests()

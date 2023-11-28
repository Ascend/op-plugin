import math
import unittest
import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestFFN(TestCase):

    def supported_op_exec(self, x, weight1, weight2, activation):
        mm1_res = torch.matmul(x, weight1)
        activation_res = torch.nn.functional.relu(mm1_res)
        mm2_res = torch.matmul(activation_res, weight2)
        return mm2_res

    def custom_op_exec(self, x, weight1, weight2, activation):
        return torch_npu.npu_ffn(x, weight1, weight2, activation, inner_precise=1)

    @unittest.skipIf(DEVICE_NAME != 'Ascend910B',
        "OP `FFN` is only supported on 910B, skip this ut for this device type!")
    def test_npu_ffn(self, device="npu"):
        torch.mannal_seed(0)
        x = torch.randn(8192, 320, dtype=torch.float16).npu()
        weight1 = torch.randn(320, 2560, dtype=torch.float16).npu()
        weight2 = torch.randn(2560, 320, dtype=torch.float16).npu()
        x_clone = x.clone()
        weight1_clone = weight1.clone()
        weight2_clone = weight2.clone()
        activation = "relu"

        supported_output = self.supported_op_exec(x, weight1, weight2, activation)
        custom_output = self.custom_op_exec(x_clone, weight1_clone, weight2_clone, activation)
        self.assertRtolEqual(x, x_clone, 0.001)
        self.assertRtolEqual(supported_output, custom_output, 0.001)


if __name__ == "__main__":
    run_tests()

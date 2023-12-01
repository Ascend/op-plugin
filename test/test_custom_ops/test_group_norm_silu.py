import math
import unittest
import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestGroupNormSilu(TestCase):

    def supported_op_exec(self, x, gama, beta, group, eps):
        res = torch.ops.aten.native_group_norm(x, gama, beta, x.shape[0], x.shape[1], x.shape[2] * x.shape[3], group, eps)
        res[0]=torch.nn.functional.silu(res[0])
        return res

    def custom_op_exec(self, x, gama, beta, group, eps):
        return torch_npu.npu_group_norm_silu(x, gama, beta, group, eps)

    @unittest.skipIf(DEVICE_NAME != 'Ascend910B',
        "OP `GroupNormSilu` is only supported on 910B, skip this ut for this device type!")
    def test_npu_(self, device="npu"):
        x = torch.randn(24, 320, 48, 48 dtype=torch.float32).npu()
        gama = torch.randn(320, dtype=torch.float32).npu()
        beta = torch.randn(320, dtype=torch.float32).npu()

        group = 32
        eps = 0.000100

        supported_output = self.supported_op_exec(x, gama, beta, group, eps)
        custom_output = self.custom_op_exec(x, gama, beta, group, eps)
        self.assertRtolEqual(supported_output, custom_output)


if __name__ == "__main__":
    run_tests()

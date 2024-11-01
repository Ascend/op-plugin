import math
import unittest
import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False


class TestNpuScatterNdUpdate(TestCase):

    def supported_op_exec(self, var, indices_tensor, updates):
        for i in range(len(var)):
            if i < len(indices_tensor):
                var[indices_tensor[i][0]][indices_tensor[i][1]] = updates[i]

        return var

    def custom_op_exec(self, var, indices_tensor, updates):
        return torch_npu.npu_scatter_nd_update(var, indices_tensor, updates)

    @SupportedDevices(['Ascend910B'])
    def test_npu_scatter_nd_update(self, device="npu"):
        var = torch.zeros([3, 2], dtype=torch.float16).npu()
        indices = np.array([[0, 0], [1, 1]])
        indices_tensor = torch.from_numpy(indices).to(device)
        updates = np.array([10, 20])
        updates_tensor = torch.from_numpy(updates.astype(np.float16)).to(device)

        supported_output = self.supported_op_exec(var, indices_tensor, updates)
        custom_output = self.custom_op_exec(var, indices_tensor, updates_tensor)
        self.assertEqual(supported_output, custom_output)


if __name__ == "__main__":
    run_tests()

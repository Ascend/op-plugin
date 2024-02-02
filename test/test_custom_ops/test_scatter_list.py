import math
import unittest
import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False


class TestScatterList(TestCase):

    def supported_op_exec(self, var_list, indice, updates, axis=-2):
        if axis == -2:
            for i, item in enumerate(var_list):
                for j in range(var_list[0].shape[0]):
                    item[j][indice[i]] = updates[i][j][0]
        elif axis == -1:
            for i, item in enumerate(var_list):
                for j in range(var_list[0].shape[0]):
                    for k in range(var_list[0].shape[1]):
                        item[j][k][indice[i]] = updates[i][j][k][0]

        return var_list

    def custom_op_exec(self, var_list, indice, updates, mask):
        reduce = 'update'
        axis = -2
        return torch_npu.npu_scatter_list(var_list, indice, updates, mask, reduce, axis)

    @SupportedDevices(['Ascend910B'])
    def test_npu_scatter_list(self, device="npu"):
        if torch.__version__ > '2.0':
            var_list = []
            for i in range(8):
                var = torch.zeros([4, 4096, 256], dtype=torch.float16).npu()
                var_list.append(var)
            indice = torch.zeros([8], dtype=torch.int32).npu()
            updates = torch.ones([8, 4, 1, 256], dtype=torch.float16).npu()
            mask = torch.ones([8], dtype=torch.uint8).npu()

            supported_output = self.supported_op_exec(var_list, indice, updates, axis=-2)
            custom_output = self.custom_op_exec(var_list, indice, updates, mask)

            for i in range(8):
                self.assertEqual(supported_output[i], custom_output[i])


if __name__ == "__main__":
    run_tests()

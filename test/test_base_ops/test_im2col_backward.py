import unittest
import itertools
import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestIm2colBackward(TestCase):

    def compute_re(self, output, golden):
        diff_value = torch.abs(torch.subtract(output.to(golden.dtype), golden))
        diff_value_rel = diff_value / (torch.abs(golden) + 1e-7)
        max_re = torch.max(diff_value_rel).item()
        avg_re = torch.mean(diff_value_rel).item()

        diff = torch.subtract(output.to(golden.dtype), golden)
        rmse = torch.sqrt(torch.sum(diff*diff)/diff.numel()).item()
        return max_re, avg_re, rmse

    def test_im2col_backward(self):
        dtype_list = [np.float16, np.float32]
        shape_list = [[1, 144, 256], [144, 256]]
        for dtype, shape in itertools.product(dtype_list, shape_list):
            fold_cpu = torch.nn.Fold(output_size=(18, 18), kernel_size=(3, 3))
            fold_npu = fold_cpu.npu()
            cpu_input, npu_input = create_common_tensor((dtype, 0, shape), -100, 100)
            cpu_output = fold_cpu(cpu_input)
            npu_output = fold_npu(npu_input)
            if dtype == np.float16:
                golden = fold_cpu(cpu_input.float())
                cpu_max_re, cpu_avg_re, cpu_rmse = self.compute_re(cpu_output, golden)
                npu_max_re, npu_avg_re, npu_rmse = self.compute_re(npu_output.cpu(), golden)
                self.assertLessEqual(npu_max_re/cpu_max_re, 2)
                self.assertLessEqual(npu_avg_re/cpu_avg_re, 1.2)
                self.assertLessEqual(npu_rmse/cpu_rmse, 1.2)

            else:
                self.assertRtolEqual(cpu_output, npu_output.cpu())


if __name__ == '__main__':
    run_tests()

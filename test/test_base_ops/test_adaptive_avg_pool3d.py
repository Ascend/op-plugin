import torch
import torch.nn as nn
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestAdaptiveAvgPool3d(TestCase):
    def cpu_op_exec(self, input1, output_size):
        m = nn.AdaptiveAvgPool3d(output_size)
        output = m(input1)
        return output.numpy()

    def npu_op_exec(self, input1, output_size):
        m = nn.AdaptiveAvgPool3d(output_size)
        output = m(input1).cpu()
        return output.numpy()

    def test_adaptive_avg_pool3d_shape_format_fp16(self, device="npu"):
        shape_format = [
            [np.float16, -1, (64, 10, 16, 32)],
            [np.float16, -1, (4, 16, 8, 4, 2)],
            [np.float16, -1, (2, 16, 4, 32)],
            [np.float16, -1, (4, 16, 8, 4, 16)],
        ]
        output_list = [(1, 1, 1)]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 1, 10)
            cpu_input = cpu_input.to(torch.float32)
            for output_size in output_list:
                cpu_output = self.cpu_op_exec(cpu_input, output_size)
                npu_output = self.npu_op_exec(npu_input, output_size)
                cpu_output = cpu_output.astype(npu_output.dtype)
                self.assertRtolEqual(cpu_output, npu_output)

    def test_adaptive_avg_pool3d_shape_format_fp32(self, device="npu"):
        shape_format = [
            [np.float32, -1, (64, 10, 16, 32)],
            [np.float32, -1, (4, 2, 2, 4, 316)],
            [np.float32, -1, (2, 16, 4, 32)],
            [np.float32, -1, (4, 16, 8, 4, 16)],
        ]
        output_list = [(1, 1, 1)]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 1, 10)
            for output_size in output_list:
                cpu_output = self.cpu_op_exec(cpu_input, output_size)
                npu_output = self.npu_op_exec(npu_input, output_size)
                self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()

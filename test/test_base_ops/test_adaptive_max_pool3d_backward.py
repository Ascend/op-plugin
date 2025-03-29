import torch
import numpy as np
import torch.nn as nn
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.common_utils import SupportedDevices


class TestAdaptiveMaxPool3dBackward(TestCase):
    def cpu_op_exec(self, input_x, output_size):
        input_x.requires_grad_(True)
        m = nn.AdaptiveMaxPool3d(output_size)
        output = m(input_x)
        output.backward(torch.ones_like(output))
        grad_output = input_x.grad
        return output.detach().numpy(), grad_output.numpy()

    def npu_op_exec(self, input_x, output_size):
        input_x.requires_grad_(True)
        m = nn.AdaptiveMaxPool3d(output_size).npu()
        output = m(input_x)
        output.backward(torch.ones_like(output))
        grad_output = input_x.grad
        return output.detach().cpu().numpy(), grad_output.cpu().numpy()

    @SupportedDevices(['Ascend910B'])
    def test_adaptive_max_pool3d_backward_fp32(self):
        format_list = [-1]
        shape_list = [
            [2, 3, 4, 5, 6],
            [1, 2, 6, 8, 8],
            [4, 3, 3, 6, 6]
        ]
        output_size_list = [(1, 1, 1), (2, 2, 2), (3, 3, 3), (1, 2, 3)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -100, 100)
            for output_size in output_size_list:
                cpu_output = self.cpu_op_exec(cpu_input, output_size)
                npu_output = self.npu_op_exec(npu_input, output_size)
                self.assertRtolEqual(cpu_output[0], npu_output[0])
                self.assertRtolEqual(cpu_output[1], npu_output[1])

    @SupportedDevices(['Ascend910B'])
    def test_adaptive_max_pool3d_backward_fp16(self):
        format_list = [-1]
        shape_list = [
            [2, 3, 4, 5, 6],
            [1, 2, 6, 8, 8],
            [4, 3, 3, 6, 6]
        ]
        output_size_list = [(1, 1, 1), (2, 2, 2), (3, 3, 3), (1, 2, 3)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -100, 100)
            cpu_input = cpu_input.to(torch.float32)
            npu_input = npu_input.to(torch.float32)
            for output_size in output_size_list:
                cpu_output = self.cpu_op_exec(cpu_input, output_size)
                npu_output = self.npu_op_exec(npu_input, output_size)
                self.assertRtolEqual(cpu_output[0].astype(np.float16),
                                   npu_output[0].astype(np.float16))
                self.assertRtolEqual(cpu_output[1].astype(np.float16),
                                   npu_output[1].astype(np.float16))

    @SupportedDevices(['Ascend910B'])
    def test_adaptive_max_pool3d_backward_fp64(self):
        format_list = [-1]
        shape_list = [
            [2, 3, 4, 5, 6],
            [1, 2, 6, 8, 8],
            [4, 3, 3, 6, 6]
        ]
        output_size_list = [(1, 1, 1), (2, 2, 2), (3, 3, 3), (1, 2, 3)]
        shape_format = [
            [np.float64, i, j] for i in format_list for j in shape_list
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -100, 100)
            for output_size in output_size_list:
                cpu_output = self.cpu_op_exec(cpu_input, output_size)
                npu_output = self.npu_op_exec(npu_input, output_size)
                self.assertRtolEqual(cpu_output[0], npu_output[0])
                self.assertRtolEqual(cpu_output[1], npu_output[1])

if __name__ == "__main__":
    run_tests()

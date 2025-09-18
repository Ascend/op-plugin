import unittest
import torch
import numpy as np
import torch.nn as nn
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestSlowConv3d(TestCase):
    def op_exec_cpu(self, x, weight, bias, kernel_size, stride=1, padding=0):
        cpuOutput = torch._C._nn.slow_conv3d(input=x, weight=weight, bias=bias, kernel_size=kernel_size,
                                             stride=stride, padding=padding)
        return cpuOutput

    def op_exec_npu(self, x, weight, bias, kernel_size, stride=1, padding=0):
        npuOutput = torch._C._nn.slow_conv3d(input=x, weight=weight, bias=bias, kernel_size=kernel_size,
                                             stride=stride, padding=padding)
        return npuOutput.to("cpu")

    def slow_conv3d_result(self, shape_format):
        for item in shape_format:
            fp16_flag = False
            np.random.seed(1234)
            input_cpu, input_npu = create_common_tensor(item[0], 0, 1)
            if input_cpu.dtype == torch.float16:
                fp16_flag = True
                input_cpu = input_cpu.to(torch.float32)
            weight_cpu, weight_npu = create_common_tensor(item[1], 0, 1)
            if weight_cpu.dtype == torch.float16:
                weight_cpu = weight_cpu.to(torch.float32)
            bias_cpu, bias_npu = create_common_tensor(item[2], 0, 1)
            if bias_cpu.dtype == torch.float16:
                bias_cpu = bias_cpu.to(torch.float32)
            kernel_size = (item[1][2][2], item[1][2][3], item[1][2][4])
            cpu_output = self.op_exec_cpu(input_cpu, weight_cpu, bias_cpu, kernel_size)
            npu_output = self.op_exec_npu(input_npu, weight_npu, bias_npu, kernel_size)
            if fp16_flag:
                cpu_output = cpu_output.to(torch.float16)
            self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.cpu().detach().numpy(), 1e-3)

    def test_slow_conv3d_fp16(self):
        shape_format = [  # input, weight, bias
            [[np.float16, 30, [1, 128, 4, 14, 14]],
             [np.float16, 30, [1, 128, 3, 3, 3]], [np.float16, 30, [1]]],
            [[np.float16, 30, [1, 64, 4, 14, 14]],
             [np.float16, 30, [1, 64, 3, 3, 3]], [np.float16, 30, [1]]],
            [[np.float16, 30, [1, 64, 4, 14, 14]],
             [np.float16, 30, [2, 64, 3, 3, 3]], [np.float16, 30, [2]]],
        ]
        self.slow_conv3d_result(shape_format)

    @unittest.skip("skip test_slow_conv3d_fp32 now")
    def test_slow_conv3d_fp32(self):
        shape_format = [  # input, weight, bias
            [[np.float32, 30, [1, 128, 4, 14, 14]],
             [np.float32, 30, [1, 128, 3, 3, 3]], [np.float32, 30, [1]]],
            [[np.float32, 30, [1, 64, 4, 14, 14]],
             [np.float32, 30, [1, 64, 3, 3, 3]], [np.float32, 30, [1]]],
            [[np.float32, 30, [1, 64, 4, 14, 14]],
             [np.float32, 30, [2, 64, 3, 3, 3]], [np.float32, 30, [2]]],
        ]
        self.slow_conv3d_result(shape_format)


if __name__ == "__main__":
    run_tests()

import torch
import numpy as np
from torch.nn import functional as F
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestAdaptiveAvgPool2dBackward(TestCase):
    def cpu_op_exec(self, input_x, input_grad):
        input_x.requires_grad_(True)
        m = torch.nn.AdaptiveAvgPool2d(input_grad)
        if input_x.dtype == torch.half:
            output = m(input_x.float()).half()
        else:
            output = m(input_x)
        output.backward(output)
        out = output.detach(), input_x.grad
        return out

    def npu_op_exec(self, input_x, input_grad):
        input_x.requires_grad_(True)
        m = torch.nn.AdaptiveAvgPool2d(input_grad)
        output = m(input_x)
        output.backward(output)
        out = output.detach().cpu(), input_x.grad.cpu()
        return out

    def cpu_op_exec_func(self, input_x, output_size):
        input_x.requires_grad_(True)
        output = F.adaptive_avg_pool2d(input_x, output_size)
        if input_x.dtype == torch.half:
            output = F.adaptive_avg_pool2d(input_x.float(), output_size).half()
        else:
            output = F.adaptive_avg_pool2d(input_x, output_size)
        output.backward(output)
        out = output.detach(), input_x.grad
        return out

    def npu_op_exec_func(self, input_x, output_size):
        input_x.requires_grad_(True)
        output = F.adaptive_avg_pool2d(input_x, output_size)
        output.backward(output)
        out = output.detach().cpu(), input_x.grad.cpu()
        return out

    def test_adaptiveAvgPool2d_backward_1(self):
        torch.manual_seed(123)
        cpu_input = torch.randn((1, 8, 9), dtype=torch.float32)
        npu_input = cpu_input.npu()
        output_size = np.array((2, 3))
        cpu_output = self.cpu_op_exec(cpu_input, output_size)
        npu_output = self.npu_op_exec(npu_input, output_size)
        self.assertRtolEqual(cpu_output[0], npu_output[0], prec=1e-3)
        self.assertRtolEqual(cpu_output[1], npu_output[1], prec=1e-3)

    def test_adaptiveAvgPool2d_backward_2(self):
        torch.manual_seed(123)
        cpu_input = torch.randn((1, 3, 3, 3), dtype=torch.float32)
        npu_input = cpu_input.npu()
        output_size = np.array((2, 2))
        cpu_output = self.cpu_op_exec(cpu_input, output_size)
        npu_output = self.npu_op_exec(npu_input, output_size)
        self.assertRtolEqual(cpu_output[0], npu_output[0], prec=1e-3)
        self.assertRtolEqual(cpu_output[1], npu_output[1], prec=1e-3)

    def test_adaptiveAvgPool2d_backward_fp16(self):
        input_x = np.random.uniform(0, 1, (1, 3, 6, 6)).astype(np.float16)
        cpu_input = torch.from_numpy(input_x)
        npu_input = cpu_input.npu()
        output_size = np.array((5, 5))
        cpu_output = self.cpu_op_exec(cpu_input, output_size)
        npu_output = self.npu_op_exec(npu_input, output_size)
        self.assertRtolEqual(cpu_output[0], npu_output[0])
        self.assertRtolEqual(cpu_output[1], npu_output[1])

    @SupportedDevices(['Ascend910B'])
    def test_adaptiveAvgPool2d_backward_special_case(self):
        torch.manual_seed(123)
        cpu_input = torch.randn((1, 1, 1, 3, 3), dtype=torch.float32)
        npu_input = cpu_input.npu()
        output_size = 1
        cpu_output = self.cpu_op_exec_func(cpu_input, output_size)
        npu_output = self.npu_op_exec_func(npu_input, output_size)
        self.assertRtolEqual(cpu_output[0], npu_output[0], prec=1e-3)
        self.assertRtolEqual(cpu_output[1], npu_output[1], prec=1e-3)

    @SupportedDevices(['Ascend910B'])
    def test_adaptiveAvgPool2d_backward_check(self):
        torch.manual_seed(123)
        npu_input = torch.randn((1, 3, 3, 3), dtype=torch.float32).npu()
        output_size = -1
        msg = "adaptive_avg_pool2d: elements of output_size must be greater than or equal to 0"
        with self.assertRaisesRegex(RuntimeError, msg):
            F.adaptive_avg_pool2d(npu_input, output_size)
        
        npu_input = torch.randn((1, 3, 3, 3), dtype=torch.float32).npu()
        output_size = np.array((2, 2, 2))
        msg = "adaptive_avg_pool2d: output size must be 2"
        with self.assertRaisesRegex(RuntimeError, msg):
            F.adaptive_avg_pool2d(npu_input, output_size)


if __name__ == "__main__":
    run_tests()

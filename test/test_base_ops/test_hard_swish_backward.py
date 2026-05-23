import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestHardSwishBackWard(TestCase):
    def cpu_op_exec(self, input1):
        input1 = input1.clone().detach().requires_grad_(True)
        cpu_output = torch.nn.functional.hardswish(input1, inplace=False)
        cpu_output.backward(torch.ones_like(cpu_output))
        output_grad = input1.grad
        output_grad = output_grad.detach().numpy()
        cpu_output = cpu_output.detach().numpy()

        return cpu_output, output_grad

    def npu_op_exec(self, input1):
        input1 = input1.clone().detach().requires_grad_(True)
        output = torch.nn.functional.hardswish(input1, inplace=False)
        output.backward(torch.ones_like(output))
        output = output.to("cpu")
        output_grad = input1.grad
        output_grad = output_grad.to("cpu")
        output_grad = output_grad.detach().numpy()
        output = output.detach().numpy()

        return output, output_grad

    def backward_create_shape_format16(self):
        backward_format_list = [0, 3, 29]
        backward_dtype_list = [np.float16]
        backward_shape_list = [[32], [32, 3], [32, 3, 3], [64, 32, 3, 3]]
        # pylint:disable = complicate-comprehension
        backward_shape_format = [[i, j, k] for i in backward_dtype_list
                                 for j in backward_format_list for k in backward_shape_list]

        return backward_shape_format

    def backward_create_shape_format32(self):
        backward_format_list32 = [0, 3, 29]
        backward_dtype_list32 = [np.float32]
        backward_shape_list32 = [[32], [32, 3], [32, 3, 3], [64, 32, 3, 3]]
        # pylint:disable = complicate-comprehension
        backward_shape_format32 = [[i, j, k] for i in backward_dtype_list32
                                   for j in backward_format_list32 for k in backward_shape_list32]

        return backward_shape_format32

    def test_hardswish_shape_format_fp16(self):
        for item in self.backward_create_shape_format16():
            cpu_input1, npu_input1 = create_common_tensor(item, 2, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)

            cpu_output, cpu_output_grad = self.cpu_op_exec(cpu_input1)
            npu_output, npu_output_grad = self.npu_op_exec(npu_input1)
            cpu_output = cpu_output.astype(npu_output.dtype)
            cpu_output_grad = cpu_output_grad.astype(npu_output_grad.dtype)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output_grad, npu_output_grad)

    def test_hardswish_shape_format_fp32(self):
        for item in self.backward_create_shape_format32():
            cpu_input1, npu_input1 = create_common_tensor(item, 2, 100)
            cpu_output, cpu_output_grad = self.cpu_op_exec(cpu_input1)
            npu_output, npu_output_grad = self.npu_op_exec(npu_input1)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output_grad, npu_output_grad)

    def _boundary_values(self):
        """Values around hardswish boundaries -3 and 3."""
        return [-3.5, -3.1, -3.0, -2.999, -2.5, -1.0, 0.0, 1.0, 2.5, 2.999, 3.0, 3.1, 3.5]

    def test_hardswish_boundary_fp32(self):
        boundary = torch.tensor([self._boundary_values()], dtype=torch.float32)
        cpu_output, cpu_grad = self.cpu_op_exec(boundary)
        npu_input = boundary.to("npu")
        npu_output, npu_grad = self.npu_op_exec(npu_input)
        self.assertRtolEqual(cpu_output, npu_output)
        self.assertRtolEqual(cpu_grad, npu_grad)

    def test_hardswish_boundary_fp16(self):
        cpu_input = torch.tensor([self._boundary_values()], dtype=torch.float32)
        cpu_output, cpu_grad = self.cpu_op_exec(cpu_input)
        npu_input = torch.tensor([self._boundary_values()], dtype=torch.float16).to("npu")
        npu_output, npu_grad = self.npu_op_exec(npu_input)
        cpu_output = cpu_output.astype(npu_output.dtype)
        cpu_grad = cpu_grad.astype(npu_grad.dtype)
        self.assertRtolEqual(cpu_output, npu_output)
        self.assertRtolEqual(cpu_grad, npu_grad)


if __name__ == "__main__":
    run_tests()

import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


torch_npu.npu.set_compile_mode(jit_compile=False)


class TestPolar(TestCase):
    def cpu_op_exec(self, input1, input2):
        input1.requires_grad_(True)
        input2.requires_grad_(True)
        output = torch.polar(input1, input2)
        output.real.sum().backward()
        return output, input1.grad, input2.grad

    def npu_op_exec(self, input1, input2):
        input1.requires_grad_(True)
        input2.requires_grad_(True)
        output = torch.polar(input1, input2)
        output.real.sum().backward()
        return output, input1.grad, input2.grad

    def test_polar_shape_format(self):
        shape_format = [
            [np.float32, (3, 3, 2)],
            [np.float32, (4, 3)],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor([item[0], 0, item[1]], -1, 1)
            cpu_input2, npu_input2 = create_common_tensor([item[0], 0, item[1]], -1, 1)
            
            cpu_output, cpu_input1_grad, cpu_input2_grad = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output, npu_input1_grad, npu_input2_grad = self.npu_op_exec(npu_input1, npu_input2)

            self.assertRtolEqual(cpu_output.real, npu_output.real)
            self.assertRtolEqual(cpu_output.imag, npu_output.imag)
            self.assertRtolEqual(cpu_input1_grad, npu_input1_grad)
            self.assertRtolEqual(cpu_input2_grad, npu_input2_grad)


if __name__ == "__main__":
    run_tests()

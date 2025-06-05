import unittest
import numpy as np
import torch

import torch_npu
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestLogit(TestCase):
    def cpu_op_exec(self, input1, eps=None):
        output = torch.logit(input1, eps)
        output = output.numpy()
        return output
    
    def cpu_backward_op_exec(self, input1, eps=None):
        input1.requires_grad_(True)
        output = torch.logit(input1, eps)
        output.backward(torch.ones_like(input1))
        return input1.grad.numpy()

    def npu_op_exec(self, input1, eps=None):
        output = torch.logit(input1, eps)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_backward_op_exec(self, input1, eps=None):
        input1.requires_grad_(True)
        output = torch.logit(input1, eps)
        output.backward(torch.ones_like(input1))
        return input1.grad.cpu().numpy()

    def npu_op_exec_out(self, input1, out, eps=None):
        torch.logit(input1, eps, out=out)
        output = out.to("cpu")
        output = output.numpy()
        return output

    def test_logit_common_shape_format(self):
        shape_format = [
            [[np.float32, 0, [3, 4]], 1e-5],
            [[np.float32, 0, [3, 128, 256]], 3e-5],
            [[np.float32, 0, [3, 256, 128, 8]], None],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0.1, 0.9)
            cpu_out, npu_out = create_common_tensor(item[0], 0.1, 0.9)
            eps = item[1]
            if eps is None:
                cpu_output = self.cpu_op_exec(cpu_input1)
                npu_output = self.npu_op_exec(npu_input1)
                npu_output_out = self.npu_op_exec_out(npu_input1, npu_out)
                npu_output_inplace = npu_input1.logit_().cpu().numpy()
            else:
                cpu_output = self.cpu_op_exec(cpu_input1, eps)
                npu_output = self.npu_op_exec(npu_input1, eps)
                npu_output_out = self.npu_op_exec_out(npu_input1, npu_out, eps)
                npu_output_inplace = npu_input1.logit_(eps).cpu().numpy()

            self.assertEqual(cpu_output, npu_output)
            self.assertEqual(cpu_output, npu_output_out)
            self.assertEqual(cpu_output, npu_output_inplace)


    def test_logit_backward(self):
        shape_format = [
            [[np.float32, 0, [3, 4]], 1e-5],
            [[np.float32, 0, [3, 128, 256]], 3e-5],
            [[np.float32, 0, [3, 256, 128, 8]], None],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0.1, 0.9)
            cpu_out, npu_out = create_common_tensor(item[0], 0.1, 0.9)
            eps = item[1]
            if eps is None:
                cpu_output = self.cpu_op_exec(cpu_input1)
                npu_output = self.npu_op_exec(npu_input1)
            else:
                cpu_output = self.cpu_op_exec(cpu_input1, eps)
                npu_output = self.npu_op_exec(npu_input1, eps)

            self.assertEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()

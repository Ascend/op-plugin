import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor

input_grad = None
npu_input_grad = None


def input_grad_hook(grad):
    global input_grad
    input_grad = grad
    input_grad = input_grad.numpy()


def npu_input_grad_hook(grad):
    global npu_input_grad
    npu_input_grad = grad.to("cpu")
    npu_input_grad = npu_input_grad.numpy()


class TestSoftmaxBackward(TestCase):
    def cpu_op_exec(self, input1, dim):
        input1.requires_grad = True
        input1.register_hook(input_grad_hook)
        output = torch.nn.functional.softmax(input1, dim)
        z = output.sum()
        z.backward()

    def npu_op_exec(self, input1, dim):
        input1.requires_grad = True
        input1.register_hook(npu_input_grad_hook)
        output = torch.nn.functional.softmax(input1, dim)
        z = output.sum()
        z.backward()
        input1 = input1.cpu()

    def npu_op_exec_half_float(self, input1, dim):
        input1.requires_grad = True
        input1.register_hook(npu_input_grad_hook)
        output = torch._softmax(input1, dim, True)
        z = output.sum()
        z.backward()
        input1 = input1.cpu()

    def test_softmax_backward_shape_format_fp16(self):
        format_list = [0]
        shape_list = [5, (64, 10), (32, 3, 3), (256, 2048, 7, 7)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            input1, npu_input1 = create_common_tensor(item, 1, 100)
            input1 = input1.to(torch.float32)
            self.cpu_op_exec(input1, 0)
            self.npu_op_exec_half_float(npu_input1, 0)
            global input_grad
            input_grad = input_grad.astype(npu_input_grad.dtype)
            self.assertRtolEqual(input_grad, npu_input_grad)

    def test_softmax_backward_shape_format_fp32(self):
        format_list = [0, 3, 4, 29]
        shape_list = [(256, 2048, 7, 7)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            input1, npu_input1 = create_common_tensor(item, 1, 100)
            self.cpu_op_exec(input1, 0)
            self.npu_op_exec(npu_input1, 0)
            self.assertRtolEqual(input_grad, npu_input_grad)


if __name__ == "__main__":
    run_tests()

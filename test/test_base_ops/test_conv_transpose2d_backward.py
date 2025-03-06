import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestConvTranspose2dBackward(TestCase):

    def cpu_op_exec(self, input1, weight, bias):
        input1.requires_grad = True
        weight.requires_grad = True
        bias.requires_grad = True

        res_forward = nn.functional.conv_transpose2d(input1, weight, padding=1, bias=bias)
        grads = torch.ones_like(res_forward)
        res_forward.backward(grads, retain_graph=True)
        input_grad = input1.grad
        weight_grad = weight.grad
        return res_forward, input_grad, weight_grad

    def npu_op_exec(self, input1, weight, bias):
        input1.requires_grad = True
        weight.requires_grad = True
        bias.requires_grad = True

        res_forward = nn.functional.conv_transpose2d(input1, weight, padding=1, bias=bias)
        grads = torch.ones_like(res_forward).npu()
        res_forward.backward(grads, retain_graph=True)
        res_forward = res_forward.cpu()
        input_grad = input1.grad.cpu()
        weight_grad = weight.grad.cpu()
        return res_forward, input_grad, weight_grad

    def conv_transpose2d_backward_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -2, 2)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -2, 2)
            cpu_bias, npu_bias = create_common_tensor((item[0][0], 0, 4), -100, 100)
            if item[0][0] == np.float16:
                cpu_input1 = cpu_input1.float()
                cpu_input2 = cpu_input2.float()
                cpu_bias = cpu_bias.float()
            else:
                # Synchronous accuracy loss: The operator converts fp32 to fp16 for calculation.
                cpu_input1 = cpu_input1.half().float()
                cpu_input2 = cpu_input2.half().float()
                cpu_bias = cpu_bias.half().float()
            cpu_output, cpu_input_grad, cpu_weight_grad = self.cpu_op_exec(cpu_input1, cpu_input2, bias=cpu_bias)
            npu_output, npu_input_grad, npu_weight_grad = self.npu_op_exec(npu_input1, npu_input2, bias=npu_bias)
            if item[0][0] == np.float16:
                cpu_output = cpu_output.half()
                cpu_input_grad = cpu_input_grad.half()
                cpu_weight_grad = cpu_weight_grad.half()

            self.assertRtolEqual(cpu_output.detach(), npu_output.detach(), prec=1e-3)
            self.assertRtolEqual(cpu_input_grad, npu_input_grad, prec=1e-3)
            self.assertRtolEqual(cpu_weight_grad, npu_weight_grad, prec=1e-3)

    def test_conv_transpose2d_backward_shape_format(self):
        shape_format = [
            [[np.float16, 0, [1, 4, 5, 5]], [np.float16, 0, [4, 4, 3, 3]]],
            [[np.float32, 0, [1, 4, 5, 5]], [np.float32, 0, [4, 4, 3, 3]]]
        ]
        self.conv_transpose2d_backward_result(shape_format)

    def test_conv_transpose2d_backward_allow_hf32(self):
        torch.npu.conv.allow_hf32 = True
        shape_format = [
            [[np.float16, 0, [1, 4, 5, 5]], [np.float16, 0, [4, 4, 3, 3]]]
        ]
        self.conv_transpose2d_backward_result(shape_format)
        torch.npu.conv.allow_hf32 = False

    def test_conv_transpose2d_abnormal_input(self):
        input1 = torch.randn(1, 320, 8, 8).npu().half()
        m = torch.nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=2, stride=2).npu()
        with self.assertRaises(RuntimeError) as cm:
            output1 = m(input1)
        exception = cm.exception
        self.assertTrue("Input type (npuHalfType) and weight type (npuFloatType) should be the same" in str(exception))

        inputs = torch.randn(1, 4, 5, 5).npu()
        weights = torch.randn(4, 8, 3, 3).npu()
        bias = torch.randn(8).npu().half()
        with self.assertRaises(RuntimeError) as cm:
            output2 = F.conv_transpose2d(inputs, weights, padding=1, bias=bias)
        exception = cm.exception
        self.assertTrue("Input type (npuFloatType) and bias type (npuHalfType) should be the same" in str(exception))

    def test_conv_transpose2d_3D_input(self):
        torch.npu.config.allow_internal_format = True
        device = torch.device('npu') 
        cpu_input = torch.randn(1, 640, 480)
        npu_input = cpu_input.to(device)

        npu_input = torch.nn.AvgPool2d(kernel_size=4, stride=4).to(device)(npu_input)
        npu_input = torch.nn.Sigmoid().to(device)((npu_input + (torch.ones_like(npu_input).to(device) * (- 0.5))))

        with self.assertRaises(RuntimeError) as cm:
            npu_output = torch.nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=2, stride=2).to(device)(npu_input)
        exception = cm.exception
        self.assertTrue("Currently the private format does not support 3D input, you can try torch.npu.config.allow_internal_format = False to resolve this functional bug" in str(exception))


if __name__ == "__main__":
    np.random.seed(1234)
    run_tests()

import unittest
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices


class TestUpsampleBicubic2dAABackward(TestCase):

    def cpu_op_exec(self, input1, output_size, align_corners, scale_h, scale_w):
        input1.requires_grad_(True)
        output = torch._C._nn._upsample_bicubic2d_aa(input1, output_size, align_corners, scale_h, scale_w)
        grad_output = torch.ones_like(output)
        output.backward(grad_output)
        output = torch.Tensor(input1.grad).numpy()
        return output

    def npu_op_exec(self, input1, output_size, align_corners, scale_h, scale_w):
        input1.requires_grad_(True)
        output = torch._C._nn._upsample_bicubic2d_aa(input1, output_size, align_corners, scale_h, scale_w)
        grad_output = torch.ones_like(output)
        output.backward(grad_output)
        output = torch.Tensor(input1.grad)
        output = output.to("cpu")
        output = output.numpy()
        return output

    @SupportedDevices(['Ascend910B'])
    def test_upsample_bicubic2d_AA_Grad_common_shape_format(self):
        shape_format = [
            [[np.float32, -1, (1, 1, 4, 4)], (16, 16), True, 0, 0, 0, 255],
            [[np.float32, -1, (1, 1, 2, 2)], (4, 4), True, 0, 0, 0, 255],
            [[np.float32, -1, (2, 2, 4, 4)], (2, 2), True, 0, 0, 0, 255],
            [[np.float32, -1, (2, 2, 2, 2)], (10, 10), True, 0, 0, 0, 255],
            [[np.float32, -1, (1, 1, 2, 2)], (4, 4), False, 0.5, 0.5, 0, 255],
            [[np.float32, -1, (2, 2, 2, 2)], (4, 4), False, 0.5, 0.5, 0, 255],
            [[np.float32, -1, (2, 2, 2, 2)], (10, 10), False, 0.5, 0.5, 0, 255],
            [[np.float32, -1, (1, 1152, 27, 27)], (32, 32), False, 0, 0, 0, 255],
            [[np.float32, -1, (2, 1, 32, 32)], (128, 128), True, 0, 0, 0, 255],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], item[5], item[6])
            cpu_output = self.cpu_op_exec(cpu_input1, item[1], item[2], item[3], item[4])
            npu_output = self.npu_op_exec(npu_input1, item[1], item[2], item[3], item[4])
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
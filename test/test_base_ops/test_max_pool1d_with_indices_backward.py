import torch
import numpy as np
import torch.nn.functional as F
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
import unittest

# enable max_pool1d_withindices
torch_npu.npu.use_compatible_impl(True)
DEVICE_NAME = torch_npu.npu.get_device_name(0)
device_is_910A = False
if "Ascend910A" in DEVICE_NAME or "Ascend910P" in DEVICE_NAME:
    device_is_910A = True


class TestMaxPool1dWithIndicesBackward(TestCase):
    # pylint:disable = huawei-too-many-arguments
    def cpu_op_exec(self, input_cpu, kernel_size, stride, padding, dilation, ceil_mode):
        input_cpu.requires_grad_(True)
        output_cpu, indices_cpu = F.max_pool1d(
            input_cpu,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            return_indices=True
        )
        output_cpu.backward(torch.ones_like(output_cpu))
        grad_cpu = input_cpu.grad
        return output_cpu.detach(), indices_cpu.detach(), grad_cpu

    # pylint:disable = huawei-too-many-arguments
    def npu_op_exec(self, input_npu, kernel_size, stride, padding, dilation, ceil_mode):
        input_npu.requires_grad_(True)
        output_npu, indices_npu = F.max_pool1d(
            input_npu,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            return_indices=True
        )
        output_npu.backward(torch.ones_like(output_npu))
        grad_npu = input_npu.grad
        return output_npu.detach().cpu(), indices_npu.detach().cpu(), grad_npu.cpu()

    @unittest.skipIf(device_is_910A, "aclnnMaxPool2dWithIndices is not supported on 910A")
    def test_max_pool1d_with_indices_backward(self):
        shape_format = [
            [np.float32, 0, [1, 2, 8], 3, 2, 0, 1, False],
            [np.float32, 0, [2, 3, 11], 5, 2, 2, 1, False],
            [np.float32, 0, [4, 5, 16], 3, 2, 1, 1, True],
            [np.float32, 0, [2, 8, 33], 5, 2, 1, 1, False],
            [np.float32, 0, [4, 16, 65], 3, 2, 1, 1, True],
            [np.float32, 0, [1, 32, 127], 7, 3, 2, 1, False],
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[:3], -50, 50)
            cpu_output, cpu_indices, cpu_grad = self.cpu_op_exec(cpu_input, item[3], item[4], item[5], item[6], item[7])
            npu_output, npu_indices, npu_grad = self.npu_op_exec(npu_input, item[3], item[4], item[5], item[6], item[7])
            cpu_output = cpu_output.to(npu_output.dtype)
            cpu_grad = cpu_grad.to(npu_grad.dtype)
            self.assertRtolEqual(cpu_output.numpy(), npu_output.numpy())
            self.assertEqual(cpu_indices.to(torch.int64).numpy(), npu_indices.to(torch.int64).numpy())
            self.assertRtolEqual(cpu_grad.numpy(), npu_grad.numpy())


if __name__ == "__main__":
    run_tests()

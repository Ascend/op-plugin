import torch
import numpy as np
import torch.nn.functional as F
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
import unittest
# enable max_pool2d_withindices
torch_npu.npu.use_compatible_impl(True)
DEVICE_NAME = torch_npu.npu.get_device_name(0)
device_is_910A = False
if "Ascend910A" in DEVICE_NAME or "Ascend910P" in DEVICE_NAME:
    device_is_910A = True

class TestMaxPool2dWithIndices(TestCase):
    # pylint:disable = huawei-too-many-arguments
    def cpu_op_exec(self, input_cpu, kernel_size, stride, padding, dilation, ceil_mode):
        output_cpu, indices_cpu = F.max_pool2d(
            input_cpu,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            return_indices=True
        )
        return output_cpu, indices_cpu

    # pylint:disable = huawei-too-many-arguments
    def npu_op_exec(self, input_npu, kernel_size, stride, padding, dilation, ceil_mode):
        output_npu, indices_npu = F.max_pool2d(
            input_npu,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            return_indices=True
        )
        output_npu = output_npu.to("cpu").detach()
        indices_npu = indices_npu.to("cpu").detach()
        return output_npu, indices_npu

    @unittest.skipIf(device_is_910A, "aclnnMaxPool2dWithIndices is not supported on 910A")
    def test_max_pool2d_with_indices(self):
        shape_format_basic = [
            [np.float32, 0, [1, 1, 3, 3], (2, 2), (1, 1), (0, 0), (1, 1), False],
            [np.float32, 0, [1, 2, 5, 5], (3, 3), (2, 2), (0, 0), (1, 1), True],
            [np.float32, 0, [2, 3, 7, 7], (2, 2), (1, 1), (0, 0), (1, 1), False],
            [np.float32, 0, [2, 3, 9, 9], (3, 3), (2, 2), (0, 0), (1, 1), False],
        ]

        shape_format_padding = [
            [np.float32, 0, [1, 2, 5, 5], (3, 3), (1, 1), (1, 1), (1, 1), False],
            [np.float32, 0, [1, 2, 5, 5], (3, 3), (2, 2), (1, 1), (1, 1), True],
            [np.float32, 0, [2, 3, 8, 8], (5, 5), (2, 2), (2, 2), (1, 1), False],
            [np.float32, 0, [2, 4, 9, 9], (3, 3), (1, 1), (1, 1), (1, 1), True],
        ]

        shape_format_fp16 = [
            [np.float16, 0, [2, 4, 8, 8], (2, 2), (2, 2), (0, 0), (1, 1), False],
            [np.float16, 0, [2, 4, 10, 10], (3, 3), (1, 1), (1, 1), (1, 1), False],
            [np.float16, 0, [3, 6, 12, 12], (3, 3), (2, 2), (0, 0), (1, 1), True],
            [np.float16, 0, [4, 8, 16, 16], (2, 2), (1, 1), (0, 0), (1, 1), False],
            [np.float16, 0, [4, 8, 16, 16], (4, 4), (2, 2), (1, 1), (1, 1), True],
        ]

        shape_format_large = [
            [np.float32, 0, [4, 5, 16, 16], (3, 3), (2, 2), (1, 1), (1, 1), False],
            [np.float32, 0, [8, 4, 17, 17], (3, 3), (2, 2), (0, 0), (1, 1), True],
            [np.float32, 0, [16, 8, 32, 32], (3, 3), (2, 2), (1, 1), (1, 1), False],
            [np.float32, 0, [16, 16, 64, 64], (4, 4), (4, 4), (0, 0), (1, 1), False],
            [np.float16, 0, [16, 16, 64, 64], (3, 3), (2, 2), (1, 1), (1, 1), True],
        ]

        shape_format = (
            shape_format_basic
            + shape_format_padding
            + shape_format_fp16
            + shape_format_large
        )

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[:3], -50, 50)
            cpu_output, cpu_indices = self.cpu_op_exec(cpu_input, item[3], item[4], item[5], item[6], item[7])
            npu_output, npu_indices = self.npu_op_exec(npu_input, item[3], item[4], item[5], item[6], item[7])
            self.assertRtolEqual(cpu_output.numpy(), npu_output.numpy())
            # Currently the indices returned by the max_pool2d function are of type int32.
            self.assertEqual(cpu_indices.to(torch.int64).numpy(), npu_indices.to(torch.int64).numpy())

if __name__ == "__main__":
    run_tests()
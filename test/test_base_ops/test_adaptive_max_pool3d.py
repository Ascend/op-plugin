import torch
import torch.nn.functional as F
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.common_utils import SupportedDevices


class TestAdaptiveMaxPool3d(TestCase):
    def cpu_op_exec(self, input1, output_size):
        if input1.dtype == torch.float16:
            input1 = input1.float()
            output = F.adaptive_max_pool3d(input1, output_size)
            output = output.half()
        else:
            output = F.adaptive_max_pool3d(input1, output_size)
        return output.numpy()

    def npu_op_exec(self, input1, output_size):
        output = F.adaptive_max_pool3d(input1, output_size)
        return output.cpu().numpy()

    @SupportedDevices(['Ascend910B'])
    def test_adaptive_max_pool3d_shape_format_fp16(self):
        shape_format = [
            [np.float16, -1, (4, 16, 8, 4, 2)],
            [np.float16, -1, (2, 16, 4, 8, 8)],
            [np.float16, -1, (4, 16, 8, 4, 16)],
            [np.float16, -1, (1, 3, 6, 6, 6)],
            [np.float16, -1, (8, 4, 12, 10, 8)],
        ]
        output_list = [(1, 1, 1), (2, 2, 2), (3, 3, 3), (1, 2, 3)]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 1, 10)
            for output_size in output_list:
                cpu_output = self.cpu_op_exec(cpu_input, output_size)
                npu_output = self.npu_op_exec(npu_input, output_size)
                self.assertEqual(cpu_output.dtype, npu_output.dtype)
                self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(['Ascend910B'])
    def test_adaptive_max_pool3d_shape_format_fp32(self):
        shape_format = [
            [np.float32, -1, (4, 2, 8, 4, 8)],
            [np.float32, -1, (2, 16, 4, 8, 8)],
            [np.float32, -1, (4, 16, 8, 4, 16)],
            [np.float32, -1, (1, 3, 6, 6, 6)],
            [np.float32, -1, (8, 4, 12, 10, 8)],
        ]
        output_list = [(1, 1, 1), (2, 2, 2), (3, 3, 3), (1, 2, 3)]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 1, 10)
            for output_size in output_list:
                cpu_output = self.cpu_op_exec(cpu_input, output_size)
                npu_output = self.npu_op_exec(npu_input, output_size)
                self.assertEqual(cpu_output.dtype, npu_output.dtype)
                self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(['Ascend910B'])
    def test_adaptive_max_pool3d_shape_format_fp64(self):
        shape_format = [
            [np.float64, -1, (4, 2, 8, 4, 8)],
            [np.float64, -1, (2, 16, 4, 8, 8)],
            [np.float64, -1, (4, 16, 8, 4, 16)],
            [np.float64, -1, (1, 3, 6, 6, 6)],
            [np.float64, -1, (8, 4, 12, 10, 8)],
        ]
        output_list = [(1, 1, 1), (2, 2, 2), (3, 3, 3), (1, 2, 3)]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 1, 10)
            for output_size in output_list:
                cpu_output = self.cpu_op_exec(cpu_input, output_size)
                npu_output = self.npu_op_exec(npu_input, output_size)
                self.assertEqual(cpu_output.dtype, npu_output.dtype)
                self.assertRtolEqual(cpu_output, npu_output)

if __name__ == "__main__":
    run_tests()

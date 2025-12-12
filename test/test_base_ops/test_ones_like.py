import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestOnesLike(TestCase):

    def cpu_op_exec(self, input1):
        output = torch.ones_like(input1)
        return output.numpy(), output.dtype

    def npu_op_exec(self, input1):
        output = torch.ones_like(input1)
        output_cpu = output.to('cpu')
        return output_cpu.numpy(), output_cpu.dtype

    def test_ones_like_shape_format(self):
        shape_format = [
            [np.float32, -1, (3, )],
            [np.float32, -1, (2, 4)],
            [np.float32, -1, (3, 6, 9)],
            [np.int8, -1, (3,)],
            [np.int8, -1, (2, 4)],
            [np.int32, -1, (3, 6, 9)],
            [np.uint8, -1, (3,)],
            [np.uint8, -1, (2, 4, 5)],
            [np.int64, -1, (1,)],
            [np.int64, -1, (2, 4)],
            [np.int64, -1, (3, 6, 9)],
            [np.int64, -1, (2, 3, 4, 5)],
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 0)

            cpu_output, cpu_dtype = self.cpu_op_exec(cpu_input)
            npu_output, npu_dtype = self.npu_op_exec(npu_input)

            self.assertEqual(cpu_dtype, npu_dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_ones_like_float16_shape_format(self):
        shape_format = [
            [np.float16, -1, (3, )],
            [np.float16, -1, (2, 4)],
            [np.float16, -1, (3, 6, 9)],
            [np.float16, -1, (3, 4, 5, 12)]
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 0)
            cpu_input_float32 = cpu_input.to(torch.float32)
            cpu_output_float32, _ = self.cpu_op_exec(cpu_input_float32)
            cpu_output = cpu_output_float32.astype(np.float16)
            cpu_dtype = torch.float16

            npu_output, npu_dtype = self.npu_op_exec(npu_input)
            self.assertEqual(cpu_dtype, npu_dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_ones_like_special_cases(self):
        special_cases = [
            [np.int64, -1, (0,)],
            [np.int64, -1, (1,)],
            [np.int32, -1, (1,)],
            [np.float64, -1, (1,)],
            [np.int64, -1, (2, 3, 4, 5, 6)],
        ]

        for item in special_cases:
            cpu_input, npu_input = create_common_tensor(item, 0, 0)
            cpu_output, cpu_dtype = self.cpu_op_exec(cpu_input)
            npu_output, npu_dtype = self.npu_op_exec(npu_input)
            self.assertEqual(cpu_dtype, npu_dtype)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()

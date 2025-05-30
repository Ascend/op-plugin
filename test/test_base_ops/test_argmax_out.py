import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestArgmax(TestCase):
    def cpu_op_exec(self, input1):
        output = torch.randint(0, 100, (), dtype=torch.int64)
        output = torch.argmax(input1, out=output)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        output = torch.randint(0, 100, (), dtype=torch.int64).npu()
        output = torch.argmax(input1, out=output)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_argmax_out_shape_format_fp16(self):
        format_list = [0]
        shape_list = [[5], [2, 4], [2, 2, 4], [2, 3, 3, 4]]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -10, 10)
            cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_argmax_out_shape_format_fp32(self):
        format_list = [0]
        shape_list = [[5], [2, 4], [2, 2, 4], [2, 3, 3, 4]]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -10, 10)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_argmax_zero_size_dim(self):
        tensor = torch.randn(2, 0, 3)
        dim = 1
        with self.assertRaises(IndexError) as cm:
            torch.argmax(tensor, dim=dim)
        exception = cm.exception
        expected_error_msg = f"argmax(): Expected reduction dim {dim} to have non-zero size."
        self.assertTrue(expected_error_msg in str(exception))

    def test_argmax_no_dim_empty_tensor(self):
        empty_tensor = torch.empty(0)
        with self.assertRaises(IndexError) as cm:
            torch.argmax(empty_tensor)
        exception = cm.exception
        expected_error_msg = "argmax(): Expected reduction dim to be specified for input.numel() == 0."
        self.assertTrue(expected_error_msg in str(exception))


if __name__ == "__main__":
    run_tests()
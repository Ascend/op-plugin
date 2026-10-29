import torch
import torch.nn as nn
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestHardtanh(TestCase):
    def cpu_op_backward_exec(self, input1, min1, max1):
        w = torch.ones_like(input1)
        input1.requires_grad_(True)
        output = torch.nn.functional.hardtanh(input1, min1, max1)
        output.backward(w)
        output = output.detach().numpy()
        res = input1.grad
        res = res.numpy()
        return output, res

    def npu_op_backward_exec(self, input1, min1, max1):
        w = torch.ones_like(input1)
        w = w.to("npu")
        input1.requires_grad_(True)
        output = torch.nn.functional.hardtanh(input1, min1, max1)
        output.backward(w)
        output = output.to("cpu").detach().numpy()
        res = input1.grad
        res = res.to("cpu").numpy()
        return output, res

    def hardtanh_result(self, shape_format):
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 2)
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)
            cpu_output_forward, cpu_output_backward = self.cpu_op_backward_exec(cpu_input, 0, 1)
            npu_output_forward, npu_output_backward = self.npu_op_backward_exec(npu_input, 0, 1)
            cpu_output_forward = cpu_output_forward.astype(npu_output_forward.dtype)
            cpu_output_backward = cpu_output_backward.astype(npu_output_backward.dtype)
            self.assertRtolEqual(cpu_output_forward, npu_output_forward)
            self.assertRtolEqual(cpu_output_backward, npu_output_backward)

    def test_hardtanh_shape_format_fp16_1d(self):
        format_list = [0, 3, 4]
        shape_format = [
            [np.float16, i, [18]] for i in format_list
        ]
        self.hardtanh_result(shape_format)

    def test_hardtanh_shape_format_fp32_1d(self):
        format_list = [0, 3, 4]
        shape_format = [
            [np.float32, i, [18]] for i in format_list
        ]
        self.hardtanh_result(shape_format)

    def test_hardtanh_shape_format_fp16_2d(self):
        format_list = [0, 3, 4, 29]
        shape_format = [
            [np.float16, i, [256, 1000]] for i in format_list
        ]
        self.hardtanh_result(shape_format)

    def test_hardtanh_shape_format_fp32_2d(self):
        format_list = [0, 3, 4, 29]
        shape_format = [
            [np.float32, i, [256, 1000]] for i in format_list
        ]
        self.hardtanh_result(shape_format)

    def test_hardtanh_shape_format_fp16_3d(self):
        format_list = [0, 3, 4, 29]
        shape_format = [
            [np.float16, i, [32, 328, 368]] for i in format_list
        ]
        self.hardtanh_result(shape_format)

    def test_hardtanh_shape_format_fp32_3d(self):
        format_list = [0, 3, 4, 29]
        shape_format = [
            [np.float32, i, [32, 328, 368]] for i in format_list
        ]
        self.hardtanh_result(shape_format)

    def test_hardtanh_shape_format_fp16_4d(self):
        format_list = [0, 3, 4, 29]
        shape_format = [
            [np.float16, i, [256, 576, 7, 7]] for i in format_list
        ]
        self.hardtanh_result(shape_format)

    def test_hardtanh_shape_format_fp32_4d(self):
        format_list = [0, 3, 4, 29]
        shape_format = [
            [np.float32, i, [256, 576, 7, 7]] for i in format_list
        ]
        self.hardtanh_result(shape_format)


class TestHardtanhNarrowInt(TestCase):
    """Hardtanh on int8/uint8/int16 must use the vector path (Ascend/pytorch#178)."""

    def _assert_same(self, cpu_out, npu_out):
        self.assertEqual(cpu_out.dtype, npu_out.dtype)
        self.assertTrue(torch.equal(cpu_out, npu_out.cpu()))

    def test_hardtanh_narrow_int(self):
        cases = [
            (torch.int8, -100, 100, -31, 31),
            (torch.uint8, 0, 200, 16, 200),
            (torch.int16, -1000, 1000, -100, 100),
        ]
        for dtype, low, high, min_val, max_val in cases:
            cpu_input = torch.randint(low, high, (1024,), dtype=dtype)
            npu_input = cpu_input.npu()
            expected = torch.nn.functional.hardtanh(cpu_input, min_val, max_val)

            self._assert_same(expected, torch.nn.functional.hardtanh(npu_input, min_val, max_val))

            npu_inplace = npu_input.clone()
            torch.nn.functional.hardtanh_(npu_inplace, min_val, max_val)
            self._assert_same(expected, npu_inplace)


if __name__ == "__main__":
    run_tests()

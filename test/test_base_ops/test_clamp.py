import unittest
import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


@unittest.skip("skip TestClamp now")
class TestClamp(TestCase):

    def npu_op_exec(self, input1, min_val, max_val):
        output = torch.clamp(input1, min_val, max_val)
        output = output.cpu().numpy()
        return output

    def cpu_op_exec(self, input1, min_val, max_val):
        output = torch.clamp(input1, min_val, max_val)
        output = output.numpy()
        return output

    def npu_inp_op_exec(self, input1, min_val, max_val):
        torch.clamp_(input1, min_val, max_val)
        output = input1.cpu().numpy()
        return output

    def cpu_inp_op_exec(self, input1, min_val, max_val):
        output = torch.clamp_(input1, min_val, max_val)
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, min_val, max_val, output):
        torch.clamp(input1, min_val, max_val, out=output)
        output = output.cpu().numpy()
        return output

    def cpu_op_exec_out(self, input1, min_val, max_val, output):
        torch.clamp(input1, min_val, max_val, out=output)
        output = output.numpy()
        return output

    def npu_inp_uncon_op_exec(self, input1, min_val, max_val):
        input1 = input1.as_strided([2, 2], [1, 2], 2)
        torch.clamp_(input1, min_val, max_val)
        output = input1.cpu().numpy()
        return output

    def cpu_inp_uncon_op_exec(self, input1, min_val, max_val):
        input_dtype = input1.dtype
        input1 = input1.as_strided([2, 2], [1, 2], 2)
        output = torch.clamp(input1, min_val, max_val)
        output = output.numpy()
        return output

    def test_clamp_common(self):
        shape_format = [
            [np.float32, 0, (4, 3)],
            [np.int32, 0, (4, 3)],
            [np.int64, 0, (4, 3)],
            [np.float16, 0, (4, 3)]
        ]
        for item in shape_format:
            input_cpu, input_npu = create_common_tensor(item, 1, 100)
            _, out_npu = create_common_tensor(item, 1, 100)

            cpu_output = self.cpu_op_exec(input_cpu, 40, 60)
            npu_output = self.npu_op_exec(input_npu, 40, 60)

            cpu_inp_output = self.cpu_inp_op_exec(input_cpu, 40, 60)
            npu_inp_output = self.npu_inp_op_exec(input_npu, 40, 60)

            npu_out_output = self.npu_op_exec_out(input_npu, 40, 60, out_npu)

            cpu_inp_uncon_output = self.cpu_inp_uncon_op_exec(input_cpu, 40, 60)
            npu_inp_uncon_output = self.npu_inp_uncon_op_exec(input_npu, 40, 60)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_inp_output, npu_inp_output)
            self.assertRtolEqual(cpu_output, npu_out_output)
            self.assertRtolEqual(cpu_inp_uncon_output, npu_inp_uncon_output)

    def test_clamp_tensor(self):
        shape_format = [
            [[np.float32, 0, (4, 3)], [np.float32, 0, (4, 3)], [np.float32, 0, (4, 3)]],
            [[np.int32, 0, (24, 13)], [np.int32, 0, (24, 1)], [np.int32, 0, (1, 13)]],
            [[np.int64, 0, (41, 32, 23)], [np.int32, 0, (41, 32, 23)], [np.int32, 0, (41, 32, 23)]],
            [[np.float16, 0, (14, 3)], [np.float32, 0, (14, 3)], [np.float32, 0, (14, 3)]],
            [[np.int32, 0, (14, 3)], [np.float32, 0, (14, 3)], [np.float32, 0, (14, 3)]],
        ]
        for item in shape_format:
            input_cpu, input_npu = create_common_tensor(item[0], 1, 100)
            min_cpu, min_npu = create_common_tensor(item[1], 1, 50)
            max_cpu, max_npu = create_common_tensor(item[2], 50, 100)
            out_cpu, out_npu = create_common_tensor(item[0], 1, 100)

            cpu_output = self.cpu_op_exec(input_cpu, min_cpu, max_cpu)
            npu_output = self.npu_op_exec(input_npu, min_npu, max_npu)
            self.assertRtolEqual(cpu_output, npu_output)

            if torch.can_cast(min_npu.dtype, input_npu.dtype):
                cpu_inp_output = self.cpu_inp_op_exec(input_cpu, min_cpu, max_cpu)
                npu_inp_output = self.npu_inp_op_exec(input_npu, min_npu, max_npu)
                self.assertRtolEqual(cpu_inp_output, npu_inp_output)

                cpu_out_output = self.cpu_op_exec_out(input_cpu, min_cpu, max_cpu, out_cpu)
                npu_out_output = self.npu_op_exec_out(input_npu, min_npu, max_npu, out_npu)
                self.assertRtolEqual(cpu_out_output, npu_out_output)
            else:
                with self.assertRaises(RuntimeError) as cpu_err:
                    self.cpu_inp_op_exec(input_cpu, min_cpu, max_cpu)
                self.assertTrue("can't be cast to the desired output" in str(cpu_err.exception))
                with self.assertRaises(RuntimeError) as npu_err:
                    self.npu_inp_op_exec(input_npu, min_npu, max_npu)
                self.assertTrue("can't be cast to the desired output" in str(npu_err.exception))
                with self.assertRaises(RuntimeError) as cpu_err:
                    self.cpu_op_exec_out(input_cpu, min_cpu, max_cpu, out_cpu)
                self.assertTrue("can't be cast to the desired output" in str(cpu_err.exception))
                with self.assertRaises(RuntimeError) as npu_err:
                    self.npu_op_exec_out(input_npu, min_npu, max_npu, out_npu)
                self.assertTrue("can't be cast to the desired output" in str(npu_err.exception))

    def test_clamp_tensor_empty_broadcast(self):
        # Regression: clamp_npu_output_size must not short-circuit empty self.
        # Broadcast with min/max must still apply (expand dims / reject invalid).
        # case 1: self.ndim < min.ndim, broadcast should expand dims -> (3, 0)
        input_cpu = torch.tensor([], dtype=torch.float32).reshape(1, 0)
        input_npu = input_cpu.npu()
        min_cpu = torch.tensor([[0.0]] * 3, dtype=torch.float32)
        min_npu = min_cpu.npu()
        max_cpu = torch.tensor([[1.0]] * 3, dtype=torch.float32)
        max_npu = max_cpu.npu()
        cpu_output = self.cpu_op_exec(input_cpu, min_cpu, max_cpu)
        npu_output = self.npu_op_exec(input_npu, min_npu, max_npu)
        self.assertEqual(tuple(cpu_output.shape), (3, 0))
        self.assertEqual(tuple(npu_output.shape), (3, 0))
        self.assertEqual(npu_output.size, 0)

        # case 2: self.ndim == min.ndim, same dim needs expand -> (3, 0)
        input_cpu = torch.tensor([], dtype=torch.float32).reshape(1, 0)
        input_npu = input_cpu.npu()
        min_cpu = torch.arange(3, dtype=torch.float32).reshape(3, 1)
        min_npu = min_cpu.npu()
        max_cpu = torch.full((3, 1), 1.0, dtype=torch.float32)
        max_npu = max_cpu.npu()
        cpu_output = self.cpu_op_exec(input_cpu, min_cpu, max_cpu)
        npu_output = self.npu_op_exec(input_npu, min_npu, max_npu)
        self.assertEqual(tuple(cpu_output.shape), (3, 0))
        self.assertEqual(tuple(npu_output.shape), (3, 0))

        # case 3: non-broadcastable empty input should raise (not silently pass)
        input_cpu = torch.tensor([], dtype=torch.float32)
        input_npu = input_cpu.npu()
        min_cpu = torch.arange(3, dtype=torch.float32)
        min_npu = min_cpu.npu()
        max_cpu = torch.full((3,), 1.0, dtype=torch.float32)
        max_npu = max_cpu.npu()
        with self.assertRaises(RuntimeError):
            self.cpu_op_exec(input_cpu, min_cpu, max_cpu)
        with self.assertRaises(RuntimeError):
            self.npu_op_exec(input_npu, min_npu, max_npu)

        # case 4: scalar min/max with empty self -> (0,) (unchanged, sanity)
        input_cpu = torch.tensor([], dtype=torch.float32)
        input_npu = input_cpu.npu()
        min_cpu = torch.tensor(0.0, dtype=torch.float32)
        min_npu = min_cpu.npu()
        max_cpu = torch.tensor(1.0, dtype=torch.float32)
        max_npu = max_cpu.npu()
        cpu_output = self.cpu_op_exec(input_cpu, min_cpu, max_cpu)
        npu_output = self.npu_op_exec(input_npu, min_npu, max_npu)
        self.assertEqual(tuple(cpu_output.shape), (0,))
        self.assertEqual(tuple(npu_output.shape), (0,))


class TestClampNarrowInt(TestCase):
    """Narrow integer clamp must use the vector path (Ascend/pytorch#178)."""

    def _make_input(self, dtype, shape):
        low, high = (0, 200) if dtype == torch.uint8 else (-100, 100)
        return torch.randint(low, high, shape, dtype=dtype)

    def _assert_same(self, cpu_out, npu_out):
        self.assertEqual(cpu_out.dtype, npu_out.dtype)
        self.assertTrue(torch.equal(cpu_out, npu_out.cpu()))

    def _check_clamp(self, dtype, min_val, max_val):
        for shape in [(1024,), (8, 16, 32)]:
            cpu_input = self._make_input(dtype, shape)
            npu_input = cpu_input.npu()
            expected = torch.clamp(cpu_input, min_val, max_val)

            self._assert_same(expected, torch.clamp(npu_input, min_val, max_val))

            npu_inplace = npu_input.clone()
            npu_inplace.clamp_(min_val, max_val)
            self._assert_same(expected, npu_inplace)

            out = torch.empty_like(npu_input)
            torch.clamp(npu_input, min_val, max_val, out=out)
            self._assert_same(expected, out)

    def test_clamp_int8(self):
        self._check_clamp(torch.int8, -31, 31)
        self._check_clamp(torch.int8, -128, 127)

    def test_clamp_uint8(self):
        self._check_clamp(torch.uint8, 16, 200)
        self._check_clamp(torch.uint8, 0, 255)

    def test_clamp_int16(self):
        self._check_clamp(torch.int16, -1000, 1000)
        self._check_clamp(torch.int16, -32768, 32767)

    def test_clamp_min_max_narrow_int(self):
        for dtype, min_val, max_val in [
            (torch.int8, -31, 31),
            (torch.uint8, 16, 200),
            (torch.int16, -1000, 1000),
        ]:
            cpu_input = self._make_input(dtype, (1024,))
            npu_input = cpu_input.npu()

            self._assert_same(torch.clamp_min(cpu_input, min_val), torch.clamp_min(npu_input, min_val))
            self._assert_same(torch.clamp_max(cpu_input, max_val), torch.clamp_max(npu_input, max_val))

            npu_inplace = npu_input.clone()
            npu_inplace.clamp_min_(min_val)
            self._assert_same(torch.clamp_min(cpu_input, min_val), npu_inplace)

            npu_inplace = npu_input.clone()
            npu_inplace.clamp_max_(max_val)
            self._assert_same(torch.clamp_max(cpu_input, max_val), npu_inplace)

    def test_clamp_int8_float_bounds_promote(self):
        cpu_input = torch.tensor([-5, 0, 5], dtype=torch.int8)
        npu_input = cpu_input.npu()
        for min_val, max_val in [(-31.5, 31.5), (-31.0, 31.0)]:
            self._assert_same(torch.clamp(cpu_input, min_val, max_val),
                              torch.clamp(npu_input, min_val, max_val))

    def test_clamp_uint8_negative_bound_fallback(self):
        cpu_input = torch.tensor([0, 1, 5, 31, 32, 200, 255], dtype=torch.uint8)
        npu_input = cpu_input.npu()
        self._assert_same(torch.clamp(cpu_input, -31, 31), torch.clamp(npu_input, -31, 31))

    def test_clamp_int8_non_contiguous(self):
        cpu_input = torch.randint(-100, 100, (64, 64), dtype=torch.int8).t()
        npu_input = cpu_input.npu()

        self._assert_same(torch.clamp(cpu_input, -31, 31), torch.clamp(npu_input, -31, 31))

        cpu_inplace = cpu_input.clone()
        cpu_inplace.clamp_(-31, 31)
        npu_input.clamp_(-31, 31)
        self._assert_same(cpu_inplace, npu_input)


if __name__ == "__main__":
    run_tests()

import warnings

import torch
import numpy as np
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestSearchsorted(TestCase):
    def cpu_sorted_input(self, input1):
        input_dim = input1.dim() - 1
        input_op, _ = input1.float().sort(input_dim)
        input_op = input_op.to(input1.dtype)
        return input_op

    def cpu_op_exec(self, input1, input2):
        output = torch.searchsorted(input1, input2)
        output = output.numpy()
        return output

    def cpu_op_exec_bool(self, input1, input2, out_int32, right):
        output = torch.searchsorted(input1, input2, out_int32=out_int32, right=right)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        output = torch.searchsorted(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_bool(self, input1, input2, out_int32, right):
        output = torch.searchsorted(input1, input2, out_int32=out_int32, right=right)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, input2, out):
        torch.searchsorted(input1, input2, out=out)
        output = out.to("cpu")
        output = output.numpy()
        return output

    def test_searchsorted_tensor_shape_format(self):
        shape_format = [
            [[np.int32, 0, [256, 40]], [np.int32, 0, [256, 20]]],
            [[np.int64, 0, [256, 40]], [np.int64, 0, [256, 20]]],
            [[np.float32, 0, [256, 40]], [np.float32, 0, [256, 20]]],
            [[np.int32, 0, [4, 12, 12, 128]], [np.int32, 0, [4, 12, 12, 23]]],
            [[np.int64, 0, [4, 12, 12, 128]], [np.int64, 0, [4, 12, 12, 23]]],
            [[np.float32, 0, [4, 12, 12, 128]], [np.float32, 0, [4, 12, 12, 23]]],
        ]

        for item in shape_format:
            cpu_input1, _ = create_common_tensor(item[0], -10, 10)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -10, 10)
            _, npu_out = create_common_tensor(item[1], -10, 10)
            cpu_input1 = self.cpu_sorted_input(cpu_input1)
            npu_input1 = cpu_input1.npu()
            npu_out = npu_out.long()
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            npu_output_out = self.npu_op_exec_out(npu_input1, npu_input2, npu_out)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output, npu_output_out)

    def test_searchsorted_tensor_bool(self):
        shape_format = [
            [[np.int32, 0, [256, 50]], [np.int32, 0, [256, 20]]],
            [[np.int64, 0, [256, 50]], [np.int64, 0, [256, 20]]],
            [[np.float32, 0, [256, 50]], [np.float32, 0, [256, 20]]],
        ]

        for item in shape_format:
            cpu_input1, _ = create_common_tensor(item[0], -10, 10)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -10, 10)
            cpu_input1 = self.cpu_sorted_input(cpu_input1)
            npu_input1 = cpu_input1.npu()
            cpu_output1 = self.cpu_op_exec_bool(cpu_input1, cpu_input2, True, False)
            npu_output1 = self.npu_op_exec_bool(npu_input1, npu_input2, True, False)
            cpu_output2 = self.cpu_op_exec_bool(cpu_input1, cpu_input2, False, True)
            npu_output2 = self.npu_op_exec_bool(npu_input1, npu_input2, False, True)
            self.assertRtolEqual(cpu_output1, npu_output1)
            self.assertRtolEqual(cpu_output2, npu_output2)

    def test_searchsorted_scalar_shape_format(self):
        shape_format = [
            [[np.int32, 0, [128]], 2],
            [[np.int64, 0, [256]], 3],
            [[np.float32, 0, [64]], 2.5],
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], -10, 10)
            cpu_input = self.cpu_sorted_input(cpu_input)
            npu_input = cpu_input.npu()
            scalar = item[1]
            cpu_output = self.cpu_op_exec(cpu_input, scalar)
            npu_output = self.npu_op_exec(npu_input, scalar)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_searchsorted_scalar_bool(self):
        shape_format = [[[np.float32, 0, [64]], 2.5], [[np.int32, 0, [128]], 2], [[np.int64, 0, [256]], 3]]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], -10, 10)
            cpu_input = self.cpu_sorted_input(cpu_input)
            npu_input = cpu_input.npu()
            scalar = item[1]
            cpu_output1 = self.cpu_op_exec_bool(cpu_input, scalar, True, False)
            npu_output1 = self.npu_op_exec_bool(npu_input, scalar, True, False)
            cpu_output2 = self.cpu_op_exec_bool(cpu_input, scalar, False, True)
            npu_output2 = self.npu_op_exec_bool(npu_input, scalar, False, True)
            self.assertRtolEqual(cpu_output1, npu_output1)
            self.assertRtolEqual(cpu_output2, npu_output2)

    def test_searchsorted_side_kwarg_aligns_with_cpu(self):
        """side='right' / side='left' must match CPU (aten passes side_opt; right alone is not enough).

        Uses float32 (+ NaN) as the main path to avoid stack-specific fp64 flaky; see
        ``test_searchsorted_side_kwarg_fp64_nan_aligns_with_cpu`` for fp64 supplement.
        """
        boundaries = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
        values = torch.tensor([1.0, float("nan"), 2.0, float("nan")], dtype=torch.float32)
        npu_b = boundaries.npu()
        npu_v = values.npu()

        cases = [
            {"side": "left"},
            {"side": "right"},
            {"right": False},
            {"right": True},
        ]
        for kwargs in cases:
            cpu_out = torch.searchsorted(boundaries, values, **kwargs)
            npu_out = torch.searchsorted(npu_b, npu_v, **kwargs)
            self.assertEqual(
                cpu_out,
                npu_out.cpu(),
                message=f"searchsorted kwargs={kwargs!r}",
            )

        # Non-NaN sanity: right branch uses strict > for insertion point
        b32 = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
        v32 = torch.tensor([0.25, 1.0, 1.5, 2.5], dtype=torch.float32)
        kwargs_r = {"side": "right"}
        cpu_r = torch.searchsorted(b32, v32, **kwargs_r)
        npu_r = torch.searchsorted(b32.npu(), v32.npu(), **kwargs_r)
        self.assertEqual(cpu_r, npu_r.cpu(), message="float32 side=right")

    def test_searchsorted_side_kwarg_fp64_nan_aligns_with_cpu(self):
        """Supplemental fp64 + NaN vs CPU for ``side``; keep narrow to limit flaky on stacks without full fp64 parity."""
        boundaries = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
        values = torch.tensor([1.0, float("nan"), 2.0, float("nan")], dtype=torch.float64)
        npu_b = boundaries.npu()
        npu_v = values.npu()
        for kwargs in ({"side": "left"}, {"side": "right"}):
            cpu_out = torch.searchsorted(boundaries, values, **kwargs)
            npu_out = torch.searchsorted(npu_b, npu_v, **kwargs)
            self.assertEqual(cpu_out, npu_out.cpu(), message=f"fp64 searchsorted kwargs={kwargs!r}")

    def test_searchsorted_pre_check_invalid_side_matches_cpu(self):
        """``searchsorted_pre_check_npu`` (SearchsortedValidateUtil): invalid ``side`` must error before aclnn / acl."""
        seq_cpu = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
        vals_cpu = torch.tensor([0.5, 1.5], dtype=torch.float32)
        seq_npu = seq_cpu.npu()
        vals_npu = vals_cpu.npu()
        pattern = r"side can only be 'left' or 'right'"
        with self.assertRaisesRegex(RuntimeError, pattern):
            torch.searchsorted(seq_cpu, vals_cpu, side="middle")
        with self.assertRaisesRegex(RuntimeError, pattern):
            torch.searchsorted(seq_npu, vals_npu, side="middle")

    def test_searchsorted_pre_check_side_right_conflict_matches_cpu(self):
        """Explicit ``side='left'`` with ``right=True`` is rejected like CPU."""
        seq_cpu = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
        vals_cpu = torch.tensor([0.5], dtype=torch.float32)
        seq_npu = seq_cpu.npu()
        vals_npu = vals_cpu.npu()
        pattern = "side and right can't be set to opposites"
        with self.assertRaisesRegex(RuntimeError, pattern):
            torch.searchsorted(seq_cpu, vals_cpu, side="left", right=True)
        with self.assertRaisesRegex(RuntimeError, pattern):
            torch.searchsorted(seq_npu, vals_npu, side="left", right=True)
        with self.assertRaisesRegex(RuntimeError, pattern):
            torch.searchsorted(seq_npu, 0.5, side="left", right=True)

    def test_searchsorted_pre_check_sorter_dtype_matches_cpu(self):
        """Sorter must be long; float dtype must raise ATen-style message (not CANN dtype mismatch)."""
        sequence = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
        values_1d = torch.tensor([1.0, 2.0], dtype=torch.float32)
        _, sorted_idx = torch.sort(sequence)
        pattern = "sorter must be a tensor of long dtype"
        with self.assertRaisesRegex(RuntimeError, pattern):
            torch.searchsorted(sequence, values_1d, sorter=sorted_idx.to(torch.float32))
        with self.assertRaisesRegex(RuntimeError, pattern):
            torch.searchsorted(sequence.npu(), values_1d.npu(), sorter=sorted_idx.to(torch.float32).npu())

    def test_searchsorted_pre_check_sorter_shape_mismatch_matches_cpu(self):
        seq = torch.arange(5.0, dtype=torch.float32)
        vals = torch.tensor([1.0, 2.0], dtype=torch.float32)
        sorter = torch.arange(4, dtype=torch.long)
        pattern = "boundary and sorter must have the same size"
        with self.assertRaisesRegex(RuntimeError, pattern):
            torch.searchsorted(seq, vals, sorter=sorter)
        with self.assertRaisesRegex(RuntimeError, pattern):
            torch.searchsorted(seq.npu(), vals.npu(), sorter=sorter.npu())

    def test_searchsorted_pre_check_sorter_index_out_of_range_matches_cpu(self):
        seq = torch.arange(5.0, dtype=torch.float32)
        vals = torch.tensor([1.0], dtype=torch.float32)
        sorter = torch.tensor([0, 1, 2, 3, 10], dtype=torch.long)
        pattern = "sorter index out of range"
        with self.assertRaisesRegex(RuntimeError, pattern):
            torch.searchsorted(seq, vals, sorter=sorter)
        with self.assertRaisesRegex(RuntimeError, pattern):
            torch.searchsorted(seq.npu(), vals.npu(), sorter=sorter.npu())

    def test_searchsorted_pre_check_device_mismatch(self):
        """Cross-device boundaries/values must fail; CPU reports ATen pre_check text, NPU often reports the generic
        wrapper_NPU same-device RuntimeError before custom pre_check runs."""
        seq_cpu = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
        vals_npu = torch.tensor([0.5], dtype=torch.float32).npu()
        pattern = (
            r"boundaries and input value tensors should have same device type"
            r"|Expected all tensors to be on the same device"
        )
        with self.assertRaisesRegex(RuntimeError, pattern):
            torch.searchsorted(seq_cpu, vals_npu)

    def test_searchsorted_pre_check_out_dtype_npu(self):
        """Tensor_out validate path: wrong ``out`` dtype vs ``out_int32`` must error before check_tensor / aclnn."""
        seq = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32).npu()
        vals = torch.tensor([0.5], dtype=torch.float32).npu()
        out = torch.empty(1, dtype=torch.int32, device="npu")
        pattern = "output tensor's dtype is wrong"
        with self.assertRaisesRegex(RuntimeError, pattern):
            torch.searchsorted(seq, vals, out_int32=False, out=out)

    def test_searchsorted_pre_check_leading_dims_mismatch_matches_cpu(self):
        seq = torch.tensor([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], dtype=torch.float32)
        vals = torch.tensor([0.5, 0.5], dtype=torch.float32)
        pattern = "first N-1 dimensions of boundaries"
        with self.assertRaisesRegex(RuntimeError, pattern):
            torch.searchsorted(seq, vals)
        with self.assertRaisesRegex(RuntimeError, pattern):
            torch.searchsorted(seq.npu(), vals.npu())

    def test_searchsorted_noncontiguous_warns(self):
        """WarnUtil: non-contiguous boundary / values should emit TORCH_WARN_ONCE-style user warning."""
        # `.t()` on a 2x3 tensor changes shape to 3x2 and breaks batch alignment with 2x2 values; keep [2, 3] x [2, 2]
        # while forcing non-contiguous strides (same trick as non-contiguous views without permuting dims wrongly).
        seq = torch.tensor([[0.0, 1.0, 2.0], [10.0, 20.0, 30.0]], dtype=torch.float32).npu()
        seq = seq.transpose(0, 1).contiguous().transpose(0, 1)
        vals = torch.tensor([[0.5, 1.5], [15.0, 25.0]], dtype=torch.float32).npu().t()
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            torch.searchsorted(seq, vals)
        msgs = [str(w.message) for w in recorded]
        self.assertTrue(
            any("non-contiguous" in m for m in msgs),
            msg=f"expected non-contiguous warning, got: {msgs}",
        )


if __name__ == "__main__":
    run_tests()

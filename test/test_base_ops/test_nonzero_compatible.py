import os

# The C++ side caches this via a static std::getenv, so it must be set before the
# first nonzero call. A separate test file runs in its own process (see
# run_test.py), which keeps the default-path cases in test_nonzero.py intact.
os.environ["TORCH_NPU_USE_COMPATIBLE_IMPL"] = "1"

import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests

torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False


class TestNonzeroCompatible(TestCase):
    """Test nonzero with TORCH_NPU_USE_COMPATIBLE_IMPL=1.

    In compatible mode NonzeroKernelNpuOpApi first counts the exact number of
    non-zero elements and allocates the output at size {num_nonzero, dim}
    instead of the conservative max size {numel, dim}, which reduces device
    memory usage. The returned indices must still match the CPU result.
    """

    def cpu_op_exec(self, input1):
        output = torch.nonzero(input1)
        return output.numpy().astype(np.int32)

    def npu_op_exec(self, input1):
        output = torch.nonzero(input1.npu())
        output = output.to("cpu")
        return output.numpy().astype(np.int32)

    def test_nonzero_partial(self):
        """Partial non-zeros; row count must equal the exact non-zero count."""
        dtype_list = [torch.float32, torch.float16, torch.int32, torch.int64, torch.bool]
        for dtype in dtype_list:
            base = torch.randn(256, 10)
            base[base < 0] = 0.0
            if dtype == torch.bool:
                cpu_input = base != 0
            else:
                cpu_input = base.to(dtype)

            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(cpu_input)

            self.assertEqual(npu_output.shape, cpu_output.shape)
            self.assertEqual(npu_output.shape[0], int((cpu_input != 0).sum().item()))
            self.assertRtolEqual(cpu_output, npu_output)

    def test_nonzero_all_zero(self):
        """All-zero input; compatible mode must allocate exactly {0, dim}."""
        for shape in [[256, 10], [5, 4, 3]]:
            cpu_input = torch.zeros(shape)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(cpu_input)

            self.assertEqual(cpu_output.shape, (0, len(shape)))
            self.assertEqual(npu_output.shape, (0, len(shape)))
            self.assertRtolEqual(cpu_output, npu_output)

    def test_nonzero_all_nonzero(self):
        """All-non-zero input; output shape is exactly {numel, dim}."""
        cpu_input = torch.ones(7, 4)
        cpu_output = self.cpu_op_exec(cpu_input)
        npu_output = self.npu_op_exec(cpu_input)

        self.assertEqual(npu_output.shape, (28, 2))
        self.assertRtolEqual(cpu_output, npu_output)

    def test_nonzero_high_dim(self):
        """3D/4D input with mixed zero and non-zero elements."""
        for shape in [[256, 256, 100], [5, 256, 256, 100]]:
            cpu_input = torch.randn(shape)
            cpu_input[cpu_input < 0] = 0.0

            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(cpu_input)

            self.assertEqual(npu_output.shape[1], len(shape))
            self.assertRtolEqual(cpu_output, npu_output)

    def test_nonzero_scalar(self):
        """0-dim inputs, both zero and non-zero."""
        for cpu_input in [torch.tensor(0, dtype=torch.bool), torch.tensor(1.0),
                          torch.tensor(0.0)]:
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(cpu_input)

            self.assertEqual(npu_output.shape, cpu_output.shape)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_nonzero_empty(self):
        """Empty tensor (numel == 0)."""
        cpu_input = torch.zeros(0, 3)
        cpu_output = self.cpu_op_exec(cpu_input)
        npu_output = self.npu_op_exec(cpu_input)

        self.assertEqual(npu_output.shape, (0, 2))
        self.assertRtolEqual(cpu_output, npu_output)

    def test_nonzero_out(self):
        """nonzero_out path allocates at the exact size and fills correctly."""
        for dtype in [torch.float32, torch.int32]:
            base = torch.randn(256, 10)
            base[base < 0] = 0.0
            cpu_input = base.to(dtype)

            cpu_output = self.cpu_op_exec(cpu_input)
            npu_input = cpu_input.npu()
            result = torch.empty(cpu_output.shape, dtype=torch.int64).npu()
            npu_result = torch.nonzero(npu_input, out=result)

            self.assertEqual(npu_result.shape, cpu_output.shape)
            self.assertRtolEqual(cpu_output, npu_result.cpu().numpy().astype(np.int32))


if __name__ == "__main__":
    run_tests()

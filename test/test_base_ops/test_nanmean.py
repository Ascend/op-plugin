import unittest

import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices


class TestNanmean(TestCase):

    @staticmethod
    def cpu_op_exec(input1, dim=None, keepdim=False, dtype=None):
        output = torch.nanmean(input1, dim=dim, keepdim=keepdim, dtype=dtype)
        if input1.dtype == torch.bfloat16 or (dtype is not None and dtype == torch.bfloat16):
            return output.float().numpy()
        return output.numpy()

    @staticmethod
    def npu_op_exec(input1, dim=None, keepdim=False, dtype=None):
        input1 = input1.to("npu")
        output = torch.nanmean(input1, dim=dim, keepdim=keepdim, dtype=dtype)
        if input1.dtype == torch.bfloat16 or (dtype is not None and dtype == torch.bfloat16):
            return output.cpu().float().numpy()
        return output.cpu().numpy()

    @staticmethod
    def cpu_op_out_exec(input1, out, dim=None, keepdim=False, dtype=None):
        torch.nanmean(input1, dim=dim, keepdim=keepdim, dtype=dtype, out=out)
        if input1.dtype == torch.bfloat16 or (dtype is not None and dtype == torch.bfloat16):
            return out.float().numpy()
        return out.numpy()

    @staticmethod
    def npu_op_out_exec(input1, out, dim=None, keepdim=False, dtype=None):
        input1 = input1.to("npu")
        out = out.to("npu")
        torch.nanmean(input1, dim=dim, keepdim=keepdim, dtype=dtype, out=out)
        if input1.dtype == torch.bfloat16 or (dtype is not None and dtype == torch.bfloat16):
            return out.cpu().float().numpy()
        return out.cpu().numpy()

    @staticmethod
    def _make_nan_input(shape, dtype):
        """Create a tensor with some NaN values for testing."""
        cpu_input = torch.rand(shape, dtype=torch.float32)
        # Inject NaN values at known positions
        cpu_input[0] = float('nan')
        if len(shape) > 1:
            cpu_input[:, 0] = float('nan')
        if len(shape) > 2:
            cpu_input[:, :, 0] = float('nan')
        return cpu_input.to(dtype)

    def test_nanmean_no_dim(self):
        """Test nanmean without dim (reduce all)."""
        shape_list = [(8,), (4, 8), (4, 8, 16)]
        dtype_list = [torch.float32, torch.float16]

        for shape in shape_list:
            for dtype in dtype_list:
                cpu_input = self._make_nan_input(shape, dtype)
                npu_input = cpu_input.npu()
                cpu_output = self.cpu_op_exec(cpu_input)
                npu_output = self.npu_op_exec(npu_input)
                self.assertRtolEqual(cpu_output, npu_output)

    def test_nanmean_with_dim(self):
        """Test nanmean with dim specified."""
        shape = (4, 8, 16)
        dtype_list = [torch.float32, torch.float16]

        for dtype in dtype_list:
            cpu_input = self._make_nan_input(shape, dtype)
            npu_input = cpu_input.npu()
            for dim in [0, 1, 2, [0, 1], [1, 2]]:
                for keepdim in [True, False]:
                    cpu_output = self.cpu_op_exec(cpu_input, dim=dim, keepdim=keepdim)
                    npu_output = self.npu_op_exec(npu_input, dim=dim, keepdim=keepdim)
                    self.assertRtolEqual(cpu_output, npu_output)

    def test_nanmean_with_dtype(self):
        """Test nanmean with dtype parameter."""
        shape = (4, 8, 16)
        cpu_input = self._make_nan_input(shape, torch.float16)

        dtype_list = [torch.float32, torch.float16]
        for dtype in dtype_list:
            cpu_output = self.cpu_op_exec(cpu_input, dim=1, dtype=dtype)
            npu_input = cpu_input.npu()
            npu_output = self.npu_op_exec(npu_input, dim=1, dtype=dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_nanmean_out(self):
        """Test nanmean with out parameter."""
        shape = (4, 8, 16)

        # out with dim specified
        cpu_input = self._make_nan_input(shape, torch.float32)
        npu_input = cpu_input.npu()
        cpu_out = torch.zeros((4, 1, 16), dtype=torch.float32)
        npu_out = cpu_out.npu()
        cpu_output = self.cpu_op_out_exec(cpu_input, cpu_out, dim=1, keepdim=True)
        npu_output = self.npu_op_out_exec(npu_input, npu_out, dim=1, keepdim=True)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_nanmean_no_nan(self):
        """Test nanmean on input without NaN (should equal mean)."""
        shape_format = [
            [np.float32, 0, (4, 8, 16)],
            [np.float16, 0, (4, 8, 16)],
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input, dim=1)
            npu_output = self.npu_op_exec(npu_input, dim=1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_nanmean_all_nan(self):
        """Test nanmean on input where all values are NaN."""
        cpu_input = torch.full((4, 8), float('nan'), dtype=torch.float32)
        npu_input = cpu_input.npu()
        cpu_output = self.cpu_op_exec(cpu_input, dim=1)
        npu_output = self.npu_op_exec(npu_input, dim=1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_nanmean_bfloat16(self):
        """Test nanmean with bfloat16."""
        shape = (4, 8, 16)
        cpu_input = self._make_nan_input(shape, torch.bfloat16)
        npu_input = cpu_input.npu()

        for dim in [None, 1, [0, 1]]:
            cpu_output = self.cpu_op_exec(cpu_input, dim=dim)
            npu_output = self.npu_op_exec(npu_input, dim=dim)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()

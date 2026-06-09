import unittest
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestConstantPadNd(TestCase):

    def test_constant_pad_nd_with_negative(self):
        x = torch.randint(0, 100, (1, 10), dtype=torch.int8)
        x_npu = x.npu()
        pad = (1, -1, 1, -1)
        value = -74
        res = torch.constant_pad_nd(x, pad, value)
        res_npu = torch.constant_pad_nd(x_npu, pad, value)
        self.assertEqual(res, res_npu)


class TestConstantPadNdFp4(TestCase):
    """Tests for constant_pad_nd with FP4 (float4_e2m1fn_x2) dtype support.

    FP4 stores 2 values per byte, so:
    - All pad values must be even
    - The padded dimension = (original_dim * 2 + pad_left + pad_right) / 2
    """

    def _create_fp4_tensor(self, shape):
        """Create an FP4 tensor on NPU with given logical shape.

        Since float4_e2m1fn_x2 packs 2 fp4 values per byte.
        """
        x = torch.randint(0, 16, shape, dtype=torch.uint8)
        return x.view(torch.float4_e2m1fn_x2).npu()

    @SupportedDevices(['Ascend950'])
    def test_fp4_pad_1d_even(self):
        """FP4 1D padding with even pad values."""
        x = self._create_fp4_tensor((8,))
        pad = (2, 4)  # both even
        result = torch.constant_pad_nd(x, pad, 0)
        # expected last dim = (8*2 + 2 + 4) // 2 = 11
        self.assertEqual(result.shape, (11,))

    @SupportedDevices(['Ascend950'])
    def test_fp4_pad_2d_even(self):
        """FP4 2D padding with even pad values."""
        x = self._create_fp4_tensor((4, 8))
        pad = (2, 4, 0, 2)  # all even
        result = torch.constant_pad_nd(x, pad, 0)
        # dim1 = (8*2 + 2 + 4) // 2 = 11, dim0 = (4 + 0 + 2) = 6
        self.assertEqual(result.shape, (6, 11,))

    @SupportedDevices(['Ascend950'])
    def test_fp4_pad_3d_even(self):
        """FP4 3D padding with even pad values."""
        x = self._create_fp4_tensor((2, 4, 8))
        pad = (0, 2, 2, 2, 0, 0)  # all even
        result = torch.constant_pad_nd(x, pad, 0)
        # dim2 = (8*2 + 0 + 2) // 2 = 9, dim1 = (4 + 2 + 2) = 8
        self.assertEqual(result.shape, (2, 8, 9))

    @SupportedDevices(['Ascend950'])
    def test_fp4_pad_zero(self):
        """FP4 padding with all zeros — shape unchanged (divided by 2)."""
        x = self._create_fp4_tensor((4, 8))
        pad = (0, 0)
        result = torch.constant_pad_nd(x, pad, 0)
        # dim1 = (8*2 + 0 + 0) // 2 = 8
        self.assertEqual(result.shape, (4, 8))

    @SupportedDevices(['Ascend950'])
    def test_fp4_pad_odd_raises(self):
        """FP4 padding with odd pad value must raise an error."""
        x = self._create_fp4_tensor((4, 8))
        pad = (1, 2)  # 1 is odd
        with self.assertRaises(RuntimeError):
            torch.constant_pad_nd(x, pad, 0)

    @SupportedDevices(['Ascend950'])
    def test_fp4_pad_all_odd_raises(self):
        """FP4 padding with all odd pad values must raise an error."""
        x = self._create_fp4_tensor((4, 8))
        pad = (3, 5)  # both odd
        with self.assertRaises(RuntimeError):
            torch.constant_pad_nd(x, pad, 0)

    @SupportedDevices(['Ascend950'])
    def test_fp4_pad_negative_even(self):
        """FP4 negative padding with even values — shrinks dimension."""
        x = self._create_fp4_tensor((4, 8))
        pad = (-2, 0)  # even negative
        with self.assertRaises(RuntimeError):
            torch.constant_pad_nd(x, pad, 0)

    @SupportedDevices(['Ascend950'])
    def test_fp4_pad_negative_odd_raises(self):
        """FP4 negative odd padding must raise an error (not divisible by 2)."""
        x = self._create_fp4_tensor((4, 8))
        pad = (-1, 0)  # -1 is odd
        with self.assertRaises(RuntimeError):
            torch.constant_pad_nd(x, pad, 0)

    @SupportedDevices(['Ascend950'])
    def test_fp4_pad_4d(self):
        """FP4 4D tensor padding."""
        x = self._create_fp4_tensor((1, 2, 4, 8))
        pad = (0, 2, 2, 0)
        result = torch.constant_pad_nd(x, pad, 0)
        # dim3 = (8*2 + 0 + 2) // 2 = 9, dim2 = (4 + 2 + 0) = 6
        self.assertEqual(result.shape, (1, 2, 6, 9))


class TestConstantPadNdFp8(TestCase):
    """Tests for constant_pad_nd with FP8 dtype support.

    FP8 uses standard 1-byte-per-element storage, so padding behaves
    like normal dtypes — no even/odd constraints.
    """

    def _create_fp8_e4m3_tensor(self, shape):
        """Create a float8_e4m3fn tensor on NPU."""
        x = torch.rand(shape, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        return x.npu()

    def _create_fp8_e5m2_tensor(self, shape):
        """Create a float8_e5m2 tensor on NPU."""
        x = torch.rand(shape, dtype=torch.bfloat16).to(torch.float8_e5m2)
        return x.npu()

    # --- float8_e4m3fn tests ---

    @SupportedDevices(['Ascend950'])
    def test_fp8_e4m3_pad_1d(self):
        """FP8 e4m3 1D padding."""
        x = self._create_fp8_e4m3_tensor((10,))
        pad = (2, 3)
        result = torch.constant_pad_nd(x, pad, 0)
        self.assertEqual(result.shape, (15,))
        self.assertEqual(result.dtype, torch.float8_e4m3fn)

    @SupportedDevices(['Ascend950'])
    def test_fp8_e4m3_pad_2d(self):
        """FP8 e4m3 2D padding."""
        x = self._create_fp8_e4m3_tensor((4, 8))
        pad = (1, 2, 3, 0)
        result = torch.constant_pad_nd(x, pad, 0)
        # dim1 = 8 + 1 + 2 = 11, dim0 = 4 + 3 + 0 = 7
        self.assertEqual(result.shape, (7, 11))
        self.assertEqual(result.dtype, torch.float8_e4m3fn)

    @SupportedDevices(['Ascend950'])
    def test_fp8_e4m3_pad_3d(self):
        """FP8 e4m3 3D padding."""
        x = self._create_fp8_e4m3_tensor((2, 4, 8))
        pad = (0, 2, 1, 1, 0, 0)
        result = torch.constant_pad_nd(x, pad, 0)
        # dim2 = 8 + 0 + 2 = 10, dim1 = 4 + 1 + 1 = 6
        self.assertEqual(result.shape, (2, 6, 10))
        self.assertEqual(result.dtype, torch.float8_e4m3fn)

    @SupportedDevices(['Ascend950'])
    def test_fp8_e4m3_pad_4d(self):
        """FP8 e4m3 4D tensor padding (NCHW-like)."""
        x = self._create_fp8_e4m3_tensor((1, 3, 8, 8))
        pad = (1, 1, 1, 1)
        result = torch.constant_pad_nd(x, pad, 0)
        self.assertEqual(result.shape, (1, 3, 10, 10))
        self.assertEqual(result.dtype, torch.float8_e4m3fn)

    @SupportedDevices(['Ascend950'])
    def test_fp8_e4m3_pad_negative(self):
        """FP8 e4m3 negative padding — shrinks dimension."""
        x = self._create_fp8_e4m3_tensor((4, 8))
        pad = (-1, -2)
        with self.assertRaises(RuntimeError):
            torch.constant_pad_nd(x, pad, 0)

    @SupportedDevices(['Ascend950'])
    def test_fp8_e4m3_pad_with_value(self):
        """FP8 e4m3 padding with non-zero pad value."""
        x = self._create_fp8_e4m3_tensor((4,))
        pad = (1, 1)
        result = torch.constant_pad_nd(x, pad, 1.0)
        self.assertEqual(result.shape, (6,))

    @SupportedDevices(['Ascend950'])
    def test_fp8_e4m3_pad_zero_no_change(self):
        """FP8 e4m3 padding with all zeros — shape unchanged."""
        x = self._create_fp8_e4m3_tensor((4, 8))
        pad = (0, 0)
        result = torch.constant_pad_nd(x, pad, 0)
        self.assertEqual(result.shape, (4, 8))

    # --- float8_e5m2 tests ---

    @SupportedDevices(['Ascend950'])
    def test_fp8_e5m2_pad_1d(self):
        """FP8 e5m2 1D padding."""
        x = self._create_fp8_e5m2_tensor((10,))
        pad = (2, 3)
        result = torch.constant_pad_nd(x, pad, 0)
        self.assertEqual(result.shape, (15,))
        self.assertEqual(result.dtype, torch.float8_e5m2)

    @SupportedDevices(['Ascend950'])
    def test_fp8_e5m2_pad_2d(self):
        """FP8 e5m2 2D padding."""
        x = self._create_fp8_e5m2_tensor((4, 8))
        pad = (1, 2, 3, 0)
        result = torch.constant_pad_nd(x, pad, 0)
        self.assertEqual(result.shape, (7, 11))
        self.assertEqual(result.dtype, torch.float8_e5m2)

    @SupportedDevices(['Ascend950'])
    def test_fp8_e5m2_pad_4d(self):
        """FP8 e5m2 4D tensor padding."""
        x = self._create_fp8_e5m2_tensor((1, 3, 8, 8))
        pad = (1, 1, 1, 1)
        result = torch.constant_pad_nd(x, pad, 0)
        self.assertEqual(result.shape, (1, 3, 10, 10))
        self.assertEqual(result.dtype, torch.float8_e5m2)

    @SupportedDevices(['Ascend950'])
    def test_fp8_e5m2_pad_negative(self):
        """FP8 e5m2 negative padding — shrinks dimension."""
        x = self._create_fp8_e5m2_tensor((4, 8))
        pad = (-2, -1)
        with self.assertRaises(RuntimeError):
            torch.constant_pad_nd(x, pad, 0)

    @SupportedDevices(['Ascend950'])
    def test_fp8_e5m2_pad_with_value(self):
        """FP8 e5m2 padding with non-zero pad value."""
        x = self._create_fp8_e5m2_tensor((4,))
        pad = (1, 1)
        result = torch.constant_pad_nd(x, pad, 1.0)
        self.assertEqual(result.shape, (6,))


if __name__ == "__main__":
    run_tests()

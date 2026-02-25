import math
import unittest
import numpy as np
import torch
import torch_npu
import torch.nn as nn
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestQuantMatmulGelu(TestCase):

    def gelu_tanh(self, x):
        """GELU tanh approximation"""
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

    def gelu_erf(self, x):
        """GELU erf"""
        m = nn.GELU()
        return m(x)

    def cpu_quant_matmul_gelu(self, x1, x2, x1_scale, x2_scale, bias=None, approximate="gelu_erf"):
        """CPU reference implementation"""
        # Handle different input types
        if x1.dtype == torch.quint4x2:
            # For quint4x2 (direct INT4), convert to int32 for computation
            # quint4x2 stores 2 INT4 values per uint8, so we view as uint8 then to int32
            x1_int32 = x1.view(torch.uint8).to(torch.int32)
            x2_int32 = x2.view(torch.uint8).to(torch.int32)
        elif x1.dtype == torch.int32:
            # For int32 (packed INT4), each int32 stores 8 int4 values
            # We need to expand the dimensions: (m, k//8) -> (m, k) and (k, n//8) -> (k, n)
            # For simplicity, we'll use the packed dimensions directly
            # Note: This is a simplified CPU reference, actual unpacking would be more complex
            x1_int32 = x1.to(torch.int32)
            x2_int32 = x2.to(torch.int32)
            # For packed INT4 in INT32, we need to handle dimension expansion
            # Since CPU reference is simplified, we'll use the packed shape directly
            # The actual unpacking logic would expand each int32 to 8 int4 values
        else:
            # A8W8 case: int8
            x1_int32 = x1.to(torch.int32)
            x2_int32 = x2.to(torch.int32)
        
        # Compute quantized matmul: x1 @ x2 * x1_scale * x2_scale
        result = torch.matmul(x1_int32, x2_int32).float()
        result = result * x1_scale.view(-1, 1) * x2_scale.view(1, -1)

        # Add bias if provided
        if bias is not None:
            result = result + bias.view(1, -1)

        # Apply GELU
        if approximate == "gelu_tanh":
            result = self.gelu_tanh(result)
        elif approximate == "gelu_erf":
            result = self.gelu_erf(result)
        else:
            raise ValueError(f"Unsupported approximate: {approximate}")

        return result

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_gelu_a8w8_nd_gelu_tanh(self):
        """Test A8W8 with ND format and gelu_tanh"""
        torch.manual_seed(0)
        m, k, n = 128, 256, 512

        x1 = torch.randint(-5, 5, (m, k), dtype=torch.int8)
        x2 = torch.randint(-5, 5, (k, n), dtype=torch.int8)
        x1_scale = torch.randn(m, dtype=torch.float32).abs() * 0.01
        x2_scale = torch.randn(n, dtype=torch.float32).abs() * 0.01

        # CPU reference
        cpu_result = self.cpu_quant_matmul_gelu(x1, x2, x1_scale, x2_scale, approximate="gelu_tanh")

        # NPU custom op
        npu_result = torch_npu.npu_quant_matmul_gelu(
            x1.npu(), x2.npu(), x1_scale.npu(), x2_scale.npu(), approximate="gelu_tanh")

        self.assertRtolEqual(cpu_result.numpy(), npu_result.cpu().float().numpy(), 0.01)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_gelu_a8w8_nd_gelu_erf(self):
        """Test A8W8 with ND format and gelu_erf"""
        torch.manual_seed(0)
        m, k, n = 128, 256, 512

        x1 = torch.randint(-5, 5, (m, k), dtype=torch.int8)
        x2 = torch.randint(-5, 5, (k, n), dtype=torch.int8)
        x1_scale = torch.randn(m, dtype=torch.float32).abs() * 0.01
        x2_scale = torch.randn(n, dtype=torch.float32).abs() * 0.01

        # CPU reference
        cpu_result = self.cpu_quant_matmul_gelu(x1, x2, x1_scale, x2_scale, approximate="gelu_erf")

        # NPU custom op
        npu_result = torch_npu.npu_quant_matmul_gelu(
            x1.npu(), x2.npu(), x1_scale.npu(), x2_scale.npu(), approximate="gelu_erf")

        self.assertRtolEqual(cpu_result.numpy(), npu_result.cpu().float().numpy(), 0.01)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_gelu_a8w8_nd_with_bias(self):
        """Test A8W8 with ND format, gelu_tanh and bias"""
        torch.manual_seed(0)
        m, k, n = 128, 256, 512

        x1 = torch.randint(-5, 5, (m, k), dtype=torch.int8)
        x2 = torch.randint(-5, 5, (k, n), dtype=torch.int8)
        x1_scale = torch.randn(m, dtype=torch.float32).abs() * 0.01
        x2_scale = torch.randn(n, dtype=torch.float32).abs() * 0.01
        bias = torch.randn(n, dtype=torch.float32) * 0.1

        # CPU reference
        cpu_result = self.cpu_quant_matmul_gelu(x1, x2, x1_scale, x2_scale, approximate="gelu_tanh", bias=bias)

        # NPU custom op
        npu_result = torch_npu.npu_quant_matmul_gelu(
            x1.npu(), x2.npu(), x1_scale.npu(), x2_scale.npu(),
            bias=bias.npu(), approximate="gelu_tanh")

        self.assertRtolEqual(cpu_result.numpy(), npu_result.cpu().float().numpy(), 0.01)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_gelu_a8w8_nz_gelu_tanh(self):
        """Test A8W8 with NZ format and gelu_tanh"""
        torch.manual_seed(0)
        m, k, n = 128, 256, 512

        x1 = torch.randint(-1, 1, (m, k), dtype=torch.int8)
        x2 = torch.randint(-1, 1, (k, n), dtype=torch.int8)
        x1_scale = torch.randn(m, dtype=torch.float32).abs() * 0.01
        x2_scale = torch.randn(n, dtype=torch.float32).abs() * 0.01

        # CPU reference
        cpu_result = self.cpu_quant_matmul_gelu(x1, x2, x1_scale, x2_scale, approximate="gelu_tanh")

        # Convert x2 to NZ format
        x2_nz = torch_npu.npu_format_cast(x2.npu().contiguous(), 29)  # 29 is ACL_FORMAT_FRACTAL_NZ

        # NPU custom op with NZ format
        npu_result = torch_npu.npu_quant_matmul_gelu(
            x1.npu(), x2_nz, x1_scale.npu(), x2_scale.npu(), approximate="gelu_tanh")

        self.assertRtolEqual(cpu_result.numpy(), npu_result.cpu().float().numpy(), 0.01)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_gelu_a8w8_nz_gelu_erf(self):
        """Test A8W8 with NZ format and gelu_erf"""
        torch.manual_seed(0)
        m, k, n = 128, 256, 512

        x1 = torch.randint(-1, 1, (m, k), dtype=torch.int8)
        x2 = torch.randint(-1, 1, (k, n), dtype=torch.int8)
        x1_scale = torch.randn(m, dtype=torch.float32).abs() * 0.01
        x2_scale = torch.randn(n, dtype=torch.float32).abs() * 0.01

        # CPU reference
        cpu_result = self.cpu_quant_matmul_gelu(x1, x2, x1_scale, x2_scale, approximate="gelu_erf")

        # Convert x2 to NZ format
        x2_nz = torch_npu.npu_format_cast(x2.npu().contiguous(), 29)  # 29 is ACL_FORMAT_FRACTAL_NZ

        # NPU custom op with NZ format
        npu_result = torch_npu.npu_quant_matmul_gelu(
            x1.npu(), x2_nz, x1_scale.npu(), x2_scale.npu(), approximate="gelu_erf")

        self.assertRtolEqual(cpu_result.numpy(), npu_result.cpu().float().numpy(), 0.01)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_gelu_a8w8_nz_simple(self):
        """Test A8W8 with NZ format - simple case with smaller dimensions"""
        torch.manual_seed(0)
        m, k, n = 32, 64, 128

        x1 = torch.randint(-1, 1, (m, k), dtype=torch.int8)
        x2 = torch.randint(-1, 1, (k, n), dtype=torch.int8)
        x1_scale = torch.randn(m, dtype=torch.float32).abs() * 0.01
        x2_scale = torch.randn(n, dtype=torch.float32).abs() * 0.01

        # CPU reference
        cpu_result = self.cpu_quant_matmul_gelu(x1, x2, x1_scale, x2_scale, approximate="gelu_tanh")

        # Convert x2 to NZ format
        x2_nz = torch_npu.npu_format_cast(x2.npu().contiguous(), 29)  # 29 is ACL_FORMAT_FRACTAL_NZ

        # NPU custom op with NZ format
        npu_result = torch_npu.npu_quant_matmul_gelu(
            x1.npu(), x2_nz, x1_scale.npu(), x2_scale.npu(), approximate="gelu_tanh")

        self.assertRtolEqual(cpu_result.numpy(), npu_result.cpu().float().numpy(), 0.01)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_gelu_a4w4_nd_gelu_tanh_int4(self):
        """Test A4W4 with ND format, gelu_tanh, using quint4x2 (direct INT4) type"""
        torch.manual_seed(0)
        m, k, n = 128, 256, 512

        # Generate INT4 data using quint4x2 type
        x1_fp = torch.randn(m, k, dtype=torch.float32)
        x2_fp = torch.randn(k, n, dtype=torch.float32)

        # Quantize to INT4 (quint4x2)
        scale_tmp = torch.ones(1, dtype=torch.float32).npu()
        x1 = torch_npu.npu_quantize(x1_fp.npu(), scale_tmp, None, torch.quint4x2, -1, False)
        x2 = torch_npu.npu_quantize(x2_fp.npu(), scale_tmp, None, torch.quint4x2, -1, False)

        x1_scale = torch.randn(m, dtype=torch.float32).abs() * 0.01
        x2_scale = torch.randn(n, dtype=torch.float32).abs() * 0.01

        # NPU custom op
        npu_result = torch_npu.npu_quant_matmul_gelu(
            x1, x2, x1_scale.npu(), x2_scale.npu(), approximate="gelu_tanh")

        # For A4W4, we mainly test that the op runs without errors
        # Full accuracy comparison requires proper INT4 CPU implementation
        self.assertEqual(npu_result.shape, (m, n))
        self.assertTrue(torch.isfinite(npu_result).all())
        self.assertEqual(npu_result.dtype, torch.float16)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_gelu_a4w4_nd_gelu_erf_int4(self):
        """Test A4W4 with ND format, gelu_erf, using quint4x2 (direct INT4) type"""
        torch.manual_seed(0)
        m, k, n = 64, 128, 256

        # Generate INT4 data using quint4x2 type
        x1_fp = torch.randn(m, k, dtype=torch.float32)
        x2_fp = torch.randn(k, n, dtype=torch.float32)

        # Quantize to INT4 (quint4x2)
        scale_tmp = torch.ones(1, dtype=torch.float32).npu()
        x1 = torch_npu.npu_quantize(x1_fp.npu(), scale_tmp, None, torch.quint4x2, -1, False)
        x2 = torch_npu.npu_quantize(x2_fp.npu(), scale_tmp, None, torch.quint4x2, -1, False)

        x1_scale = torch.randn(m, dtype=torch.float32).abs() * 0.01
        x2_scale = torch.randn(n, dtype=torch.float32).abs() * 0.01

        # NPU custom op
        npu_result = torch_npu.npu_quant_matmul_gelu(
            x1, x2, x1_scale.npu(), x2_scale.npu(), approximate="gelu_erf")

        self.assertEqual(npu_result.shape, (m, n))
        self.assertTrue(torch.isfinite(npu_result).all())
        self.assertEqual(npu_result.dtype, torch.float16)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_gelu_a4w4_nd_with_bias_int4(self):
        """Test A4W4 with ND format, gelu_tanh, bias, using quint4x2 (direct INT4) type"""
        torch.manual_seed(0)
        m, k, n = 64, 128, 256

        # Generate INT4 data using quint4x2 type
        x1_fp = torch.randn(m, k, dtype=torch.float32)
        x2_fp = torch.randn(k, n, dtype=torch.float32)

        # Quantize to INT4 (quint4x2)
        scale_tmp = torch.ones(1, dtype=torch.float32).npu()
        x1 = torch_npu.npu_quantize(x1_fp.npu(), scale_tmp, None, torch.quint4x2, -1, False)
        x2 = torch_npu.npu_quantize(x2_fp.npu(), scale_tmp, None, torch.quint4x2, -1, False)

        x1_scale = torch.randn(m, dtype=torch.float32).abs() * 0.01
        x2_scale = torch.randn(n, dtype=torch.float32).abs() * 0.01
        bias = torch.randint(-5, 5, (n,), dtype=torch.int32)

        # NPU custom op
        npu_result = torch_npu.npu_quant_matmul_gelu(
            x1, x2, x1_scale.npu(), x2_scale.npu(),
            bias=bias.npu(), approximate="gelu_tanh")

        self.assertEqual(npu_result.shape, (m, n))
        self.assertTrue(torch.isfinite(npu_result).all())
        self.assertEqual(npu_result.dtype, torch.float16)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_gelu_a4w4_nd_gelu_tanh_int32(self):
        """Test A4W4 with ND format, gelu_tanh, using int32 (packed INT4) type"""
        torch.manual_seed(0)
        m, k, n = 128, 256, 512

        # Generate INT4 data packed in INT32 format (8 int4 values per int32)
        # Shape for int32: (m, k//8) and (k, n//8)
        k_packed = k // 8
        n_packed = n // 8
        
        x1 = torch.randint(-8, 8, (m, k_packed), dtype=torch.int32)
        x2 = torch.randint(-8, 8, (k, n_packed), dtype=torch.int32)

        x1_scale = torch.randn(m, dtype=torch.float32).abs() * 0.01
        x2_scale = torch.randn(n, dtype=torch.float32).abs() * 0.01

        # NPU custom op
        npu_result = torch_npu.npu_quant_matmul_gelu(
            x1.npu(), x2.npu(), x1_scale.npu(), x2_scale.npu(), approximate="gelu_tanh")

        # Verify output shape: (m, n) where n is recovered from n_packed * 8
        self.assertEqual(npu_result.shape, (m, n))
        self.assertTrue(torch.isfinite(npu_result).all())
        self.assertEqual(npu_result.dtype, torch.float16)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_gelu_a4w4_nd_gelu_erf_int32(self):
        """Test A4W4 with ND format, gelu_erf, using int32 (packed INT4) type"""
        torch.manual_seed(0)
        m, k, n = 64, 128, 256

        # Generate INT4 data packed in INT32 format
        k_packed = k // 8
        n_packed = n // 8
        
        x1 = torch.randint(-8, 8, (m, k_packed), dtype=torch.int32)
        x2 = torch.randint(-8, 8, (k, n_packed), dtype=torch.int32)

        x1_scale = torch.randn(m, dtype=torch.float32).abs() * 0.01
        x2_scale = torch.randn(n, dtype=torch.float32).abs() * 0.01

        # NPU custom op
        npu_result = torch_npu.npu_quant_matmul_gelu(
            x1.npu(), x2.npu(), x1_scale.npu(), x2_scale.npu(), approximate="gelu_erf")

        self.assertEqual(npu_result.shape, (m, n))
        self.assertTrue(torch.isfinite(npu_result).all())
        self.assertEqual(npu_result.dtype, torch.float16)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_gelu_a4w4_nd_with_bias_int32(self):
        """Test A4W4 with ND format, gelu_tanh, bias, using int32 (packed INT4) type"""
        torch.manual_seed(0)
        m, k, n = 64, 128, 256

        # Generate INT4 data packed in INT32 format
        k_packed = k // 8
        n_packed = n // 8
        
        x1 = torch.randint(-8, 8, (m, k_packed), dtype=torch.int32)
        x2 = torch.randint(-8, 8, (k, n_packed), dtype=torch.int32)

        x1_scale = torch.randn(m, dtype=torch.float32).abs() * 0.01
        x2_scale = torch.randn(n, dtype=torch.float32).abs() * 0.01
        bias = torch.randint(-5, 5, (n,), dtype=torch.int32)

        # NPU custom op
        npu_result = torch_npu.npu_quant_matmul_gelu(
            x1.npu(), x2.npu(), x1_scale.npu(), x2_scale.npu(),
            bias=bias.npu(), approximate="gelu_tanh")

        self.assertEqual(npu_result.shape, (m, n))
        self.assertTrue(torch.isfinite(npu_result).all())
        self.assertEqual(npu_result.dtype, torch.float16)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_gelu_bf16_output(self):
        """Test with BF16 output (x2_scale is BF16)"""
        torch.manual_seed(0)
        m, k, n = 64, 128, 256

        x1 = torch.randint(-5, 5, (m, k), dtype=torch.int8)
        x2 = torch.randint(-5, 5, (k, n), dtype=torch.int8)
        x1_scale = torch.randn(m, dtype=torch.float32).abs() * 0.01
        x2_scale = torch.randn(n, dtype=torch.bfloat16).abs() * 0.01  # BF16 scale

        # NPU custom op
        npu_result = torch_npu.npu_quant_matmul_gelu(
            x1.npu(), x2.npu(), x1_scale.npu(), x2_scale.npu(), approximate="gelu_tanh")

        # Check output dtype is BF16
        self.assertEqual(npu_result.dtype, torch.bfloat16)
        self.assertEqual(npu_result.shape, (m, n))
        self.assertTrue(torch.isfinite(npu_result.float()).all())

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_gelu_batch(self):
        """Test with batch dimensions"""
        torch.manual_seed(0)
        batch, m, k, n = 4, 64, 128, 256

        x1 = torch.randint(-5, 5, (batch, m, k), dtype=torch.int8)
        x2 = torch.randint(-5, 5, (batch, k, n), dtype=torch.int8)
        # x1_scale: per-token scale, shape should be (m,) for all batches or (batch*m,) for per-batch-per-token
        # For simplicity, use same scale for all batches: (m,)
        x1_scale = torch.randn(m, dtype=torch.float32).abs() * 0.01
        x2_scale = torch.randn(n, dtype=torch.float32).abs() * 0.01

        # Process each batch separately for CPU reference
        cpu_results = []
        for i in range(batch):
            cpu_result = self.cpu_quant_matmul_gelu(
                x1[i], x2[i], x1_scale, x2_scale, approximate="gelu_tanh")
            cpu_results.append(cpu_result)
        cpu_result = torch.stack(cpu_results, dim=0)

        # NPU custom op should handle batch automatically
        npu_result = torch_npu.npu_quant_matmul_gelu(
            x1.npu(), x2.npu(),
            x1_scale.npu(),  # Per-token scale: (m,)
            x2_scale.npu(),
            approximate="gelu_tanh")

        self.assertEqual(npu_result.shape, (batch, m, n))
        self.assertTrue(torch.isfinite(npu_result.float()).all())

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_gelu_invalid_approximate(self):
        """Test with invalid approximate values"""
        m, k, n = 64, 128, 256

        x1 = torch.randint(-5, 5, (m, k), dtype=torch.int8)
        x2 = torch.randint(-5, 5, (k, n), dtype=torch.int8)
        x1_scale = torch.randn(m, dtype=torch.float32).abs() * 0.01
        x2_scale = torch.randn(n, dtype=torch.float32).abs() * 0.01

        # Test various invalid approximate values
        invalid_values = ["invalid_type", "gelu", "tanh", "erf", "gelu_relu", "", "none"]
        for invalid_val in invalid_values:
            with self.assertRaisesRegex(RuntimeError, "approximate must be 'gelu_tanh' or 'gelu_erf'"):
                torch_npu.npu_quant_matmul_gelu(
                    x1.npu(), x2.npu(), x1_scale.npu(), x2_scale.npu(), approximate=invalid_val)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_gelu_default_approximate(self):
        """Test with default approximate (gelu_erf)"""
        torch.manual_seed(0)
        m, k, n = 64, 128, 256

        x1 = torch.randint(-5, 5, (m, k), dtype=torch.int8)
        x2 = torch.randint(-5, 5, (k, n), dtype=torch.int8)
        x1_scale = torch.randn(m, dtype=torch.float32).abs() * 0.01
        x2_scale = torch.randn(n, dtype=torch.float32).abs() * 0.01

        # CPU reference with default gelu_erf
        cpu_result = self.cpu_quant_matmul_gelu(x1, x2, x1_scale, x2_scale, approximate="gelu_erf")

        # NPU custom op without specifying approximate (should use default "gelu_erf")
        npu_result = torch_npu.npu_quant_matmul_gelu(
            x1.npu(), x2.npu(), x1_scale.npu(), x2_scale.npu())

        self.assertRtolEqual(cpu_result.numpy(), npu_result.cpu().float().numpy(), 0.01)


if __name__ == "__main__":
    run_tests()

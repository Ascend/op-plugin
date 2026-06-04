import math
import unittest
import copy
import struct
from struct import pack, unpack
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices



class TestScaledMm(TestCase):

    # ========== 入参约束测试 ==========

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_invalid_mat_a_dtype(self):
        """测试 mat_a 不是 float8 类型时报错"""
        x1 = torch.rand(16, 32, dtype=torch.float16)  # 使用 float16 而不是 float8
        x2 = torch.rand(32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(1, dtype=torch.float32)
        scale_b = torch.rand(1, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm(x1.npu(), x2.npu(), scale_a.npu(), scale_b.npu(), out_dtype=torch.float32)
        self.assertTrue("mat_a must be float8 type" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_invalid_mat_b_dtype(self):
        """测试 mat_b 不是 float8 类型时报错"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        x2 = torch.rand(32, 64, dtype=torch.float16)  # 使用 float16 而不是 float8
        scale_a = torch.rand(1, dtype=torch.float32)
        scale_b = torch.rand(1, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm(x1.npu(), x2.npu(), scale_a.npu(), scale_b.npu(), out_dtype=torch.float32)
        self.assertTrue("mat_b must be float8 type" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_two_float8_e5m2_not_supported(self):
        """测试两个 float8_e5m2 矩阵相乘时报错"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e5m2)
        x2 = torch.rand(32, 64, dtype=torch.bfloat16).to(torch.float8_e5m2)
        scale_a = torch.rand(1, dtype=torch.float32)
        scale_b = torch.rand(1, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm(x1.npu(), x2.npu(), scale_a.npu(), scale_b.npu(), out_dtype=torch.float32)
        self.assertTrue("Multiplication of two Float8_e5m2 matrices is not supported" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_invalid_scale_a_dtype(self):
        """测试 scale_a 不是 float32 或 float8_e8m0 类型时报错"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        x2 = torch.rand(32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(1, dtype=torch.float16)  # 使用 float16 而不是 float32
        scale_b = torch.rand(1, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm(x1.npu(), x2.npu(), scale_a.npu(), scale_b.npu(), out_dtype=torch.float32)
        self.assertTrue("scale_a must be float32 or float8_e8m0 type" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_invalid_scale_b_dtype(self):
        """测试 scale_b 不是 float32 或 float8_e8m0 类型时报错"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        x2 = torch.rand(32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(1, dtype=torch.float32)
        scale_b = torch.rand(1, dtype=torch.float16)  # 使用 float16 而不是 float32
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm(x1.npu(), x2.npu(), scale_a.npu(), scale_b.npu(), out_dtype=torch.float32)
        self.assertTrue("scale_b must be float32 or float8_e8m0 type" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_invalid_out_dtype(self):
        """测试 out_dtype 不支持的类型时报错"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        x2 = torch.rand(32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(1, dtype=torch.float32)
        scale_b = torch.rand(1, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm(x1.npu(), x2.npu(), scale_a.npu(), scale_b.npu(), out_dtype=torch.int8)
        self.assertTrue("out_dtype must be Float32, BFloat16, or Float16" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_invalid_mat_a_dim(self):
        """测试 mat_a 不是 2 维时报错"""
        x1 = torch.rand(2, 16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 3维张量
        x2 = torch.rand(32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(1, dtype=torch.float32)
        scale_b = torch.rand(1, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm(x1.npu(), x2.npu(), scale_a.npu(), scale_b.npu(), out_dtype=torch.float32)
        self.assertTrue("mat_a must be a matrix" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_invalid_mat_b_dim(self):
        """测试 mat_b 不是 2 维时报错"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        x2 = torch.rand(2, 32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 3维张量
        scale_a = torch.rand(1, dtype=torch.float32)
        scale_b = torch.rand(1, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm(x1.npu(), x2.npu(), scale_a.npu(), scale_b.npu(), out_dtype=torch.float32)
        self.assertTrue("mat_b must be a matrix" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_invalid_shape_mismatch(self):
        """测试 mat_a 的列数与 mat_b 的行数不匹配时报错"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        x2 = torch.rand(64, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 行数是64, 不是32
        scale_a = torch.rand(1, dtype=torch.float32)
        scale_b = torch.rand(1, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm(x1.npu(), x2.npu(), scale_a.npu(), scale_b.npu(), out_dtype=torch.float32)
        self.assertTrue("shapes cannot be multiplied" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_invalid_scale_result(self):
        """测试传入 scale_result 时报错（当前不支持输出 float8 类型）"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        x2 = torch.rand(32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(1, dtype=torch.float32)
        scale_b = torch.rand(1, dtype=torch.float32)
        scale_result = torch.rand(1, dtype=torch.float32)  # 传入 scale_result
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm(x1.npu(), x2.npu(), scale_a.npu(), scale_b.npu(),
                           bias=None, scale_result=scale_result.npu(), out_dtype=torch.float32)
        self.assertTrue("scale_result is not supported currently" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_invalid_bias_size(self):
        """测试 bias 大小与 mat_b 的输出维度不匹配时报错"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        x2 = torch.rand(32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(1, dtype=torch.float32)
        scale_b = torch.rand(1, dtype=torch.float32)
        bias = torch.rand(32, dtype=torch.bfloat16)  # bias 大小应该是 64，但传入 32
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm(x1.npu(), x2.npu(), scale_a.npu(), scale_b.npu(),
                           bias=bias.npu(), out_dtype=torch.bfloat16)
        self.assertTrue("Bias must be size" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_bias_not_supported_with_float32_out(self):
        """测试 out_dtype 为 Float32 时传入 bias 报错"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        x2 = torch.rand(32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(1, dtype=torch.float32)
        scale_b = torch.rand(1, dtype=torch.float32)
        bias = torch.rand(64, dtype=torch.bfloat16)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm(x1.npu(), x2.npu(), scale_a.npu(), scale_b.npu(),
                           bias=bias.npu(), out_dtype=torch.float32)
        self.assertTrue("Bias is not supported when out_dtype is set to Float32" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_invalid_bias_dtype(self):
        """测试 bias 数据类型不是 BFloat16 或 Half 时报错"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        x2 = torch.rand(32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(1, dtype=torch.float32)
        scale_b = torch.rand(1, dtype=torch.float32)
        bias = torch.rand(64, dtype=torch.float32)  # bias 应该是 BFloat16 或 Half
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm(x1.npu(), x2.npu(), scale_a.npu(), scale_b.npu(),
                           bias=bias.npu(), out_dtype=torch.bfloat16)
        self.assertTrue("Bias must be BFloat16 or Half" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_bias_out_dtype_mismatch_bf16(self):
        """测试 out_dtype 为 BFloat16 但 bias 为 Float16 时报错"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        x2 = torch.rand(32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(1, dtype=torch.float32)
        scale_b = torch.rand(1, dtype=torch.float32)
        bias = torch.rand(64, dtype=torch.float16)  # out_dtype 为 bfloat16 时 bias 也应为 bfloat16
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm(x1.npu(), x2.npu(), scale_a.npu(), scale_b.npu(),
                           bias=bias.npu(), out_dtype=torch.bfloat16)
        self.assertTrue("Bias must be BFloat16 to compute" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_bias_out_dtype_mismatch_fp16(self):
        """测试 out_dtype 为 Float16 但 bias 为 BFloat16 时报错"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        x2 = torch.rand(32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(1, dtype=torch.float32)
        scale_b = torch.rand(1, dtype=torch.float32)
        bias = torch.rand(64, dtype=torch.bfloat16)  # out_dtype 为 float16 时 bias 也应为 float16
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm(x1.npu(), x2.npu(), scale_a.npu(), scale_b.npu(),
                           bias=bias.npu(), out_dtype=torch.float16)
        self.assertTrue("Bias must be Float16 to compute" in str(context.exception))





    # ========== FP8 Per-Token 功能测试 ==========

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_fp8_pertoken_basic(self):
        """测试用例1: FP8 Per-Token 基础功能测试 (M=8192, K=320, N=2560)"""
        M, K, N = 8192, 320, 2560
        seed = 42
        torch.manual_seed(seed)

        # 生成数据
        X_fp8 = torch.rand(M, K, dtype=torch.bfloat16).to(torch.float8_e4m3fn).npu()
        W_fp8 = torch.rand(N, K, dtype=torch.bfloat16).to(torch.float8_e4m3fn).t().npu()  # (K, N)
        pertoken_scale = torch.rand(M, dtype=torch.float32).unsqueeze(-1).npu()  # (M, 1)
        scale = torch.rand(N, dtype=torch.float32).unsqueeze(0).npu()  # (1, N)

        # NPU 计算
        output = torch._scaled_mm(X_fp8, W_fp8, pertoken_scale, scale, out_dtype=torch.bfloat16)

        # 验证输出形状和类型
        self.assertEqual(output.shape, (M, N))
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(output.device.type == 'npu')

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_fp8_pertoken_medium(self):
        """测试用例2: FP8 Per-Token 中等尺寸功能测试 (M=256, K=512, N=1024)"""
        M, K, N = 256, 512, 1024
        seed = 123
        torch.manual_seed(seed)

        # 生成数据
        X_fp8 = torch.rand(M, K, dtype=torch.bfloat16).to(torch.float8_e4m3fn).npu()
        W_fp8 = torch.rand(N, K, dtype=torch.bfloat16).to(torch.float8_e4m3fn).t().npu()  # (K, N)
        pertoken_scale = torch.rand(M, dtype=torch.float32).unsqueeze(-1).npu()  # (M, 1)
        scale = torch.rand(N, dtype=torch.float32).unsqueeze(0).npu()  # (1, N)

        # NPU 计算
        output = torch._scaled_mm(X_fp8, W_fp8, pertoken_scale, scale, out_dtype=torch.bfloat16)

        # 验证输出形状和类型
        self.assertEqual(output.shape, (M, N))
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(output.device.type == 'npu')

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_fp8_pertoken_large(self):
        """测试用例3: FP8 Per-Token 大尺寸功能测试 (M=4096, K=2048, N=4096)"""
        M, K, N = 4096, 2048, 4096
        seed = 456
        torch.manual_seed(seed)

        # 生成数据
        X_fp8 = torch.rand(M, K, dtype=torch.bfloat16).to(torch.float8_e4m3fn).npu()
        W_fp8 = torch.rand(N, K, dtype=torch.bfloat16).to(torch.float8_e4m3fn).t().npu()  # (K, N)
        pertoken_scale = torch.rand(M, dtype=torch.float32).unsqueeze(-1).npu()  # (M, 1)
        scale = torch.rand(N, dtype=torch.float32).unsqueeze(0).npu()  # (1, N)

        # NPU 计算
        output = torch._scaled_mm(X_fp8, W_fp8, pertoken_scale, scale, out_dtype=torch.bfloat16)

        # 验证输出形状和类型
        self.assertEqual(output.shape, (M, N))
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(output.device.type == 'npu')

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_fp8_pertoken_float16_output(self):
        """测试用例4: FP8 Per-Token Float16 输出功能测试"""
        M, K, N = 256, 512, 1024
        seed = 1001
        torch.manual_seed(seed)

        # 生成数据
        X_fp8 = torch.rand(M, K, dtype=torch.bfloat16).to(torch.float8_e4m3fn).npu()
        W_fp8 = torch.rand(N, K, dtype=torch.bfloat16).to(torch.float8_e4m3fn).t().npu()  # (K, N)
        pertoken_scale = torch.rand(M, dtype=torch.float32).unsqueeze(-1).npu()  # (M, 1)
        scale = torch.rand(N, dtype=torch.float32).unsqueeze(0).npu()  # (1, N)

        # NPU 计算（Float16 输出）
        output = torch._scaled_mm(X_fp8, W_fp8, pertoken_scale, scale, out_dtype=torch.float16)

        # 验证输出形状和类型
        self.assertEqual(output.shape, (M, N))
        self.assertEqual(output.dtype, torch.float16)
        self.assertTrue(output.device.type == 'npu')

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_fp8_pertoken_float32_output(self):
        """测试用例5: FP8 Per-Token Float32 输出功能测试"""
        M, K, N = 256, 512, 1024
        seed = 2002
        torch.manual_seed(seed)

        # 生成数据
        X_fp8 = torch.rand(M, K, dtype=torch.bfloat16).to(torch.float8_e4m3fn).npu()
        W_fp8 = torch.rand(N, K, dtype=torch.bfloat16).to(torch.float8_e4m3fn).t().npu()  # (K, N)
        pertoken_scale = torch.rand(M, dtype=torch.float32).unsqueeze(-1).npu()  # (M, 1)
        scale = torch.rand(N, dtype=torch.float32).unsqueeze(0).npu()  # (1, N)

        # NPU 计算（Float32 输出）
        output = torch._scaled_mm(X_fp8, W_fp8, pertoken_scale, scale, out_dtype=torch.float32)

        # 验证输出形状和类型
        self.assertEqual(output.shape, (M, N))
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(output.device.type == 'npu')


if __name__ == "__main__":
    run_tests()
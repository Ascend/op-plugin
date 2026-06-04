import math
import unittest
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestScaledMmV2(TestCase):

    # ========== 入参约束测试 ==========

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_v2_invalid_mat_a_dtype(self):
        """测试 mat_a 不是 float8 类型时报错"""
        x1 = torch.rand(16, 32, dtype=torch.float16)  # 使用 float16 而不是 float8
        x2 = torch.rand(32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = [torch.rand(1, dtype=torch.float32)]
        scale_b = [torch.rand(1, dtype=torch.float32)]
        scale_recipe_a = [0]  # ScalingType enum
        scale_recipe_b = [0]
        swizzle_a = []
        swizzle_b = []
        contraction_dim = []
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm_v2(x1.npu(), x2.npu(),
                               [s.npu() for s in scale_a], scale_recipe_a, swizzle_a,
                               [s.npu() for s in scale_b], scale_recipe_b, swizzle_b,
                               None, torch.float32, contraction_dim, False)
        self.assertTrue("mat_a must be float8 type" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_v2_invalid_mat_b_dtype(self):
        """测试 mat_b 不是 float8 类型时报错"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        x2 = torch.rand(32, 64, dtype=torch.float16)  # 使用 float16 而不是 float8
        scale_a = [torch.rand(1, dtype=torch.float32)]
        scale_b = [torch.rand(1, dtype=torch.float32)]
        scale_recipe_a = [0]
        scale_recipe_b = [0]
        swizzle_a = []
        swizzle_b = []
        contraction_dim = []
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm_v2(x1.npu(), x2.npu(),
                               [s.npu() for s in scale_a], scale_recipe_a, swizzle_a,
                               [s.npu() for s in scale_b], scale_recipe_b, swizzle_b,
                               None, torch.float32, contraction_dim, False)
        self.assertTrue("mat_b must be float8 type" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_v2_two_float8_e5m2_not_supported(self):
        """测试两个 float8_e5m2 矩阵相乘时报错"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e5m2)
        x2 = torch.rand(32, 64, dtype=torch.bfloat16).to(torch.float8_e5m2)
        scale_a = [torch.rand(1, dtype=torch.float32)]
        scale_b = [torch.rand(1, dtype=torch.float32)]
        scale_recipe_a = [0]
        scale_recipe_b = [0]
        swizzle_a = []
        swizzle_b = []
        contraction_dim = []
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm_v2(x1.npu(), x2.npu(),
                               [s.npu() for s in scale_a], scale_recipe_a, swizzle_a,
                               [s.npu() for s in scale_b], scale_recipe_b, swizzle_b,
                               None, torch.float32, contraction_dim, False)
        self.assertTrue("Multiplication of two Float8_e5m2 matrices is not supported" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_v2_invalid_scale_a_dtype(self):
        """测试 scale_a 不是 float32 或 float8_e8m0 类型时报错"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        x2 = torch.rand(32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = [torch.rand(1, dtype=torch.float16)]  # 使用 float16 而不是 float32
        scale_b = [torch.rand(1, dtype=torch.float32)]
        scale_recipe_a = [0]
        scale_recipe_b = [0]
        swizzle_a = []
        swizzle_b = []
        contraction_dim = []
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm_v2(x1.npu(), x2.npu(),
                               [s.npu() for s in scale_a], scale_recipe_a, swizzle_a,
                               [s.npu() for s in scale_b], scale_recipe_b, swizzle_b,
                               None, torch.float32, contraction_dim, False)
        self.assertTrue("scale_a must be float32 or float8_e8m0 type" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_v2_invalid_scale_b_dtype(self):
        """测试 scale_b 不是 float32 或 float8_e8m0 类型时报错"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        x2 = torch.rand(32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = [torch.rand(1, dtype=torch.float32)]
        scale_b = [torch.rand(1, dtype=torch.float16)]  # 使用 float16 而不是 float32
        scale_recipe_a = [0]
        scale_recipe_b = [0]
        swizzle_a = []
        swizzle_b = []
        contraction_dim = []
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm_v2(x1.npu(), x2.npu(),
                               [s.npu() for s in scale_a], scale_recipe_a, swizzle_a,
                               [s.npu() for s in scale_b], scale_recipe_b, swizzle_b,
                               None, torch.float32, contraction_dim, False)
        self.assertTrue("scale_b must be float32 or float8_e8m0 type" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_v2_invalid_out_dtype(self):
        """测试 out_dtype 不支持的类型时报错"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        x2 = torch.rand(32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = [torch.rand(1, dtype=torch.float32)]
        scale_b = [torch.rand(1, dtype=torch.float32)]
        scale_recipe_a = [0]
        scale_recipe_b = [0]
        swizzle_a = []
        swizzle_b = []
        contraction_dim = []
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm_v2(x1.npu(), x2.npu(),
                               [s.npu() for s in scale_a], scale_recipe_a, swizzle_a,
                               [s.npu() for s in scale_b], scale_recipe_b, swizzle_b,
                               None, torch.int8, contraction_dim, False)
        self.assertTrue("out_dtype must be Float32, BFloat16, or Float16" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_v2_invalid_mat_a_dim(self):
        """测试 mat_a 不是 2 维时报错"""
        x1 = torch.rand(2, 16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 3维张量
        x2 = torch.rand(32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = [torch.rand(1, dtype=torch.float32)]
        scale_b = [torch.rand(1, dtype=torch.float32)]
        scale_recipe_a = [0]
        scale_recipe_b = [0]
        swizzle_a = []
        swizzle_b = []
        contraction_dim = []
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm_v2(x1.npu(), x2.npu(),
                               [s.npu() for s in scale_a], scale_recipe_a, swizzle_a,
                               [s.npu() for s in scale_b], scale_recipe_b, swizzle_b,
                               None, torch.float32, contraction_dim, False)
        self.assertTrue("mat_a must be a matrix" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_v2_invalid_mat_b_dim(self):
        """测试 mat_b 不是 2 维时报错"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        x2 = torch.rand(2, 32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 3维张量
        scale_a = [torch.rand(1, dtype=torch.float32)]
        scale_b = [torch.rand(1, dtype=torch.float32)]
        scale_recipe_a = [0]
        scale_recipe_b = [0]
        swizzle_a = []
        swizzle_b = []
        contraction_dim = []
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm_v2(x1.npu(), x2.npu(),
                               [s.npu() for s in scale_a], scale_recipe_a, swizzle_a,
                               [s.npu() for s in scale_b], scale_recipe_b, swizzle_b,
                               None, torch.float32, contraction_dim, False)
        self.assertTrue("mat_b must be a matrix" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_v2_invalid_shape_mismatch(self):
        """测试 mat_a 的列数与 mat_b 的行数不匹配时报错"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        x2 = torch.rand(64, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 行数是64, 不是32
        scale_a = [torch.rand(1, dtype=torch.float32)]
        scale_b = [torch.rand(1, dtype=torch.float32)]
        scale_recipe_a = [0]
        scale_recipe_b = [0]
        swizzle_a = []
        swizzle_b = []
        contraction_dim = []
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm_v2(x1.npu(), x2.npu(),
                               [s.npu() for s in scale_a], scale_recipe_a, swizzle_a,
                               [s.npu() for s in scale_b], scale_recipe_b, swizzle_b,
                               None, torch.float32, contraction_dim, False)
        self.assertTrue("shapes cannot be multiplied" in str(context.exception))

    # ========== bias 相关测试 ==========

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_v2_invalid_bias_size(self):
        """测试 bias 大小与 mat_b 的输出维度不匹配时报错"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        x2 = torch.rand(32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = [torch.rand(1, dtype=torch.float32)]
        scale_b = [torch.rand(1, dtype=torch.float32)]
        scale_recipe_a = [0]
        scale_recipe_b = [0]
        swizzle_a = []
        swizzle_b = []
        contraction_dim = []
        bias = torch.rand(32, dtype=torch.bfloat16)  # bias 大小应该是 64，但传入 32
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm_v2(x1.npu(), x2.npu(),
                               [s.npu() for s in scale_a], scale_recipe_a, swizzle_a,
                               [s.npu() for s in scale_b], scale_recipe_b, swizzle_b,
                               bias.npu(), torch.bfloat16, contraction_dim, False)
        self.assertTrue("Bias must be size" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_v2_bias_not_supported_with_float32_out(self):
        """测试 out_dtype 为 Float32 时传入 bias 报错"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        x2 = torch.rand(32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = [torch.rand(1, dtype=torch.float32)]
        scale_b = [torch.rand(1, dtype=torch.float32)]
        scale_recipe_a = [0]
        scale_recipe_b = [0]
        swizzle_a = []
        swizzle_b = []
        contraction_dim = []
        bias = torch.rand(64, dtype=torch.bfloat16)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm_v2(x1.npu(), x2.npu(),
                               [s.npu() for s in scale_a], scale_recipe_a, swizzle_a,
                               [s.npu() for s in scale_b], scale_recipe_b, swizzle_b,
                               bias.npu(), torch.float32, contraction_dim, False)
        self.assertTrue("Bias is not supported when out_dtype is set to Float32" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_v2_invalid_bias_dtype(self):
        """测试 bias 数据类型不是 BFloat16 或 Half 时报错"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        x2 = torch.rand(32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = [torch.rand(1, dtype=torch.float32)]
        scale_b = [torch.rand(1, dtype=torch.float32)]
        scale_recipe_a = [0]
        scale_recipe_b = [0]
        swizzle_a = []
        swizzle_b = []
        contraction_dim = []
        bias = torch.rand(64, dtype=torch.float32)  # bias 应该是 BFloat16 或 Half
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm_v2(x1.npu(), x2.npu(),
                               [s.npu() for s in scale_a], scale_recipe_a, swizzle_a,
                               [s.npu() for s in scale_b], scale_recipe_b, swizzle_b,
                               bias.npu(), torch.bfloat16, contraction_dim, False)
        self.assertTrue("Bias must be BFloat16 or Half" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_v2_bias_out_dtype_mismatch_bf16(self):
        """测试 out_dtype 为 BFloat16 但 bias 为 Float16 时报错"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        x2 = torch.rand(32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = [torch.rand(1, dtype=torch.float32)]
        scale_b = [torch.rand(1, dtype=torch.float32)]
        scale_recipe_a = [0]
        scale_recipe_b = [0]
        swizzle_a = []
        swizzle_b = []
        contraction_dim = []
        bias = torch.rand(64, dtype=torch.float16)  # out_dtype 为 bfloat16 时 bias 也应为 bfloat16
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm_v2(x1.npu(), x2.npu(),
                               [s.npu() for s in scale_a], scale_recipe_a, swizzle_a,
                               [s.npu() for s in scale_b], scale_recipe_b, swizzle_b,
                               bias.npu(), torch.bfloat16, contraction_dim, False)
        self.assertTrue("Bias must be BFloat16 to compute" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_v2_bias_out_dtype_mismatch_fp16(self):
        """测试 out_dtype 为 Float16 但 bias 为 BFloat16 时报错"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        x2 = torch.rand(32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = [torch.rand(1, dtype=torch.float32)]
        scale_b = [torch.rand(1, dtype=torch.float32)]
        scale_recipe_a = [0]
        scale_recipe_b = [0]
        swizzle_a = []
        swizzle_b = []
        contraction_dim = []
        bias = torch.rand(64, dtype=torch.bfloat16)  # out_dtype 为 float16 时 bias 也应为 float16
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm_v2(x1.npu(), x2.npu(),
                               [s.npu() for s in scale_a], scale_recipe_a, swizzle_a,
                               [s.npu() for s in scale_b], scale_recipe_b, swizzle_b,
                               bias.npu(), torch.float16, contraction_dim, False)
        self.assertTrue("Bias must be Float16 to compute" in str(context.exception))

    # ========== contraction_dim 相关测试 ==========

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_v2_contraction_dim_invalid_length(self):
        """测试 contraction_dim 不是 2 个元素时报错"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        x2 = torch.rand(32, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = [torch.rand(1, dtype=torch.float32)]
        scale_b = [torch.rand(1, dtype=torch.float32)]
        scale_recipe_a = [0]
        scale_recipe_b = [0]
        swizzle_a = []
        swizzle_b = []
        contraction_dim = [0]  # 应该有2个元素
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm_v2(x1.npu(), x2.npu(),
                               [s.npu() for s in scale_a], scale_recipe_a, swizzle_a,
                               [s.npu() for s in scale_b], scale_recipe_b, swizzle_b,
                               None, torch.float32, contraction_dim, False)
        self.assertTrue("contraction_dim must have exactly 2 elements" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_v2_contraction_dim_shape_mismatch(self):
        """测试 contraction_dim 指定的维度大小不匹配时报错"""
        x1 = torch.rand(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        x2 = torch.rand(64, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = [torch.rand(1, dtype=torch.float32)]
        scale_b = [torch.rand(1, dtype=torch.float32)]
        scale_recipe_a = [0]
        scale_recipe_b = [0]
        swizzle_a = []
        swizzle_b = []
        contraction_dim = [1, 0]  # x1[1]=32, x2[0]=64, 不匹配
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_mm_v2(x1.npu(), x2.npu(),
                               [s.npu() for s in scale_a], scale_recipe_a, swizzle_a,
                               [s.npu() for s in scale_b], scale_recipe_b, swizzle_b,
                               None, torch.float32, contraction_dim, False)
        self.assertTrue("shapes cannot be multiplied" in str(context.exception))


    # ========== FP8 Per-Token 功能测试 ==========

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_v2_fp8_pertoken_basic(self):
        """测试用例1: FP8 Per-Token 基础功能测试 (M=8192, K=320, N=2560)"""
        M, K, N = 8192, 320, 2560
        seed = 42
        torch.manual_seed(seed)

        # 生成数据
        X_fp8 = torch.rand(M, K, dtype=torch.bfloat16).to(torch.float8_e4m3fn).npu()
        W_fp8 = torch.rand(N, K, dtype=torch.bfloat16).to(torch.float8_e4m3fn).t().npu()  # (K, N)
        scale_a = [torch.rand(M, dtype=torch.float32).npu()]
        scale_b = [torch.rand(N, dtype=torch.float32).npu()]
        scale_recipe_a = [0]  # ScalingType enum
        scale_recipe_b = [0]
        swizzle_a = []
        swizzle_b = []
        contraction_dim = []

        # NPU 计算
        output = torch._scaled_mm_v2(X_fp8, W_fp8,
                                     scale_a, scale_recipe_a, swizzle_a,
                                     scale_b, scale_recipe_b, swizzle_b,
                                     None, torch.bfloat16, contraction_dim, False)

        # 验证输出形状和类型
        self.assertEqual(output.shape, (M, N))
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(output.device.type == 'npu')

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_v2_fp8_pertoken_medium(self):
        """测试用例2: FP8 Per-Token 中等尺寸功能测试 (M=256, K=512, N=1024)"""
        M, K, N = 256, 512, 1024
        seed = 123
        torch.manual_seed(seed)

        # 生成数据
        X_fp8 = torch.rand(M, K, dtype=torch.bfloat16).to(torch.float8_e4m3fn).npu()
        W_fp8 = torch.rand(N, K, dtype=torch.bfloat16).to(torch.float8_e4m3fn).t().npu()  # (K, N)
        scale_a = [torch.rand(M, dtype=torch.float32).npu()]
        scale_b = [torch.rand(N, dtype=torch.float32).npu()]
        scale_recipe_a = [0]
        scale_recipe_b = [0]
        swizzle_a = []
        swizzle_b = []
        contraction_dim = []

        # NPU 计算
        output = torch._scaled_mm_v2(X_fp8, W_fp8,
                                     scale_a, scale_recipe_a, swizzle_a,
                                     scale_b, scale_recipe_b, swizzle_b,
                                     None, torch.bfloat16, contraction_dim, False)

        # 验证输出形状和类型
        self.assertEqual(output.shape, (M, N))
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(output.device.type == 'npu')

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_v2_fp8_pertoken_large(self):
        """测试用例3: FP8 Per-Token 大尺寸功能测试 (M=4096, K=2048, N=4096)"""
        M, K, N = 4096, 2048, 4096
        seed = 456
        torch.manual_seed(seed)

        # 生成数据
        X_fp8 = torch.rand(M, K, dtype=torch.bfloat16).to(torch.float8_e4m3fn).npu()
        W_fp8 = torch.rand(N, K, dtype=torch.bfloat16).to(torch.float8_e4m3fn).t().npu()  # (K, N)
        scale_a = [torch.rand(M, dtype=torch.float32).npu()]
        scale_b = [torch.rand(N, dtype=torch.float32).npu()]
        scale_recipe_a = [0]
        scale_recipe_b = [0]
        swizzle_a = []
        swizzle_b = []
        contraction_dim = []

        # NPU 计算
        output = torch._scaled_mm_v2(X_fp8, W_fp8,
                                     scale_a, scale_recipe_a, swizzle_a,
                                     scale_b, scale_recipe_b, swizzle_b,
                                     None, torch.bfloat16, contraction_dim, False)

        # 验证输出形状和类型
        self.assertEqual(output.shape, (M, N))
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(output.device.type == 'npu')


    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_v2_fp8_pertoken_float16_output(self):
        """测试用例4: FP8 Per-Token Float16 输出功能测试"""
        M, K, N = 256, 512, 1024
        seed = 1001
        torch.manual_seed(seed)

        # 生成数据
        X_fp8 = torch.rand(M, K, dtype=torch.bfloat16).to(torch.float8_e4m3fn).npu()
        W_fp8 = torch.rand(N, K, dtype=torch.bfloat16).to(torch.float8_e4m3fn).t().npu()  # (K, N)
        scale_a = [torch.rand(M, dtype=torch.float32).npu()]
        scale_b = [torch.rand(N, dtype=torch.float32).npu()]
        scale_recipe_a = [0]
        scale_recipe_b = [0]
        swizzle_a = []
        swizzle_b = []
        contraction_dim = []

        # NPU 计算（Float16 输出）
        output = torch._scaled_mm_v2(X_fp8, W_fp8,
                                     scale_a, scale_recipe_a, swizzle_a,
                                     scale_b, scale_recipe_b, swizzle_b,
                                     None, torch.float16, contraction_dim, False)

        # 验证输出形状和类型
        self.assertEqual(output.shape, (M, N))
        self.assertEqual(output.dtype, torch.float16)
        self.assertTrue(output.device.type == 'npu')

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_mm_v2_fp8_pertoken_float32_output(self):
        """测试用例5: FP8 Per-Token Float32 输出功能测试"""
        M, K, N = 256, 512, 1024
        seed = 2002
        torch.manual_seed(seed)

        # 生成数据
        X_fp8 = torch.rand(M, K, dtype=torch.bfloat16).to(torch.float8_e4m3fn).npu()
        W_fp8 = torch.rand(N, K, dtype=torch.bfloat16).to(torch.float8_e4m3fn).t().npu()  # (K, N)
        scale_a = [torch.rand(M, dtype=torch.float32).npu()]
        scale_b = [torch.rand(N, dtype=torch.float32).npu()]
        scale_recipe_a = [0]
        scale_recipe_b = [0]
        swizzle_a = []
        swizzle_b = []
        contraction_dim = []

        # NPU 计算（Float32 输出）
        output = torch._scaled_mm_v2(X_fp8, W_fp8,
                                     scale_a, scale_recipe_a, swizzle_a,
                                     scale_b, scale_recipe_b, swizzle_b,
                                     None, torch.float32, contraction_dim, False)

        # 验证输出形状和类型
        self.assertEqual(output.shape, (M, N))
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(output.device.type == 'npu')


if __name__ == "__main__":
    run_tests()
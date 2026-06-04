import unittest
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestScaledGroupedMm(TestCase):

    # ========== 基本功能测试 ==========
    # 注意：由于 split_item 固定为 IN_NOT_SPLIT_OUT_SPLIT(2)，
    # group_type 不能为 DEFAULT_SPLIT(-1)，因此基本功能测试使用 M_SPLIT 场景

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_basic_function_2d_a_3d_b(self):
        """测试基本功能 - 2D mat_a 和 3D mat_b (M_SPLIT 场景)"""
        mat_a = torch.rand(256, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 2D [M, K]
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 3D [G, K, N]
        scale_a = torch.rand(256, dtype=torch.float32)  # KC mode: (M,)
        scale_b = torch.rand(4, 32, dtype=torch.float32)  # KC mode: (G, N)
        offs = torch.tensor([64, 128, 192, 256], dtype=torch.int32)  # offsets for 2D mat_a
        result = torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                          offs.npu(), None, None, torch.bfloat16, False)
        self.assertEqual(result.shape, (256, 32))

    # ========== 维度约束测试 ==========

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_invalid_mat_a_dim_1d(self):
        """测试 mat_a 为 1D 时报错 """
        mat_a = torch.rand(128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(128, dtype=torch.float32)
        scale_b = torch.rand(4, 64, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     None, None, None, torch.bfloat16, False)
        self.assertTrue("mat_a dimension must be 2D or 3D" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_invalid_mat_a_dim_4d(self):
        """测试 mat_a 为 4D 时报错 """
        mat_a = torch.rand(4, 2, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 2, 128, dtype=torch.float32)
        scale_b = torch.rand(4, 32, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     None, None, None, torch.bfloat16, False)
        self.assertTrue("mat_a dimension must be 2D or 3D" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_invalid_mat_b_dim_1d(self):
        """测试 mat_b 为 1D 时报错 """
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 128, dtype=torch.float32)
        scale_b = torch.rand(4, 1, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     None, None, None, torch.bfloat16, False)
        self.assertTrue("mat_b dimension must be 2D or 3D" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_invalid_mat_b_dim_4d(self):
        """测试 mat_b 为 4D 时报错 """
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(4, 2, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 128, dtype=torch.float32)
        scale_b = torch.rand(4, 2, 32, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     None, None, None, torch.bfloat16, False)
        self.assertTrue("mat_b dimension must be 2D or 3D" in str(context.exception))

    # ========== Scale维度约束测试 ==========

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_invalid_scale_a_dtype(self):
        """测试 scale_a 数据类型不支持时报错"""
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 128, dtype=torch.bfloat16)  # 不支持的dtype
        scale_b = torch.rand(4, 32, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     None, None, None, torch.bfloat16, False)
        self.assertTrue("scale_a must be float32 or float8_e8m0fnu" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_invalid_scale_b_dtype(self):
        """测试 scale_b 数据类型不支持时报错"""
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 128, dtype=torch.float32)
        scale_b = torch.rand(4, 32, dtype=torch.bfloat16)  # 不支持的dtype
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     None, None, None, torch.bfloat16, False)
        self.assertTrue("scale_b must be float32 or float8_e8m0fnu" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_invalid_scale_a_dim_fp8(self):
        """测试 FP8 模式下 scale_a 维度不正确时报错 """
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 128, 1, dtype=torch.float32)  # 3D for fp8 is invalid
        scale_b = torch.rand(4, 32, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     None, None, None, torch.bfloat16, False)
        self.assertTrue("scale_a dimension must be 1D or 2D for fp8" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_invalid_scale_b_dim_fp8(self):
        """测试 FP8 模式下 scale_b 维度不正确时报错 """
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 128, dtype=torch.float32)
        scale_b = torch.rand(4, 32, 1, dtype=torch.float32)  # 3D for fp8 is invalid
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     None, None, None, torch.bfloat16, False)
        self.assertTrue("scale_b dimension must be 1D or 2D for fp8" in str(context.exception))

    # ========== 收缩维度约束测试 ==========

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_contraction_dim_mismatch(self):
        """测试 mat_a 和 mat_b 的收缩维度不匹配时报错 """
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # K=64
        mat_b = torch.rand(4, 128, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # K=128
        scale_a = torch.rand(4, 128, dtype=torch.float32)
        scale_b = torch.rand(4, 32, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     None, None, None, torch.bfloat16, False)
        self.assertTrue("contraction dimension mismatch between mat_a and mat_b" in str(context.exception))

    # ========== 不支持特性测试 ==========

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_bias_not_supported(self):
        """测试传入 bias 时报错 """
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 128, dtype=torch.float32)
        scale_b = torch.rand(4, 32, dtype=torch.float32)
        bias = torch.rand(4, 32, dtype=torch.bfloat16)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     None, bias.npu(), None, torch.bfloat16, False)
        self.assertTrue("NPU _scaled_grouped_mm does not support bias yet" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_scale_result_not_supported(self):
        """测试传入 scale_result 时报错 """
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 128, dtype=torch.float32)
        scale_b = torch.rand(4, 32, dtype=torch.float32)
        scale_result = torch.rand(4, 32, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     None, None, scale_result.npu(), torch.bfloat16, False)
        self.assertTrue("NPU _scaled_grouped_mm does not support scale_result yet" in str(context.exception))

    # ========== Offsets约束测试 ==========

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_missing_offsets_for_2d_mat_a(self):
        """测试 mat_a 为 2D 但未提供 offs 时报错 """
        mat_a = torch.rand(256, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 2D
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 3D
        scale_a = torch.rand(256, dtype=torch.float32)
        scale_b = torch.rand(4, 32, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     None, None, None, torch.bfloat16, False)
        self.assertTrue("offsets required when using 2D input tensor" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_missing_offsets_for_2d_mat_b(self):
        """测试 mat_b 为 2D 但未提供 offs 时报错 """
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 3D
        mat_b = torch.rand(64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 2D
        scale_a = torch.rand(4, 128, dtype=torch.float32)
        scale_b = torch.rand(32, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     None, None, None, torch.bfloat16, False)
        self.assertTrue("offsets required when using 2D input tensor" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_invalid_offs_dim(self):
        """测试 offs 不是 1D 时报错 """
        mat_a = torch.rand(256, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 2D
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 3D
        scale_a = torch.rand(256, dtype=torch.float32)
        scale_b = torch.rand(4, 32, dtype=torch.float32)
        offs = torch.tensor([[256]], dtype=torch.int32)  # 2D tensor
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     offs.npu(), None, None, torch.bfloat16, False)
        self.assertTrue("offsets tensor must be 1D" in str(context.exception))

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_invalid_offs_dtype(self):
        """测试 offs 不是 int32 时报错 """
        mat_a = torch.rand(256, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 2D
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 3D
        scale_a = torch.rand(256, dtype=torch.float32)
        scale_b = torch.rand(4, 32, dtype=torch.float32)
        offs = torch.tensor([256], dtype=torch.int64)  # int64 instead of int32
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     offs.npu(), None, None, torch.bfloat16, False)
        self.assertTrue("offsets data type must be int32" in str(context.exception))

    # ========== 输出类型约束测试 ==========

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_invalid_out_dtype(self):
        """测试 out_dtype 不是 bfloat16 时报错 """
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        mat_b = torch.rand(4, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_a = torch.rand(4, 128, dtype=torch.float32)
        scale_b = torch.rand(4, 32, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     None, None, None, torch.float32, False)
        self.assertTrue("_scaled_grouped_mm on NPU only supports BF16 output type" in str(context.exception))

    # ========== K_SPLIT 不支持测试 ==========

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_k_split_not_supported_single_weight(self):
        """测试 mat_b 只有 1 个 weight 时触发 K_SPLIT 报错 """
        mat_a = torch.rand(4, 128, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 4 groups
        mat_b = torch.rand(1, 64, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # 1 weight shared
        scale_a = torch.rand(4, 128, dtype=torch.float32)
        scale_b = torch.rand(1, 32, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as context:
            torch._scaled_grouped_mm(mat_a.npu(), mat_b.npu(), scale_a.npu(), scale_b.npu(),
                                     None, None, None, torch.bfloat16, False)
        self.assertTrue("K_SPLIT (group_type=2) is not supported yet" in str(context.exception))

    # ========== FP8 K-C 量化功能测试 (pertoken-perchannel, scale 为 fp32) ==========

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_fp8_kc_type_m_e1(self):
        """测试用例: TypeM (M分组), E=1, 1个group - FP8 K-C量化 (pertoken-perchannel)

        X: (128, 256), M=128, K=256
        W: (1, 256, 64), E=1, K=256, N=64
        scale_a (pertoken): (128,)
        scale_b (perchannel): (1, 64)
        offs: [128]
        输出: (128, 64)
        """
        M, K, N = 128, 256, 64
        E = 1
        seed = 42
        torch.manual_seed(seed)

        # 生成 FP8 数据
        X_fp8 = torch.rand(M, K, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        W_fp8 = torch.rand(E, K, N, dtype=torch.bfloat16).to(torch.float8_e4m3fn)

        # scale: pertoken (M,) 和 perchannel (E, N)
        scale_a = torch.rand(M, dtype=torch.float32)  # pertoken
        scale_b = torch.rand(E, N, dtype=torch.float32)  # perchannel

        # offs: [M]
        offs = torch.tensor([M], dtype=torch.int32)

        # NPU 计算
        output = torch._scaled_grouped_mm(
            X_fp8.npu(), W_fp8.npu(),
            scale_a.npu(), scale_b.npu(),
            offs.npu(), None, None,
            torch.bfloat16, False
        )

        # 验证输出形状和类型
        if isinstance(output, (list, tuple)):
            output = output[0]
        self.assertEqual(output.shape, (M, N))
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(output.device.type == 'npu')

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_fp8_kc_type_m_e2(self):
        """测试用例: TypeM (M分组), E=2, 2个group - FP8 K-C量化 (pertoken-perchannel)

        Group 0: M1=128, K=256, N=64
        Group 1: M2=128, K=256, N=64
        X: (256, 256), M_total=256
        W: (2, 256, 64), E=2
        scale_a (pertoken): (256,)
        scale_b (perchannel): (2, 64)
        offs: [128, 256]
        输出: (256, 64)
        """
        group_m = [128, 128]
        E = len(group_m)
        M_total = sum(group_m)
        K, N = 256, 64
        seed = 123
        torch.manual_seed(seed)

        # 生成 FP8 数据
        X_fp8 = torch.rand(M_total, K, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        W_fp8 = torch.rand(E, K, N, dtype=torch.bfloat16).to(torch.float8_e4m3fn)

        # scale: pertoken (M_total,) 和 perchannel (E, N)
        scale_a = torch.rand(M_total, dtype=torch.float32)  # pertoken
        scale_b = torch.rand(E, N, dtype=torch.float32)  # perchannel

        # offs: [M1, M_total]
        offs = torch.tensor([group_m[0], M_total], dtype=torch.int32)

        # NPU 计算
        output = torch._scaled_grouped_mm(
            X_fp8.npu(), W_fp8.npu(),
            scale_a.npu(), scale_b.npu(),
            offs.npu(), None, None,
            torch.bfloat16, False
        )

        # 验证输出形状和类型
        if isinstance(output, (list, tuple)):
            output = output[0]
        self.assertEqual(output.shape, (M_total, N))
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(output.device.type == 'npu')

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_fp8_kc_type_m_e4(self):
        """测试用例: TypeM (M分组), E=4, 4个group - FP8 K-C量化 (pertoken-perchannel)

        Group 0-3: M=64 each, K=256, N=128
        X: (256, 256), M_total=256
        W: (4, 256, 128), E=4
        scale_a (pertoken): (256,)
        scale_b (perchannel): (4, 128)
        offs: [64, 128, 192, 256]
        输出: (256, 128)
        """
        group_m = [64, 64, 64, 64]
        E = len(group_m)
        M_total = sum(group_m)
        K, N = 256, 128
        seed = 456
        torch.manual_seed(seed)

        # 生成 FP8 数据
        X_fp8 = torch.rand(M_total, K, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        W_fp8 = torch.rand(E, K, N, dtype=torch.bfloat16).to(torch.float8_e4m3fn)

        # scale: pertoken (M_total,) 和 perchannel (E, N)
        scale_a = torch.rand(M_total, dtype=torch.float32)  # pertoken
        scale_b = torch.rand(E, N, dtype=torch.float32)  # perchannel

        # offs: cumsum of group_m
        import itertools
        offs = torch.tensor(list(itertools.accumulate(group_m)), dtype=torch.int32)

        # NPU 计算
        output = torch._scaled_grouped_mm(
            X_fp8.npu(), W_fp8.npu(),
            scale_a.npu(), scale_b.npu(),
            offs.npu(), None, None,
            torch.bfloat16, False
        )

        # 验证输出形状和类型
        if isinstance(output, (list, tuple)):
            output = output[0]
        self.assertEqual(output.shape, (M_total, N))
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(output.device.type == 'npu')

    # ========== MXFP8 功能测试 (scale 为 float8_e8m0fnu) ==========

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_mxfp8_type_m_e1(self):
        """测试用例: TypeM (M分组), E=1, 1个group - MXFP8 (scale 为 float8_e8m0fnu)

        X: (128, 128), M=128, K=128
        W: (1, 128, 64), E=1, K=128, N=64
        scale_a (pertoken): (128, num_blocks/2, 2) = (128, 2, 2)
        scale_b (perchannel): (num_blocks/2, N, 2) = (2, 64, 2)  (E=1时squeeze)
        offs: [128]
        输出: (128, 64)
        """
        M, K, N = 128, 128, 64
        E = 1
        num_blocks_half = K // 64  # K/32/2 = K/64
        seed = 42
        torch.manual_seed(seed)

        # 生成 FP8 数据
        X_fp8 = torch.rand(M, K, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        W_fp8 = torch.rand(E, K, N, dtype=torch.bfloat16).to(torch.float8_e4m3fn)

        # scale: MXFP8 格式的 scale (float8_e8m0fnu)
        # scale_a (pertoken): (M, num_blocks/2, 2)
        scale_a = torch.rand(M, num_blocks_half, 2, dtype=torch.bfloat16).to(torch.float8_e8m0fnu)
        # scale_b (perchannel): (E, num_blocks/2, N, 2) -> E=1 时 squeeze 为 (num_blocks/2, N, 2)
        scale_b = torch.rand(num_blocks_half, N, 2, dtype=torch.bfloat16).to(torch.float8_e8m0fnu)

        # offs: [M]
        offs = torch.tensor([M], dtype=torch.int32)

        # NPU 计算
        output = torch._scaled_grouped_mm(
            X_fp8.npu(), W_fp8.npu(),
            scale_a.npu(), scale_b.npu(),
            offs.npu(), None, None,
            torch.bfloat16, False
        )

        # 验证输出形状和类型
        if isinstance(output, (list, tuple)):
            output = output[0]
        self.assertEqual(output.shape, (M, N))
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(output.device.type == 'npu')

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_mxfp8_type_m_e2(self):
        """测试用例: TypeM (M分组), E=2, 2个group - MXFP8 (scale 为 float8_e8m0fnu)

        Group 0: M1=128, K=128, N=64
        Group 1: M2=128, K=128, N=64
        X: (256, 128), M_total=256
        W: (2, 128, 64), E=2
        scale_a (pertoken): (256, num_blocks/2, 2) = (256, 2, 2)
        scale_b (perchannel): (E, num_blocks/2, N, 2) = (2, 2, 64, 2)
        offs: [128, 256]
        输出: (256, 64)
        """
        group_m = [128, 128]
        E = len(group_m)
        M_total = sum(group_m)
        K, N = 128, 64
        num_blocks_half = K // 64  # K/32/2 = K/64
        seed = 456
        torch.manual_seed(seed)

        # 生成 FP8 数据
        X_fp8 = torch.rand(M_total, K, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        W_fp8 = torch.rand(E, K, N, dtype=torch.bfloat16).to(torch.float8_e4m3fn)

        # scale: MXFP8 格式的 scale (float8_e8m0fnu)
        # scale_a (pertoken): (M_total, num_blocks/2, 2)
        scale_a = torch.rand(M_total, num_blocks_half, 2, dtype=torch.bfloat16).to(torch.float8_e8m0fnu)
        # scale_b (perchannel): (E, num_blocks/2, N, 2)
        scale_b = torch.rand(E, num_blocks_half, N, 2, dtype=torch.bfloat16).to(torch.float8_e8m0fnu)

        # offs: [M1, M_total]
        offs = torch.tensor([group_m[0], M_total], dtype=torch.int32)

        # NPU 计算
        output = torch._scaled_grouped_mm(
            X_fp8.npu(), W_fp8.npu(),
            scale_a.npu(), scale_b.npu(),
            offs.npu(), None, None,
            torch.bfloat16, False
        )

        # 验证输出形状和类型
        if isinstance(output, (list, tuple)):
            output = output[0]
        self.assertEqual(output.shape, (M_total, N))
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(output.device.type == 'npu')

    @SupportedDevices(['Ascend950'])
    def test_npu_scaled_grouped_mm_mxfp8_type_m_multi_groups(self):
        """测试用例: TypeM (M分组), E=6, 多个group - MXFP8 (scale 为 float8_e8m0fnu)

        group_m = [128, 128, 256, 128, 512, 6]
        X: (1138, 128), M_total=1138
        W: (6, 128, 64), E=6
        scale_a (pertoken): (1138, num_blocks/2, 2)
        scale_b (perchannel): (6, num_blocks/2, N, 2)
        offs: cumsum of group_m
        输出: (1138, 64)
        """
        group_m = [128, 128, 256, 128, 512, 6]
        E = len(group_m)
        M_total = sum(group_m)
        K, N = 128, 64
        num_blocks_half = K // 64  # K/32/2 = K/64
        seed = 789
        torch.manual_seed(seed)

        # 生成 FP8 数据
        X_fp8 = torch.rand(M_total, K, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        W_fp8 = torch.rand(E, K, N, dtype=torch.bfloat16).to(torch.float8_e4m3fn)

        # scale: MXFP8 格式的 scale (float8_e8m0fnu)
        # scale_a (pertoken): (M_total, num_blocks/2, 2)
        scale_a = torch.rand(M_total, num_blocks_half, 2, dtype=torch.bfloat16).to(torch.float8_e8m0fnu)
        # scale_b (perchannel): (E, num_blocks/2, N, 2)
        scale_b = torch.rand(E, num_blocks_half, N, 2, dtype=torch.bfloat16).to(torch.float8_e8m0fnu)

        # offs: cumsum of group_m
        import itertools
        offs = torch.tensor(list(itertools.accumulate(group_m)), dtype=torch.int32)

        # NPU 计算
        output = torch._scaled_grouped_mm(
            X_fp8.npu(), W_fp8.npu(),
            scale_a.npu(), scale_b.npu(),
            offs.npu(), None, None,
            torch.bfloat16, False
        )

        # 验证输出形状和类型
        if isinstance(output, (list, tuple)):
            output = output[0]
        self.assertEqual(output.shape, (M_total, N))
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(output.device.type == 'npu')


if __name__ == "__main__":
    run_tests()
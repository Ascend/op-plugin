import torch
import torch_npu

from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests

FRACTAL_NZ_C0_16 = 50


class TestNpuWeightQuantPreprocess(TestCase):
    @SupportedDevices(['Ascend950'])
    def test_npu_weight_quant_preprocess_a8w4(self):
        k, n = 128, 64
        weight = torch.zeros((n, k), dtype=torch.uint8).npu().transpose(0, 1)
        weight_scale = torch.zeros((n, k // 64, 2), dtype=torch.uint8).npu().transpose(0, 1)

        out_weight, out_weight_scale, _, _ = torch_npu.npu_weight_quant_preprocess(
            weight,
            weight_scale,
            x_dtype=torch.float8_e4m3fn,
            weight_dtype=torch_npu.float4_e2m1fn_x2,
            weight_scale_dtype=torch_npu.float8_e8m0fnu,
            x_scale_dtype=torch_npu.float8_e8m0fnu,
            k_group_size=32)

        self.assertEqual(out_weight.shape, weight.shape)
        self.assertEqual(out_weight.stride(), weight.stride())
        self.assertEqual(out_weight.dtype, weight.dtype)
        self.assertEqual(out_weight_scale.shape, weight_scale.shape)
        self.assertEqual(out_weight_scale.stride(), weight_scale.stride())
        self.assertEqual(out_weight_scale.dtype, weight_scale.dtype)
        self.assertEqual(out_weight.cpu(), weight.cpu())
        self.assertEqual(out_weight_scale.cpu(), weight_scale.cpu())
        if torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT") == b"enable":
            self.assertEqual(torch_npu.get_npu_format(out_weight), FRACTAL_NZ_C0_16)

    @SupportedDevices(['Ascend950'])
    def test_npu_weight_quant_preprocess_a16mxf4_trans_nd(self):
        # A16MXF4 转置：weight 物理 {N, K/2} 沿 K 打包、视图 {K/2, N} strides [1, K/2]；
        # scale E8M0 转置视图 {G, N} strides [1, G]（G=K/32）。转置走 ND 直拷物理透传，
        # 输出与输入共享内容，format 保持 ND
        k, n = 256, 128
        g = k // 32
        weight = torch.zeros((n, k // 2), dtype=torch.uint8).npu().transpose(0, 1)
        weight_scale = torch.zeros((n, g), dtype=torch.uint8).view(
            torch.float8_e8m0fnu).npu().transpose(0, 1)

        out_weight, out_weight_scale, _, _ = torch_npu.npu_weight_quant_preprocess(
            weight,
            weight_scale,
            x_dtype=torch.float16,
            weight_dtype=torch_npu.float4_e2m1fn_x2,
            weight_scale_dtype=torch_npu.float8_e8m0fnu,
            k_group_size=32)

        self.assertEqual(out_weight.shape, weight.shape)
        self.assertEqual(out_weight.stride(), weight.stride())
        self.assertEqual(out_weight.dtype, weight.dtype)
        self.assertEqual(out_weight_scale.shape, (g, n))
        self.assertEqual(out_weight_scale.stride(), weight_scale.stride())
        self.assertEqual(out_weight.cpu(), weight.cpu())
        # e8m0 转置视图直接 .cpu() 会走 AICPU Transpose（不支持该 dtype），按 uint8 回读比对
        self.assertEqual(out_weight_scale.view(torch.uint8).cpu(), weight_scale.view(torch.uint8).cpu())
        # ND 直拷透传：out_weight format 与输入一致（ND；返回值类型随 ALLOW_INTERNAL_FORMAT 开关变化）
        self.assertEqual(torch_npu.get_npu_format(out_weight), torch_npu.get_npu_format(weight))

    @SupportedDevices(['Ascend950'])
    def test_npu_weight_quant_preprocess_f8f4_tcg_nz(self):
        # 非转置: weight uint8 {K, N/2} 沿 N 打包, scale FP16 {G, N}(G=K/32>1);
        # ND->NZ 输出 uint8 视图挂假标签 C0_16(字节口径), 物理为 C0=32 个 FP4(16B) 的 NZ 分形, N 块在前
        k, n = 256, 128
        g = k // 32
        weight = torch.zeros((k, n // 2), dtype=torch.uint8).npu()
        weight_scale = torch.zeros((g, n), dtype=torch.float16).npu()

        out_weight, out_weight_scale, _, _ = torch_npu.npu_weight_quant_preprocess(
            weight,
            weight_scale,
            x_dtype=torch.float8_e4m3fn,
            weight_dtype=torch_npu.float4_e2m1fn_x2,
            weight_scale_dtype=torch.float16,
            k_group_size=32)

        self.assertEqual(out_weight.shape, weight.shape)
        self.assertEqual(out_weight.dtype, weight.dtype)
        self.assertEqual(out_weight_scale.shape, weight_scale.shape)
        self.assertEqual(out_weight_scale.cpu(), weight_scale.cpu())
        # 注：FRACTAL_NZ_C0_16 fake format 的 uint8 out_weight 不支持 format-aware .cpu() 回读
        # （copy_ 不支持 internal format），物理布局 bit-wise 校验由 aclnn STC 覆盖
        if torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT") == b"enable":
            self.assertEqual(torch_npu.get_npu_format(out_weight), FRACTAL_NZ_C0_16)

    @SupportedDevices(['Ascend950'])
    def test_npu_weight_quant_preprocess_f8f4_tcg_trans_nd(self):
        # F8F4 T-CG 转置：weight 物理 {N, K/2} 沿 K 打包、视图 {K/2, N} strides [1, K/2]；
        # scale FP16 转置视图 {G, N} strides [1, G]。转置走 ND 直拷物理透传，format 保持 ND
        k, n = 256, 128
        g = k // 32
        weight = torch.zeros((n, k // 2), dtype=torch.uint8).npu().transpose(0, 1)
        weight_scale = torch.zeros((n, g), dtype=torch.float16).npu().transpose(0, 1)

        out_weight, out_weight_scale, _, _ = torch_npu.npu_weight_quant_preprocess(
            weight,
            weight_scale,
            x_dtype=torch.float8_e4m3fn,
            weight_dtype=torch_npu.float4_e2m1fn_x2,
            weight_scale_dtype=torch.float16,
            k_group_size=32)

        self.assertEqual(out_weight.shape, weight.shape)
        self.assertEqual(out_weight.stride(), weight.stride())
        self.assertEqual(out_weight_scale.shape, (g, n))
        self.assertEqual(out_weight_scale.stride(), weight_scale.stride())
        self.assertEqual(out_weight.cpu(), weight.cpu())
        self.assertEqual(out_weight_scale.cpu(), weight_scale.cpu())
        self.assertEqual(torch_npu.get_npu_format(out_weight), torch_npu.get_npu_format(weight))

    @SupportedDevices(['Ascend950'])
    def test_npu_weight_quant_preprocess_a16w4_int8_carrier_rejected(self):
        # A16W4 的 4-bit weight 只接受 uint8 载体（每字节 2 个 4-bit），int8 载体应被拒绝
        k, n = 256, 128
        weight = torch.zeros((k, n // 2), dtype=torch.int8).npu()
        weight_scale = torch.ones((n,), dtype=torch.float16).npu()

        with self.assertRaisesRegex(RuntimeError, "must be packed into a uint8 tensor"):
            torch_npu.npu_weight_quant_preprocess(
                weight,
                weight_scale,
                x_dtype=torch.float16,
                weight_dtype=torch_npu.int4,
                weight_scale_dtype=torch.float16)

    @SupportedDevices(['Ascend950'])
    def test_npu_weight_quant_preprocess_a16int8_per_tensor(self):
        # A16INT8 pertensor：转置/非转置均 ND 直拷
        k, n = 256, 128
        weight = torch.randint(low=-128, high=128, size=(k, n), dtype=torch.int8).npu()
        weight_scale = torch.randn((1,), dtype=torch.float16).npu()

        out_weight, out_weight_scale, out_weight_offset, out_bias = torch_npu.npu_weight_quant_preprocess(
            weight,
            weight_scale,
            x_dtype=torch.float16,
            weight_dtype=torch.int8,
            weight_scale_dtype=torch.float16)

        self.assertEqual(out_weight.shape, weight.shape)
        self.assertEqual(out_weight.stride(), weight.stride())
        self.assertEqual(out_weight.dtype, weight.dtype)
        self.assertEqual(out_weight.cpu(), weight.cpu())
        self.assertEqual(out_weight_scale.cpu(), weight_scale.cpu())

    @SupportedDevices(['Ascend950'])
    def test_npu_weight_quant_preprocess_a16int8_per_tensor_trans(self):
        # A16INT8 pertensor 转置：ND 直拷（转置视图透传）
        k, n = 256, 128
        weight = torch.randint(low=-128, high=128, size=(n, k), dtype=torch.int8).npu().transpose(0, 1)
        weight_scale = torch.randn((1,), dtype=torch.float16).npu()

        out_weight, out_weight_scale, out_weight_offset, out_bias = torch_npu.npu_weight_quant_preprocess(
            weight,
            weight_scale,
            x_dtype=torch.float16,
            weight_dtype=torch.int8,
            weight_scale_dtype=torch.float16)

        self.assertEqual(out_weight.shape, weight.shape)
        self.assertEqual(out_weight.stride(), weight.stride())
        self.assertEqual(out_weight.cpu(), weight.cpu())

    @SupportedDevices(['Ascend950'])
    def test_npu_weight_quant_preprocess_a16int8_per_channel_nd(self):
        # A16INT8 perchannel 非转置：ND 直拷透传（A16W8 全场景不做 NZ 转换）
        k, n = 256, 128
        weight = torch.randint(low=-128, high=128, size=(k, n), dtype=torch.int8).npu()
        weight_scale = torch.randn((n,), dtype=torch.float16).npu()
        weight_offset = torch.randn((n,), dtype=torch.float16).npu()

        out_weight, out_weight_scale, out_weight_offset, out_bias = torch_npu.npu_weight_quant_preprocess(
            weight,
            weight_scale,
            x_dtype=torch.float16,
            weight_dtype=torch.int8,
            weight_scale_dtype=torch.float16,
            weight_offset=weight_offset)

        self.assertEqual(out_weight.shape, weight.shape)
        self.assertEqual(out_weight_scale.cpu(), weight_scale.cpu())
        self.assertEqual(out_weight_offset.cpu(), weight_offset.cpu())
        # ND 直拷：物理元素数与 weight 一致
        self.assertEqual(out_weight.numel(), weight.numel())
        self.assertEqual(out_weight.cpu().view(torch.int8).flatten(), weight.cpu().flatten())

    @SupportedDevices(['Ascend950'])
    def test_npu_weight_quant_preprocess_a16int8_per_channel_trans(self):
        # A16INT8 perchannel 转置：ND 直拷
        k, n = 256, 128
        weight = torch.randint(low=-128, high=128, size=(n, k), dtype=torch.int8).npu().transpose(0, 1)
        weight_scale = torch.randn((n,), dtype=torch.float16).npu()

        out_weight, out_weight_scale, out_weight_offset, out_bias = torch_npu.npu_weight_quant_preprocess(
            weight,
            weight_scale,
            x_dtype=torch.float16,
            weight_dtype=torch.int8,
            weight_scale_dtype=torch.float16)

        self.assertEqual(out_weight.shape, weight.shape)
        self.assertEqual(out_weight.stride(), weight.stride())
        self.assertEqual(out_weight.cpu(), weight.cpu())

    @SupportedDevices(['Ascend950'])
    def test_npu_weight_quant_preprocess_a16int8_per_group(self):
        # A16INT8 pergroup：转置/非转置均 ND 直拷
        k, n = 256, 128
        k_group_size = 64
        g = (k + k_group_size - 1) // k_group_size
        weight = torch.randint(low=-128, high=128, size=(k, n), dtype=torch.int8).npu()
        weight_scale = torch.randn((g, n), dtype=torch.float16).npu()

        out_weight, out_weight_scale, out_weight_offset, out_bias = torch_npu.npu_weight_quant_preprocess(
            weight,
            weight_scale,
            x_dtype=torch.float16,
            weight_dtype=torch.int8,
            weight_scale_dtype=torch.float16,
            k_group_size=k_group_size)

        self.assertEqual(out_weight.shape, weight.shape)
        self.assertEqual(out_weight.stride(), weight.stride())
        self.assertEqual(out_weight.cpu(), weight.cpu())
        self.assertEqual(out_weight_scale.cpu(), weight_scale.cpu())

    @SupportedDevices(['Ascend950'])
    def test_npu_weight_quant_preprocess_a16int8_per_group_trans(self):
        # A16INT8 pergroup 转置：ND 直拷（转置视图透传）
        k, n = 256, 128
        k_group_size = 64
        g = (k + k_group_size - 1) // k_group_size
        weight = torch.randint(low=-128, high=128, size=(n, k), dtype=torch.int8).npu().transpose(0, 1)
        weight_scale = torch.randn((g, n), dtype=torch.float16)
        weight_scale = weight_scale.t().contiguous().t().npu()

        out_weight, out_weight_scale, out_weight_offset, out_bias = torch_npu.npu_weight_quant_preprocess(
            weight,
            weight_scale,
            x_dtype=torch.float16,
            weight_dtype=torch.int8,
            weight_scale_dtype=torch.float16,
            k_group_size=k_group_size)

        self.assertEqual(out_weight.shape, weight.shape)
        self.assertEqual(out_weight.stride(), weight.stride())
        self.assertEqual(out_weight.cpu(), weight.cpu())

    @SupportedDevices(['Ascend950'])
    def test_npu_weight_quant_preprocess_a16fp8hif8_e4m3_per_channel(self):
        # A16FP8HIF8 E4M3FN perchannel 非转置：ND 直拷
        k, n = 256, 128
        weight = torch.randint(0, 255, (k, n), dtype=torch.uint8).to(torch.float8_e4m3fn).npu()
        weight_scale = torch.randn((n,), dtype=torch.float16).npu()

        out_weight, out_weight_scale, out_weight_offset, out_bias = torch_npu.npu_weight_quant_preprocess(
            weight,
            weight_scale,
            x_dtype=torch.float16,
            weight_dtype=torch.float8_e4m3fn,
            weight_scale_dtype=torch.float16)

        self.assertEqual(out_weight.shape, weight.shape)
        self.assertEqual(out_weight.stride(), weight.stride())
        self.assertEqual(out_weight.dtype, weight.dtype)
        # FP8 dtype 无 sub 实现，经 uint8 视图逐元素比较
        self.assertEqual(out_weight.view(torch.uint8).cpu(), weight.view(torch.uint8).cpu())
        self.assertEqual(out_weight_scale.cpu(), weight_scale.cpu())

    @SupportedDevices(['Ascend950'])
    def test_npu_weight_quant_preprocess_a16fp8hif8_e4m3_per_channel_trans(self):
        # A16FP8HIF8 E4M3FN perchannel 转置：ND 直拷
        k, n = 256, 128
        weight = torch.randint(0, 255, (n, k), dtype=torch.uint8).to(torch.float8_e4m3fn).npu().transpose(0, 1)
        weight_scale = torch.randn((n,), dtype=torch.float16).npu()

        out_weight, out_weight_scale, out_weight_offset, out_bias = torch_npu.npu_weight_quant_preprocess(
            weight,
            weight_scale,
            x_dtype=torch.float16,
            weight_dtype=torch.float8_e4m3fn,
            weight_scale_dtype=torch.float16)

        self.assertEqual(out_weight.shape, weight.shape)
        self.assertEqual(out_weight.stride(), weight.stride())
        self.assertEqual(out_weight.dtype, weight.dtype)
        # FP8 dtype 无 sub 实现，经 uint8 视图逐元素比较
        self.assertEqual(out_weight.view(torch.uint8).cpu(), weight.view(torch.uint8).cpu())


if __name__ == "__main__":
    run_tests()

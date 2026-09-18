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


if __name__ == "__main__":
    run_tests()

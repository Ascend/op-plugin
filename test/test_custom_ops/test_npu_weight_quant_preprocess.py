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


if __name__ == "__main__":
    run_tests()

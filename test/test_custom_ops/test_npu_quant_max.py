import unittest

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices
from torch.testing import assert_close


class TestQuantMax(TestCase):
    def supported_op_exec(self, x, scale, dst_dtype):
        # CPU fp32 golden：y = round(x / scale) 转 dst_dtype，amax = |x|.max()
        x_fp32 = x.detach().float().cpu()
        scale_fp32 = scale.detach().float().cpu()
        amax = x_fp32.abs().max().reshape(1).to(x.dtype)
        y_fp32 = x_fp32 / scale_fp32
        if dst_dtype == torch_npu.float8_e5m2:
            y = y_fp32.to(torch.float8_e5m2)
        elif dst_dtype == torch_npu.float8_e4m3fn:
            y = y_fp32.to(torch.float8_e4m3fn)
        else:
            y = y_fp32.to(torch.uint8)
        return y, amax

    def custom_op_exec(self, x, scale, round_mode, dst_dtype):
        return torch_npu.npu_quant_max(x, scale, round_mode=round_mode, dst_dtype=dst_dtype)

    @SupportedDevices(['Ascend950'])
    def test_npu_quant_max_fp16_e5m2(self, device="npu"):
        x = torch.randn(8, 256, dtype=torch.float16).to(device)
        scale = torch.tensor([0.5], dtype=torch.float32).to(device)
        custom_y, custom_amax = self.custom_op_exec(
            x.clone(), scale.clone(), round_mode="rint", dst_dtype=torch_npu.float8_e5m2)
        self.assertEqual(custom_y.shape, x.shape)
        self.assertEqual(custom_amax.shape, torch.Size([1]))
        self.assertEqual(custom_amax.dtype, x.dtype)

    @SupportedDevices(['Ascend950'])
    def test_npu_quant_max_bf16_e4m3fn(self, device="npu"):
        x = torch.randn(4, 16, 128, dtype=torch.bfloat16).to(device)
        scale = torch.tensor([1.0], dtype=torch.float32).to(device)
        custom_y, custom_amax = self.custom_op_exec(
            x.clone(), scale.clone(), round_mode="rint", dst_dtype=torch_npu.float8_e4m3fn)
        self.assertEqual(custom_y.shape, x.shape)
        self.assertEqual(custom_amax.shape, torch.Size([1]))
        self.assertEqual(custom_amax.dtype, x.dtype)

    @SupportedDevices(['Ascend950'])
    def test_npu_quant_max_fp32_hifloat8(self, device="npu"):
        x = torch.randn(2, 64, dtype=torch.float32).to(device)
        scale = torch.tensor([2.0], dtype=torch.float32).to(device)
        custom_y, custom_amax = self.custom_op_exec(
            x.clone(), scale.clone(), round_mode="round", dst_dtype=torch_npu.hifloat8)
        self.assertEqual(custom_y.shape, x.shape)
        self.assertEqual(custom_amax.shape, torch.Size([1]))
        self.assertEqual(custom_amax.dtype, x.dtype)


if __name__ == "__main__":
    run_tests()

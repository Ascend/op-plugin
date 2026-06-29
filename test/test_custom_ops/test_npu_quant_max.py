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

    def group_supported_op_exec(self, x, scales, group_list, dst_dtype):
        # CPU fp32 golden：按 group_list 切分后逐组 y = round(x / scale) 转 dst_dtype，amax[g] = |x_g|.max()
        x_fp32 = x.detach().float().cpu()
        scales_fp32 = scales.detach().float().cpu()
        group_list_cpu = group_list.detach().cpu()
        num_groups = group_list_cpu.shape[0]
        y_parts = []
        amax_parts = []

        for g in range(num_groups):
            start = 0 if g == 0 else group_list_cpu[g - 1].item()
            end = group_list_cpu[g].item()
            if start == end:
                amax_parts.append(torch.zeros(1, dtype=x.dtype))
                continue

            x_g = x_fp32[start:end] if x.dim() == 2 else x_fp32[start:end, :, :]
            y_g = x_g / scales_fp32[g]
            if dst_dtype == torch_npu.float8_e5m2:
                y_g = y_g.to(torch.float8_e5m2)
            elif dst_dtype == torch_npu.float8_e4m3fn:
                y_g = y_g.to(torch.float8_e4m3fn)
            else:
                y_g = y_g.to(torch.uint8)
            y_parts.append(y_g)
            amax_g = x_g.abs().max().reshape(1).to(x.dtype)
            amax_parts.append(amax_g)

        if x.dim() == 2:
            y = torch.cat(y_parts, dim=0)
        else:
            y = torch.cat(y_parts, dim=0).reshape(x.shape)
        amax = torch.cat(amax_parts, dim=0)
        return y, amax

    def group_custom_op_exec(self, x, scales, group_list, round_mode, dst_dtype):
        return torch_npu.npu_quant_max(
            x, scales, round_mode=round_mode, dst_dtype=dst_dtype, group_list=group_list)

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

    @SupportedDevices(['Ascend950'])
    def test_npu_quant_max_group_fp16_e5m2_2d(self, device="npu"):
        x = torch.randn(8, 256, dtype=torch.float16).to(device)
        scales = torch.tensor([0.5, 1.0, 0.25, 2.0], dtype=torch.float32).to(device)
        group_list = torch.tensor([2, 4, 6, 8], dtype=torch.int64).to(device)
        custom_y, custom_amax = self.group_custom_op_exec(
            x.clone(), scales.clone(), group_list.clone(),
            round_mode="rint", dst_dtype=torch_npu.float8_e5m2)
        self.assertEqual(custom_y.shape, x.shape)
        self.assertEqual(custom_amax.shape, torch.Size([4]))
        self.assertEqual(custom_amax.dtype, x.dtype)

    @SupportedDevices(['Ascend950'])
    def test_npu_quant_max_group_bf16_e4m3fn_3d(self, device="npu"):
        x = torch.randn(4, 16, 128, dtype=torch.bfloat16).to(device)
        scales = torch.tensor([1.0, 0.5], dtype=torch.float32).to(device)
        group_list = torch.tensor([2, 4], dtype=torch.int64).to(device)
        custom_y, custom_amax = self.group_custom_op_exec(
            x.clone(), scales.clone(), group_list.clone(),
            round_mode="rint", dst_dtype=torch_npu.float8_e4m3fn)
        self.assertEqual(custom_y.shape, x.shape)
        self.assertEqual(custom_amax.shape, torch.Size([2]))
        self.assertEqual(custom_amax.dtype, x.dtype)

    @SupportedDevices(['Ascend950'])
    def test_npu_quant_max_group_fp32_hifloat8(self, device="npu"):
        x = torch.randn(2, 64, dtype=torch.float32).to(device)
        scales = torch.tensor([2.0], dtype=torch.float32).to(device)
        group_list = torch.tensor([2], dtype=torch.int64).to(device)
        custom_y, custom_amax = self.group_custom_op_exec(
            x.clone(), scales.clone(), group_list.clone(),
            round_mode="round", dst_dtype=torch_npu.hifloat8)
        self.assertEqual(custom_y.shape, x.shape)
        self.assertEqual(custom_amax.shape, torch.Size([1]))
        self.assertEqual(custom_amax.dtype, x.dtype)

if __name__ == "__main__":
    run_tests()

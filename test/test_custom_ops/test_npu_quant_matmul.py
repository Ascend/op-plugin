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


def quantize_w_per_group_asymmetric(w_fp32: torch.Tensor, group_size: int = 256):
    K, N = w_fp32.shape
    q_min, q_max = -7, 7  # int4
    w_fp32_grouped = w_fp32.view(-1, group_size, N)
    min_vals = torch.min(w_fp32_grouped, dim=1, keepdim=True)[0]
    max_vals = torch.max(w_fp32_grouped, dim=1, keepdim=True)[0]
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1e-5
    w_scales = range_vals / (q_max - q_min)
    w_zeros_fp = torch.round(q_min - min_vals / w_scales)
    w_quant_fp = torch.round(w_fp32_grouped / w_scales) + w_zeros_fp
    w_quant_grouped = torch.clamp(w_quant_fp, q_min, q_max).to(torch.int8)
    w_quant = w_quant_grouped.view(K, N)
    w_scales = w_scales.squeeze(1) / 15.0
    w_zeros = w_zeros_fp.squeeze(1).to(torch.int8)
    return w_quant, w_scales, w_zeros

def unpack_int4_lsb_first_signed(w_packed: torch.Tensor) -> torch.Tensor:
    assert w_packed.dtype == torch.int32, "Input tensor must be of dtype torch.int32"
    N, M = w_packed.shape
    shifts = torch.arange(0, 32, 4, device=w_packed.device).view(1, 1, 8)
    unpacked = (w_packed.unsqueeze(-1).cpu() >> shifts.cpu()) & 0xF
    unpacked = unpacked.to(torch.int8)
    unpacked_unsigned = unpacked.view(N, -1).to(w_packed.device)
    unpacked_signed = torch.where(unpacked_unsigned > 7, unpacked_unsigned - 16, unpacked_unsigned)
    return unpacked_signed.to(torch.int8)

def cpu_quant_matmul_a4w4_pergroup(x, weight, x1scale, x2scale, x2offset, groupsize=256):
    x = unpack_int4_lsb_first_signed(x).to(torch.int32)
    weight = unpack_int4_lsb_first_signed(weight).T.to(torch.int32)
    m, k = x.shape
    k, n = weight.shape
    golden_tensor = torch.zeros((m, n), dtype=torch.float32, device=x.device)
    for kstart in range(0, k, groupsize):
        # y = x1_scale[m, 1] * x2_scale[1, n] * (x1[m, groupsize] @ x2[groupsize, n]  +  rowsum(x1)[m, 1] * x2offset[1, n])
        kend = kstart + groupsize if kstart + groupsize < k else k
        subBlockX = x[:, kstart:kend]
        subBlockWeight = weight[kstart:kend, :]
        subBlockWeight = subBlockWeight - x2offset[int(kstart//groupsize)].int()
        mm_result = torch.matmul(subBlockX.cpu(), subBlockWeight.cpu()).float().npu()
        mm_result *= x2scale[int(kstart//groupsize)]
        golden_tensor += mm_result
    return golden_tensor.float() * x1scale

class TestQuantMatmul(TestCase):

    @SupportedDevices(['Ascend910B', 'Ascend950'])
    def test_npu_quant_matmul_a8w8(self):
        torch.manual_seed(0)
        x1 = torch.randint(-5, 5, (8192, 320), dtype=torch.int8)
        x2 = torch.randint(-5, 5, (320, 2560), dtype=torch.int8)
        x1_clone = x1.clone()
        x2_clone = x2.clone()
        scale = torch.randn(1, dtype=torch.float32)
        scale_quant = torch_npu.npu_trans_quant_param(scale.npu(), None)
        supported_output = torch.matmul(x1.to(torch.int32), x2.to(torch.int32)) * scale
        custom_output = torch_npu.npu_quant_matmul(x1_clone.npu(), x2_clone.npu(), scale_quant.npu(), output_dtype=torch.float16)
        self.assertRtolEqual(supported_output.float().cpu().numpy(), custom_output.float().cpu().numpy(), 0.01)

    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_a4w4(self):
        torch.manual_seed(0)
        x1 = torch.randint(-5, 5, (8192, 320), dtype=torch.int32)
        x2 = torch.randint(-5, 5, (320, 2560), dtype=torch.int32)
        x1_clone = x1.clone().float().npu()
        x2_clone = x2.clone().float().npu()
        scale = torch.randn(1, dtype=torch.float32)
        scale_clone = scale.clone().npu()
        scale_quant = torch_npu.npu_trans_quant_param(scale_clone, None)

        # convert int32 to int4*8
        scale_tmp = scale.clone().npu()
        scale_tmp[0] = 1
        x1_clone = torch_npu.npu_quantize(x1_clone, scale_tmp, None, torch.quint4x2, -1, False)
        x2_clone = torch_npu.npu_quantize(x2_clone, scale_tmp, None, torch.quint4x2, -1, False)

        supported_output = torch.matmul(x1.to(torch.int32), x2.to(torch.int32)) * scale
        custom_output = torch_npu.npu_quant_matmul(x1_clone, x2_clone, scale_quant, output_dtype=torch.float16)
        self.assertRtolEqual(supported_output.float().cpu().numpy(), custom_output.float().cpu().numpy(), 0.01)

    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_a4w4_pertoken(self):
        torch.manual_seed(0)
        x1 = torch.randint(-5, 5, (8192, 320), dtype=torch.int32)
        x2 = torch.randint(-5, 5, (320, 2560), dtype=torch.int32)
        x1_clone = x1.clone().float().npu()
        x2_clone = x2.clone().float().npu()
        scale = torch.randn(1, dtype=torch.float32)
        pertoken_scale = torch.randn(8192, dtype=torch.float32)
        scale_clone = scale.clone().npu()
        pertoken_scale_clone = pertoken_scale.clone().npu()

        # convert int32 to int4*8
        scale_tmp = scale.clone().npu()
        scale_tmp[0] = 1
        x1_clone = torch_npu.npu_quantize(x1_clone, scale_tmp, None, torch.quint4x2, -1, False)
        x2_clone = torch_npu.npu_quantize(x2_clone, scale_tmp, None, torch.quint4x2, -1, False)

        supported_output = torch.matmul(x1.to(torch.int32), x2.to(torch.int32)) * scale * pertoken_scale.unsqueeze(-1)
        custom_output_bf16 = torch_npu.npu_quant_matmul(x1_clone, x2_clone, scale_clone, pertoken_scale=pertoken_scale_clone, output_dtype=torch.bfloat16)
        custom_output_fp16 = torch_npu.npu_quant_matmul(x1_clone, x2_clone, scale_clone, pertoken_scale=pertoken_scale_clone, output_dtype=torch.float16)
        self.assertRtolEqual(supported_output.float().cpu().numpy(), custom_output_bf16.float().cpu().numpy(), 0.01)
        self.assertRtolEqual(supported_output.float().cpu().numpy(), custom_output_fp16.float().cpu().numpy(), 0.01)

    def gen_inputs_for_npu_quant_matmul_a4w4_pergroup(self, dtype):
        x1 = torch.randint(-5, 5, (256, 2048), dtype=dtype, device="npu")
        x2 = torch.randint(-5, 5, (5120, 2048), dtype=dtype, device="npu")
        x1, x1_scale = torch_npu.npu_dynamic_quant(x1, dst_type=torch.quint4x2)
        x2, x2_scale, x2_offset = quantize_w_per_group_asymmetric(x2.transpose(-1, -2))
        x2_offset = x2_offset.to(torch.float16)
        x2_scale = x2_scale.to(torch.float32)
        x2 = torch_npu.npu_convert_weight_to_int4pack(x2.to(torch.int32).transpose(-1, -2).contiguous()).transpose(-1, -2)
        return x1, x2, x1_scale.reshape(-1, 1), x2_scale, x2_offset

    @unittest.skip("skip test_npu_quant_matmul_a4w4_pergroup")
    def test_npu_quant_matmul_a4w4_pergroup(self):
        torch.manual_seed(0)
        x1, x2, x1_scale, x2_scale, x2_offset = self.gen_inputs_for_npu_quant_matmul_a4w4_pergroup(torch.bfloat16)
        supported_output_bf16 = cpu_quant_matmul_a4w4_pergroup(x1, x2.T, x1_scale, x2_scale, x2_offset)
        custom_output_bf16 = torch_npu.npu_quant_matmul(x1, x2, scale=x2_scale, offset=x2_offset, pertoken_scale=x1_scale, bias=None, output_dtype=torch.bfloat16, group_sizes=[1, 1, 256])
        self.assertRtolEqual(custom_output_bf16.float().cpu().numpy(), supported_output_bf16.float().cpu().numpy(), 0.01)
        x1, x2, x1_scale, x2_scale, x2_offset = self.gen_inputs_for_npu_quant_matmul_a4w4_pergroup(torch.float16)
        supported_output_fp16 = cpu_quant_matmul_a4w4_pergroup(x1, x2.T, x1_scale, x2_scale, x2_offset)
        custom_output_fp16 = torch_npu.npu_quant_matmul(x1, x2, scale=x2_scale, offset=x2_offset, pertoken_scale=x1_scale, bias=None, output_dtype=torch.float16, group_sizes=[1, 1, 256])
        self.assertRtolEqual(custom_output_fp16.float().cpu().numpy(), supported_output_fp16.float().cpu().numpy(), 0.01)

    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_continuous_x2_tensor(self):
        torch.manual_seed(0)
        x1 = torch.randint(-5, 5, (5, 160), dtype=torch.int32)
        x2 = torch.randint(-5, 5, (80, 160), dtype=torch.int32)
        x1_clone = x1.clone().float().npu()
        x2_clone = x2.clone().float().npu()
        scale = torch.randn(1, dtype=torch.float32)
        scale_clone = scale.clone().npu()
        scale_quant = torch_npu.npu_trans_quant_param(scale_clone, None)

        # convert int32 to int4*8
        scale_tmp = scale.clone().npu()
        scale_tmp[0] = 1
        x1_clone = torch_npu.npu_quantize(x1_clone, scale_tmp, None, torch.quint4x2, -1, False)
        x2_clone = torch_npu.npu_quantize(x2_clone, scale_tmp, None, torch.quint4x2, -1, False)

        supported_output = torch.matmul(x1.to(torch.int32), x2.t().to(torch.int32)) * scale
        custom_output = torch_npu.npu_quant_matmul(x1_clone, x2_clone.t(), scale_quant, output_dtype=torch.float16)
        self.assertRtolEqual(supported_output.float().cpu().numpy(), custom_output.float().cpu().numpy(), 0.01)

    @SupportedDevices(['Ascend910B', 'Ascend950'])
    def test_npu_quant_matmul_bf16(self):
        x1 = torch.randint(-5, 5, (8192, 320), dtype=torch.int8)
        x2 = torch.randint(-5, 5, (320, 2560), dtype=torch.int8)
        x1_clone = x1.clone()
        x2_clone = x2.clone()
        scale = torch.randn(1, dtype=torch.float32)
        supported_output = torch.matmul(x1.to(torch.int32), x2.to(torch.int32)) * scale
        custom_output = torch_npu.npu_quant_matmul(x1_clone.npu(), x2_clone.npu(), scale.npu(), output_dtype=torch.bfloat16)
        self.assertRtolEqual(supported_output.float().cpu().numpy(), custom_output.float().cpu().numpy(), 0.01)

    @SupportedDevices(['Ascend910B', 'Ascend950'])
    def test_npu_quant_matmul_bf16_nz(self):
        x1 = torch.randint(-5, 5, (8192, 320), dtype=torch.int8)
        x2 = torch.randint(-5, 5, (320, 2560), dtype=torch.int8)
        x1_clone = x1.clone()
        x2_clone = x2.clone()
        scale = torch.randn(1, dtype=torch.float32)
        supported_output = torch.matmul(x1.to(torch.int32), x2.to(torch.int32)) * scale
        x2_nz = torch_npu.npu_format_cast(x2_clone.npu().contiguous(), 29)
        custom_output = torch_npu.npu_quant_matmul(x1_clone.npu(), x2_nz.npu(), scale.npu(), output_dtype=torch.bfloat16)
        self.assertRtolEqual(supported_output.float().cpu().numpy(), custom_output.float().cpu().numpy(), 0.01)

    @SupportedDevices(['Ascend910B', 'Ascend950'])
    def test_npu_quant_matmul_fp16_nz(self):
        x1 = torch.randint(-5, 5, (8192, 320), dtype=torch.int8)
        x2 = torch.randint(-5, 5, (320, 2560), dtype=torch.int8)
        x1_clone = x1.clone()
        x2_clone = x2.clone()
        scale = torch.randn(1, dtype=torch.float32)
        supported_output = torch.matmul(x1.to(torch.int32), x2.to(torch.int32)) * scale
        x2_nz = torch_npu.npu_format_cast(x2_clone.npu().contiguous(), 29)
        custom_output = torch_npu.npu_quant_matmul(x1_clone.npu(), x2_nz.npu(), scale.npu(), output_dtype=torch.float16)
        self.assertRtolEqual(supported_output.float().cpu().numpy(), custom_output.float().cpu().numpy(), 0.01)

    @unittest.skip("skip test_npu_quant_matmul_int32")
    def test_npu_quant_matmul_int32(self):
        x1 = torch.randint(-5, 5, (16, 6656), dtype=torch.int8)
        x2 = torch.randint(-5, 5, (6656, 4992), dtype=torch.int8)
        x1_clone = x1.clone()
        x2_clone = x2.clone()
        scale = torch.randn(1, dtype=torch.float32)
        supported_output = torch.matmul(x1.to(torch.int32), x2.to(torch.int32))
        custom_output = torch_npu.npu_quant_matmul(x1_clone.npu(), x2_clone.npu(), scale.npu(), output_dtype=torch.int32)
        self.assertRtolEqual(supported_output.cpu().numpy().astype(np.float32), custom_output.cpu().numpy().astype(np.float32), 0.001)

    @SupportedDevices(['Ascend950'])
    def test_npu_quant_matmul_fp8_pertensor(self):
        x1 = torch.randint(-5, 5, (16, 38), dtype=torch.float8_e4m3fn)
        x2 = torch.randint(-5, 5, (16, 112), dtype=torch.float8_e5m2)
        x1_clone = x1.clone()
        x2_clone = x2.clone()
        scale = torch.randn(1, dtype=torch.float32)
        scale_clone = scale_clone.clone()
        supported_output = torch.matmul(x1.to(torch.float32), x2.to(torch.float32)) * scale
        custom_output = torch_npu.npu_quant_matmul(x1_clone.npu().t(), x2_clone.npu(), scale_clone.npu(), output_dtype=torch.float32)
        self.assertRtolEqual(supported_output.float().cpu().numpy(), custom_output.float().cpu().numpy(), 0.0001)

    @SupportedDevices(['Ascend950'])
    def test_npu_quant_matmul_hif8_perchannel(self):
        x1 = torch.randint(0, 255, (224, 64), dtype=torch.int8)
        x2 = torch.randint(0, 255, (64, 8360), dtype=torch.int8)
        x1_clone = x1.clone()
        x2_clone = x2.clone()
        x1 = torch.from_numpy(trans_np_hifuint8_tensor_to_float32(x1.numpy()))
        x2 = torch.from_numpy(trans_np_hifuint8_tensor_to_float32(x2.numpy()))
        scale = torch.randn(8360, dtype=torch.float32)
        scale_clone = scale_clone.clone()
        supported_output = torch.matmul(x1.to(torch.float32), x2.to(torch.float32)) * scale
        custom_output = torch_npu.npu_quant_matmul(x1_clone.npu(), x2_clone.npu(), scale_clone.npu(), x1_dtype=torch_npu.hifloat8, x2_dtype=torch_npu.hifloat8, output_dtype=torch.bfloat16)
        self.assertRtolEqual(supported_output.float().cpu().numpy(), custom_output.float().cpu().numpy(), 0.001)

    @SupportedDevices(['Ascend950'])
    def test_npu_quant_matmul_int8_pertoken(self):
        x1 = torch.randint(-5, 5, (4183, 1088), dtype=torch.int8)
        x2 = torch.randint(-5, 5, (960, 1088), dtype=torch.int8)
        x1_clone = x1.clone()
        x2_clone = x2.clone()
        scale = torch.randn(1, dtype=torch.float32)
        pertoken_scale = torch.randn(4183, dtype=torch.float32)
        scale_clone = scale_clone.clone()
        pertoken_scale_clone = pertoken_scale.clone()
        supported_output = torch.matmul(x1.to(torch.int32), x2.to(torch.int32)) * scale * pertoken_scale.unsqueeze(-1)
        custom_output = torch_npu.npu_quant_matmul(x1_clone.npu(), x2_clone.npu().t(), scale_clone.npu(), pertoken_scale=pertoken_scale_clone.npu(), output_dtype=torch.bfloat16)
        self.assertRtolEqual(supported_output.float().cpu().numpy(), custom_output.float().cpu().numpy(), 0.001)

    @SupportedDevices(['Ascend950'])
    def test_npu_quant_matmul_fp8_pertoken(self):
        x1 = torch.randint(-5, 5, (8192, 320), dtype=torch.float8_e4m3fn)
        x2 = torch.randint(-5, 5, (320, 2560), dtype=torch.float8_e4m3fn)
        x1_clone = x1.clone()
        x2_clone = x2.clone()
        scale = torch.randn(2560, dtype=torch.float32)
        pertoken_scale = torch.randn(4183, dtype=torch.float32)
        scale_clone = scale_clone.clone()
        pertoken_scale_clone = pertoken_scale.clone()
        supported_output = torch.matmul(x1.to(torch.float32), x2.to(torch.float32)) * scale * pertoken_scale.unsqueeze(-1)
        custom_output = torch_npu.npu_quant_matmul(x1_clone.npu(), x2_clone.npu(), scale_clone.npu(), pertoken_scale=pertoken_scale_clone.npu(), output_dtype=torch.float16)
        self.assertRtolEqual(supported_output.float().cpu().numpy(), custom_output.float().cpu().numpy(), 0.001)

    @SupportedDevices(['Ascend950'])
    def test_npu_quant_matmul_hif8_pertoken(self):
        x1 = torch.randint(0, 255, (8192, 320), dtype=torch.uint8)
        x2 = torch.randint(0, 255, (320, 2560), dtype=torch.uint8)
        x1_clone = x1.clone()
        x2_clone = x2.clone()
        x1 = torch.from_numpy(trans_np_hifuint8_tensor_to_float32(x1.numpy()))
        x2 = torch.from_numpy(trans_np_hifuint8_tensor_to_float32(x2.numpy()))
        scale = torch.randn(2560, dtype=torch.float32)
        pertoken_scale = torch.randn(4183, dtype=torch.float32)
        scale_clone = scale_clone.clone()
        pertoken_scale_clone = pertoken_scale.clone()
        supported_output = torch.matmul(x1.to(torch.float32), x2.to(torch.float32)) * scale * pertoken_scale.unsqueeze(-1)
        custom_output = torch_npu.npu_quant_matmul(x1_clone.npu(), x2_clone.npu(), scale_clone.npu(), pertoken_scale=pertoken_scale_clone.npu(), x1_dtype=torch_npu.hifloat8, x2_dtype=torch_npu.hifloat8, output_dtype=torch.float32)
        self.assertRtolEqual(supported_output.float().cpu().numpy(), custom_output.float().cpu().numpy(), 0.0001)

    @SupportedDevices(['Ascend950'])
    def test_npu_quant_matmul_fp8_doublescale(self):
        x1 = torch.randint(-5, 5, (544, 236), dtype=torch.float8_e5m2)
        x2 = torch.randint(-5, 5, (544, 568), dtype=torch.float8_e5m2)
        x1_clone = x1.clone()
        x2_clone = x2.clone()
        scale = torch.randn(1, dtype=torch.float32)
        pertoken_scale = torch.randn(1, dtype=torch.float32)
        scale_clone = scale_clone.clone()
        pertoken_scale_clone = pertoken_scale.clone()
        double_scale = scale * pertoken_scale
        supported_output = torch.matmul(x1.to(torch.float32), x2.to(torch.float32)) * double_scale
        custom_output = torch_npu.npu_quant_matmul(x1_clone.npu().t(), x2_clone.npu(), scale_clone.npu(), pertoken_scale=pertoken_scale_clone.npu(), output_dtype=torch.float32)
        self.assertRtolEqual(supported_output.float().cpu().numpy(), custom_output.float().cpu().numpy(), 0.0001)

    @SupportedDevices(['Ascend950'])
    def test_npu_quant_matmul_hif8_doublescale(self):
        x1 = torch.randint(0, 255, (30, 32), dtype=torch.uint8)
        x2 = torch.randint(0, 255, (32, 8), dtype=torch.uint8)
        x1_clone = x1.clone()
        x2_clone = x2.clone()
        x1 = torch.from_numpy(trans_np_hifuint8_tensor_to_float32(x1.numpy()))
        x2 = torch.from_numpy(trans_np_hifuint8_tensor_to_float32(x2.numpy()))
        scale = torch.randn(1, dtype=torch.float32)
        pertoken_scale = torch.randn(1, dtype=torch.float32)
        scale_clone = scale_clone.clone()
        pertoken_scale_clone = pertoken_scale.clone()
        double_scale = scale * pertoken_scale
        supported_output = torch.matmul(x1.to(torch.float32), x2.to(torch.float32)) * double_scale
        custom_output = torch_npu.npu_quant_matmul(x1_clone.npu(), x2_clone.npu(), scale_clone.npu(), pertoken_scale=pertoken_scale_clone.npu(), x1_dtype=torch_npu.hifloat8, x2_dtype=torch_npu.hifloat8, output_dtype=torch.bfloat16)
        self.assertRtolEqual(supported_output.float().cpu().numpy(), custom_output.float().cpu().numpy(), 0.001)

    @SupportedDevices(['Ascend950'])
    def test_npu_quant_matmul_mxfp8_odd_scaleK(self):
        x1 = torch.randint(-5, 5, (185, 480), dtype=torch.float8_e4m3fn)
        x2 = torch.randint(-5, 5, (1880, 480), dtype=torch.float8_e4m3fn)
        x1_clone = x1.clone()
        x2_clone = x2.clone()
        scale = torch.randint(-5, 5, (1880, 16), dtype=torch.int8)
        pertoken_scale = torch.randint(-5, 5, (185, 16), dtype=torch.int8)
        scale_clone = scale_clone.clone()
        pertoken_scale_clone = pertoken_scale.clone()
        x1 = x1.numpy()
        x2 = x2.numpy()
        scale = scale.numpy().transpose()
        pertoken_scale = pertoken_scale.numpy()
        scale = scale[:-1, :]
        pertoken_scale = pertoken_scale[:, :-1]
        scale_broadcast = np.repeat(scale, 32, axis=-1)
        pertoken_scale_broadcast = np.repeat(pertoken_scale, 32, axis=-2)
        x1_pad_len = pertoken_scale_broadcast.shape[-1] - x1.shape[-1]
        x2_pad_len = scale_broadcast[-2] - x2.shape[-2]
        x1_dims = len(x1.shape)
        x2_dims = len(x2.shape)
        x1 = np.pad(x1, [(0, 0)] * (x1_dims - 1) + [(0, x1_pad_len)], mode='constant', constant_values=0)
        x2 = np.pad(x2, [(0, 0)] * (x2_dims - 2) + [(0, x2_pad_len)] + [(0, 0)], mode='constant', constant_values=0)
        x1 = x1 * pertoken_scale_broadcast
        x2 = x2 * scale_broadcast
        supported_output = torch.matmul(x1.to(torch.float32), x2.to(torch.float32))
        custom_output = torch_npu.npu_quant_matmul(x1_clone.npu(), x2_clone.npu().t(), scale_clone.npu(), pertoken_scale=pertoken_scale_clone.npu(), scale_dtype=torch_npu.float8_e8m0fnu, pertoken_scale_dtype=torch_npu.float8_e8m0fnu, output_dtype=torch.float16)
        self.assertRtolEqual(supported_output.float().cpu().numpy(), custom_output.float().cpu().numpy(), 0.001)

    @SupportedDevices(['Ascend950'])
    def test_npu_quant_matmul_mxfp8_even_scaleK(self):
        x1 = torch.randint(-5, 5, (112, 944), dtype=torch.float8_e4m3fn)
        x2 = torch.randint(-5, 5, (184, 944), dtype=torch.float8_e4m3fn)
        x1_clone = x1.clone()
        x2_clone = x2.clone()
        scale = torch.randint(-5, 5, (184, 30), dtype=torch.int8)
        pertoken_scale = torch.randint(-5, 5, (112, 30), dtype=torch.int8)
        scale_clone = scale_clone.clone()
        pertoken_scale_clone = pertoken_scale.clone()
        x1 = x1.numpy()
        x2 = x2.numpy()
        scale = scale.numpy().transpose()
        pertoken_scale = pertoken_scale.numpy()
        scale = scale[:-1, :]
        pertoken_scale = pertoken_scale[:, :-1]
        scale_broadcast = np.repeat(scale, 32, axis=-1)
        pertoken_scale_broadcast = np.repeat(pertoken_scale, 32, axis=-2)
        x1_pad_len = pertoken_scale_broadcast.shape[-1] - x1.shape[-1]
        x2_pad_len = scale_broadcast[-2] - x2.shape[-2]
        x1_dims = len(x1.shape)
        x2_dims = len(x2.shape)
        x1 = np.pad(x1, [(0, 0)] * (x1_dims - 1) + [(0, x1_pad_len)], mode='constant', constant_values=0)
        x2 = np.pad(x2, [(0, 0)] * (x2_dims - 2) + [(0, x2_pad_len)] + [(0, 0)], mode='constant', constant_values=0)
        x1 = x1 * pertoken_scale_broadcast
        x2 = x2 * scale_broadcast
        supported_output = torch.matmul(x1.to(torch.float32), x2.to(torch.float32))
        custom_output = torch_npu.npu_quant_matmul(x1_clone.npu(), x2_clone.npu().t(), scale_clone.npu(), pertoken_scale=pertoken_scale_clone.npu(), scale_dtype=torch_npu.float8_e8m0fnu, pertoken_scale_dtype=torch_npu.float8_e8m0fnu, output_dtype=torch.bfloat16)
        self.assertRtolEqual(supported_output.float().cpu().numpy(), custom_output.float().cpu().numpy(), 0.001)

    @SupportedDevices(['Ascend950'])
    def test_npu_quant_matmul_mxfp4(self):
        x1 = torch.randint(-5, 5, (112, 944), dtype=torch.int8)
        x2 = torch.randint(-5, 5, (184, 944), dtype=torch.int8)
        x1_clone = x1.clone()
        x2_clone = x2.clone()
        scale = torch.randint(-5, 5, (184, 30), dtype=torch.int8)
        pertoken_scale = torch.randint(-5, 5, (112, 30), dtype=torch.int8)
        scale_clone = scale_clone.clone()
        pertoken_scale_clone = pertoken_scale.clone()
        x1 = x1.numpy()
        x2 = x2.numpy()
        scale = scale.numpy().transpose()
        pertoken_scale = pertoken_scale.numpy()
        scale = scale[:-1, :]
        pertoken_scale = pertoken_scale[:, :-1]
        scale_broadcast = np.repeat(scale, 32, axis=-1)
        pertoken_scale_broadcast = np.repeat(pertoken_scale, 32, axis=-2)
        x1_pad_len = pertoken_scale_broadcast.shape[-1] - x1.shape[-1]
        x2_pad_len = scale_broadcast[-2] - x2.shape[-2]
        x1_dims = len(x1.shape)
        x2_dims = len(x2.shape)
        x1 = np.pad(x1, [(0, 0)] * (x1_dims - 1) + [(0, x1_pad_len)], mode='constant', constant_values=0)
        x2 = np.pad(x2, [(0, 0)] * (x2_dims - 2) + [(0, x2_pad_len)] + [(0, 0)], mode='constant', constant_values=0)
        x1 = x1 * pertoken_scale_broadcast
        x2 = x2 * scale_broadcast
        supported_output = torch.matmul(x1.to(torch.float32), x2.to(torch.float32))
        custom_output = torch_npu.npu_quant_matmul(x1_clone.npu(), x2_clone.npu().t(), scale_clone.npu(), pertoken_scale=pertoken_scale_clone.npu(), x1_dtype=torch_npu.float4_e2m1fn_x2, x2_dtype=torch_npu.float4_e2m1fn_x2, scale_dtype=torch_npu.float8_e8m0fnu, pertoken_scale_dtype=torch_npu.float8_e8m0fnu, output_dtype=torch.bfloat16)
        self.assertRtolEqual(supported_output.float().cpu().numpy(), custom_output.float().cpu().numpy(), 0.001)

    @SupportedDevices(['Ascend950'])
    def test_npu_quant_matmul_hif8_perblock(self):
        x1 = torch.randint(0, 255, (8192, 320), dtype=torch.uint8)
        x2 = torch.randint(0, 255, (320, 2560), dtype=torch.uint8)
        x1_clone = x1.clone()
        x2_clone = x2.clone()
        x1 = torch.from_numpy(trans_np_hifuint8_tensor_to_float32(x1.numpy()))
        x2 = torch.from_numpy(trans_np_hifuint8_tensor_to_float32(x2.numpy()))
        scale = torch.randint(-5, 5, (3, 20), dtype=torch.float32)
        pertoken_scale = torch.randint(-5, 5, (64, 3), dtype=torch.float32)
        scale_clone = scale_clone.clone()
        pertoken_scale_clone = pertoken_scale.clone()
        double_scale = scale * pertoken_scale
        supported_output = torch.matmul(x1.to(torch.float32), x2.to(torch.float32)) * double_scale
        custom_output = torch_npu.npu_quant_matmul(x1_clone.npu(), x2_clone.npu(), scale_clone.npu(), pertoken_scale=pertoken_scale_clone.npu(), x1_dtype=torch_npu.hifloat8, x2_dtype=torch_npu.hifloat8, output_dtype=torch.bfloat16)
        self.assertRtolEqual(supported_output.float().cpu().numpy(), custom_output.float().cpu().numpy(), 0.001)

    @SupportedDevices(['Ascend950'])
    def test_npu_quant_matmul_fp8_pertile(self):
        x1 = torch.randint(-5, 5, (8192, 320), dtype=torch.float8_e5m2)
        x2 = torch.randint(-5, 5, (320, 2560), dtype=torch.float8_e5m2)
        x1_clone = x1.clone()
        x2_clone = x2.clone()
        scale = torch.randint(-5, 5, (3, 20), dtype=torch.float32)
        pertoken_scale = torch.randint(-5, 5, (8192, 3), dtype=torch.float32)
        scale_clone = scale_clone.clone()
        pertoken_scale_clone = pertoken_scale.clone()
        double_scale = scale * pertoken_scale
        supported_output = torch.matmul(x1.to(torch.float32), x2.to(torch.float32)) * double_scale
        custom_output = torch_npu.npu_quant_matmul(x1_clone.npu().t(), x2_clone.npu(), scale_clone.npu(), pertoken_scale=pertoken_scale_clone.npu(), output_dtype=torch.float32)
        self.assertRtolEqual(supported_output.float().cpu().numpy(), custom_output.float().cpu().numpy(), 0.0001)

    @SupportedDevices(['Ascend950'])
    def test_npu_quant_matmul_fp8_pertile_invaild_trans(self):
        x1 = torch.randint(-5, 5, (10, 256), dtype=torch.float8_e5m2)
        x2 = torch.randint(-5, 5, (512, 256), dtype=torch.float8_e5m2)
        scale = torch.randint(-5, 5, (4, 2), dtype=torch.float32)
        pertoken_scale = torch.randint(-5, 5, (10, 2), dtype=torch.float32)
        try:
            custom_output = torch_npu.npu_quant_matmul(x1.npu(), x2.npu().t(), scale.npu(), pertoken_scale=pertoken_scale.npu(), output_dtype=torch.float32)
        except RuntimeError as e:
            self.assertTrue("transpose are not same, please check input" in str(e))

    # pylint:disable = huawei-too-many-arguments
    def a8w4_quant_golden(self, x, weight, scale, perTokenScale, groupList, bias, output_dtype):
        m, k = x.shape
        k, n = weight.shape
        x = x.numpy()
        weight = weight.cpu().numpy()
        scale = scale.numpy()
        perTokenScale = perTokenScale.numpy()
        bias = bias.numpy()
        bias = bias.reshape(1, -1)
        weight_int8 = weight.astype(np.int8)
        x = np.concatenate([x.reshape(m, 1, k) // 16, (x.reshape(m, 1, k) & 0x0F) - 8], axis=1).reshape(m * 2, k)

        groupNum = 1
        quantGroupNum = scale.shape[0]
        index = np.cumsum(groupList)
        xSplit = np.split(x, index * 2, axis=0)
        xSplit[0] = x
        perTokenScaleSplit = np.split(perTokenScale, index, axis=0)
        weightGroup = weight_int8.reshape(groupNum, quantGroupNum, k // quantGroupNum, n).astype(np.int32)
        mmOuts = []
        atomic = np.float16
        mmi = []
        for i in range(groupNum):
            xi = xSplit[i].reshape(-1, quantGroupNum, k // quantGroupNum).astype(np.int32)
            mmi = np.zeros([xi.shape[0], n], dtype=atomic)
            for j in range(quantGroupNum):
                mm = np.matmul(xi[:, j, :], weightGroup[i, j, ...])
                mm = mm.astype(np.float32) * scale[j, :].reshape(1, -1)
                mmi = (mmi.astype(atomic) + mm.astype(atomic)).astype(atomic)

            mmi = mmi.reshape(-1, 2, n).astype(np.float32)
            mmi = mmi[:, 0, :] * 16 + mmi[:, 1, :] + bias[i].reshape(1, n)
            mmi = mmi * perTokenScale
            mmOuts.append(mmi)
        golden = np.concatenate(mmOuts, axis=0)
        golden_tensor = torch.from_numpy(mmi)

        return golden_tensor.to(output_dtype)

    def get_eb(self, golden: torch.Tensor, actual: torch.Tensor):
        golden = golden.to(torch.float32)
        golden_nmax = torch.clamp(torch.abs(golden), min=1)
        actual_error = actual.to(torch.float32) - golden
        EB = torch.mean(actual_error / golden_nmax)
        if golden.dtype == torch.float16:
            result = EB <= 2 ** (-10)
        else:
            result = EB <= 2 ** (-7)
        return result

    def ref_compare(self, golden: torch.Tensor, actual: torch.Tensor, thresh: float):
        golden = golden.to(torch.float32)
        golden_nmax = torch.clamp(torch.abs(golden), min=1)
        abs_error = torch.abs(actual.to(torch.float32) - golden)
        result = (abs_error <= thresh * golden_nmax).all()
        return result

    def golden_compare(self, out_tensor, golden_out_tensor, ksize):
        eb = self.get_eb(golden_out_tensor, out_tensor)
        if out_tensor.dtype == torch.float16:
            cmp = self.ref_compare(golden_out_tensor, out_tensor, 2 ** -8 if ksize < 2048 else 2**-7)
        else:
            cmp = self.ref_compare(golden_out_tensor, out_tensor, 2 ** -7 if ksize < 2048 else 2**-6)
        return eb and cmp

    def trans_np_hifuint8_tensor_to_float32(self, in_tensor):
        for i in range(in_tensor.shape[0]):
            for j in range(in_tensor.shape[1]):
                if in_tensor[i][j] == 0 or in_tensor[i][j] == 128:
                    in_tensor[i][j] = float(0)
                elif in_tensor[i][j] == 239:
                    in_tensor[i][j] = -32768
                elif in_tensor[i][j] == 111:
                    in_tensor[i][j] = 32768
                else:
                    if in_tensor[i][j] >= 128:
                        sign = -1.0
                    else:
                        sign = 1.0
                    dot_4_bits = in_tensor[i][j] & 120
                    dot_4_value = dot_4_bits >> 3
                    if dot_4_value >= 12:
                        exponent = in_tensor[i][j] & 30
                        exponent_int = exponent >> 1
                        if exponent_int >= 8:
                            exponent_value = -exponent_int
                        else:
                            exponent_value = exponent_int + 8
                        fra_int = in_tensor[i][j] & 1
                        m_value = 1.0 + fra_int * 0.5
                    elif dot_4_value >= 8:
                        exponent = in_tensor[i][j] & 28
                        exponent_int = exponent >> 2
                        if exponent_int >= 4:
                            exponent_value = -exponent_int
                        else:
                            exponent_value = exponent_int + 4
                        fra_int = in_tensor[i][j] & 3
                        m_value = 1.0 + fra_int * 0.25
                    elif dot_4_value >= 4:
                        exponent = in_tensor[i][j] & 24
                        exponent_int = exponent >> 3
                        if exponent_int >= 2:
                            exponent_value = -exponent_int
                        else:
                            exponent_value = exponent_int + 2
                        fra_int = in_tensor[i][j] & 7
                        m_value = 1.0 + fra_int * 0.125
                    elif dot_4_value >= 2:
                        exponent = in_tensor[i][j] & 8
                        exponent_sign = exponent >> 3
                        if exponent_sign >= 1:
                            exponent_value = -1
                        else:
                            exponent_value = 1
                        fra_int = in_tensor[i][j] & 7
                        m_value = 1.0 + fra_int * 0.125
                    elif dot_4_value == 1:
                        exponent_value = 0
                        fra_int = in_tensor[i][j] & 7
                        m_value = 1.0 + fra_int * 0.125
                    elif dot_4_value == 0:
                        m_value = 1
                        exponent_value = (in_tensor[i][j] & 7) - 23
                    else:
                        print("error, dot error")
                        m_value = 0.0
                        exponent_value = 0.0
                    in_tensor[i][j] = sign + pow(2.0, exponent_value) * m_value
        return in_tensor.astype(np.float32)

    @unittest.skip("skip test_npu_quant_matmul_A8W4")
    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_A8W4_float16(self):
        M = 1
        K = 8192
        N = 128 * 8
        group_size = 256

        x1 = torch.randint(-128, 128, [M, K], dtype=torch.int8)
        x2 = torch.randint(1, 8, (K, N), dtype=torch.int32)
        y_offset = torch.randn([N, ], dtype=torch.float32)
        x1_scale = torch.randn([M, 1], dtype=torch.float32) * 0.01
        x2_scale = torch.randn([K // group_size, N], dtype=torch.float32).uniform_(0, 1)
        group_size_list = [0, 0, group_size]

        x1_npu = x1.npu()
        x2_npu = torch_npu.npu_convert_weight_to_int4pack(x2.npu())
        y_offset_npu = y_offset.npu()
        x1_scale_npu = x1_scale.npu()
        x2_scale_tmp = torch_npu.npu_trans_quant_param(x2_scale.npu().reshape([K // group_size * N, ]))
        x2_scale_npu = x2_scale_tmp.reshape([K // group_size, N])

        npu_out = torch_npu.npu_quant_matmul(x1_npu, x2_npu, scale=x2_scale_npu,
                                             offset=y_offset_npu, pertoken_scale=x1_scale_npu,
                                             bias=None, output_dtype=torch.float16, group_sizes=group_size_list)
        cpu_out = self.a8w4_quant_golden(x=x1, weight=x2, scale=x2_scale, perTokenScale=x1_scale,
                                         groupList=group_size_list, bias=y_offset, output_dtype=torch.float16)

        self.assertTrue(self.golden_compare(npu_out.cpu(), cpu_out, K))

    @unittest.skip("skip test_npu_quant_matmul_A8W4")
    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_A8W4_bfloat16(self):
        M = 1
        K = 8192
        N = 128 * 8
        group_size = 256

        x1 = torch.randint(-128, 128, [M, K], dtype=torch.int8)
        x2 = torch.randint(1, 8, (K, N), dtype=torch.int32)
        y_offset = torch.randn([N, ], dtype=torch.float32)
        x1_scale = torch.randn([M, 1], dtype=torch.float32) * 0.01
        x2_scale = torch.randn([K // group_size, N], dtype=torch.float32).uniform_(0, 1)
        group_size_list = [0, 0, group_size]

        x1_npu = x1.npu()
        x2_npu = torch_npu.npu_convert_weight_to_int4pack(x2.npu())
        y_offset_npu = y_offset.npu()
        x1_scale_npu = x1_scale.npu()
        x2_scale_tmp = torch_npu.npu_trans_quant_param(x2_scale.npu().reshape([K // group_size * N, ]))
        x2_scale_npu = x2_scale_tmp.reshape([K // group_size, N])

        npu_out = torch_npu.npu_quant_matmul(x1_npu, x2_npu, scale=x2_scale_npu,
                                             offset=y_offset_npu, pertoken_scale=x1_scale_npu,
                                             bias=None, output_dtype=torch.bfloat16, group_sizes=group_size_list)
        cpu_out = self.a8w4_quant_golden(x=x1, weight=x2, scale=x2_scale, perTokenScale=x1_scale,
                                         groupList=group_size_list, bias=y_offset, output_dtype=torch.bfloat16)

        self.assertTrue(self.golden_compare(npu_out.cpu(), cpu_out, K))


if __name__ == "__main__":
    run_tests()

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


class TestQuantMatmul(TestCase):

    @SupportedDevices(['Ascend910B'])
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

    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_bf16(self):
        x1 = torch.randint(-5, 5, (8192, 320), dtype=torch.int8)
        x2 = torch.randint(-5, 5, (320, 2560), dtype=torch.int8)
        x1_clone = x1.clone()
        x2_clone = x2.clone()
        scale = torch.randn(1, dtype=torch.float32)
        supported_output = torch.matmul(x1.to(torch.int32), x2.to(torch.int32)) * scale
        custom_output = torch_npu.npu_quant_matmul(x1_clone.npu(), x2_clone.npu(), scale.npu(), output_dtype=torch.bfloat16)
        self.assertRtolEqual(supported_output.float().cpu().numpy(), custom_output.float().cpu().numpy(), 0.01)
    
    @SupportedDevices(['Ascend910B'])
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
    
    @SupportedDevices(['Ascend910B'])
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

if __name__ == "__main__":
    run_tests()

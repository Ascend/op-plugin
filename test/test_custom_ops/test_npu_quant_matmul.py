
import math
import unittest
import copy
import struct
from struct import pack, unpack
from dataclasses import dataclass
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


def float32_to_int9(fp32_offset):
    int_value = (np.round(fp32_offset)).astype(int)
    int9_value = np.clip(int_value, -256, 255)
    return int9_value


def deq_scale_generate_v1(scale, output_dtype=None):
    if output_dtype == 2 or output_dtype is None:
        fp32_deq_scale = scale.numpy().astype(np.float32)
    elif output_dtype == 1:
        deq_scale_shape = scale.numpy().shape
        fp32_deq_scale = scale.numpy().astype(np.float32)
        uint32_deq_scale = np.frombuffer(fp32_deq_scale, np.uint32).reshape(deq_scale_shape)
        #与高19位运算，模拟硬件
        uint32_deq_scale &= 0XFFFFE000
        fp32_deq_scale = np.frombuffer(uint32_deq_scale, np.float32)
    elif output_dtype == 27 or output_dtype == 3:
        fp32_deq_scale = scale.float().numpy().astype(np.float32)
    else:
        raise ValueError("invalid output_dtype")
    return fp32_deq_scale


@dataclass
class TensorInfo:
    x1: torch.Tensor
    x2: torch.Tensor
    scale: torch.Tensor
    offset_flag: bool = False
    offset: torch.Tensor = None
    bias_flag: bool = False
    bias: torch.Tensor = None
    output_dtype: str = None
    bias_dtype: str = None


def cpu_golden_func(tensor_info: TensorInfo):
    out = torch.matmul(tensor_info.x1, tensor_info.x2)
    if tensor_info.bias_flag and tensor_info.bias_dtype == "int32":
        out = torch.add(out, bias)
    fp32_deq_scale = deq_scale_generate_v1(tensor_info.scale, tensor_info.output_dtype)
    deq_scale_slice = fp32_deq_scale.reshape(1, -1)[:, :out.shape[-1]]
    if tensor_info.output_dtype == 3:
        out = out.numpy()
    else:
        out = (out * deq_scale_slice).numpy()
    if tensor_info.output_dtype == 2 or tensor_info.output_dtype is None:
        if tensor_info.offset_flag:
            out = float32_to_int9(out) + float32_to_int9(tensor_info.offset.numpy().astype(np.float32))
        out = np.clip(out, -128, 127)
        out = torch.tensor(out, dtype=torch.int8)
    elif tensor_info.output_dtype == 1:
        out = torch.tensor(out, dtype=torch.float16)
    elif tensor_info.output_dtype == 27:
        if tensor_info.bias_flag and tensor_info.bias_dtype != "int32":
            out = torch.tensor(out, dtype=torch.float)
            bias = torch.tensor(bias, dtype=torch.float)
            out = torch.add(out, bias)
        out = torch.tensor(out, dtype=torch.bfloat16)
    elif tensor_info.output_dtype == 3:
        out = torch.tensor(out, dtype=torch.int32)
    else:
        raise ValueError("output invalid")
    return out


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
        tensor_info = TensorInfo(x1.to(torch.int32), x2.to(torch.int32), scale)
        tensor_info.output_dtype = 1
        supported_output = cpu_golden_func(tensor_info)
        custom_output = torch_npu.npu_quant_matmul(x1_clone.npu(), x2_clone.npu(), scale_quant.npu(), output_dtype=torch.float16)
        self.assertRtolEqual(supported_output.float().cpu().numpy(), custom_output.float().cpu().numpy(), 0.001)

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

        tensor_info = TensorInfo(x1.to(torch.int32), x2.to(torch.int32), scale)
        tensor_info.output_dtype = 1
        supported_output = cpu_golden_func(tensor_info)
        custom_output = torch_npu.npu_quant_matmul(x1_clone, x2_clone, scale_quant, output_dtype=torch.float16)
        self.assertRtolEqual(supported_output.float().cpu().numpy(), custom_output.float().cpu().numpy(), 0.001)

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

        tensor_info = TensorInfo(x1.to(torch.int32), x2.t().to(torch.int32), scale)
        tensor_info.output_dtype = 1
        supported_output = cpu_golden_func(tensor_info)
        custom_output = torch_npu.npu_quant_matmul(x1_clone, x2_clone.t(), scale_quant, output_dtype=torch.float16)
        self.assertRtolEqual(supported_output.float().cpu().numpy(), custom_output.float().cpu().numpy(), 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_bf16(self):
        x1 = torch.randint(-5, 5, (8192, 320), dtype=torch.int8)
        x2 = torch.randint(-5, 5, (320, 2560), dtype=torch.int8)
        x1_clone = x1.clone()
        x2_clone = x2.clone()
        scale = torch.randn(1, dtype=torch.float32)
        tensor_info = TensorInfo(x1.to(torch.int32), x2.to(torch.int32), scale)
        tensor_info.output_dtype = 27
        supported_output = cpu_golden_func(tensor_info)
        custom_output = torch_npu.npu_quant_matmul(x1_clone.npu(), x2_clone.npu(), scale.npu(), output_dtype=torch.bfloat16)
        self.assertRtolEqual(supported_output.float().cpu().numpy(), custom_output.float().cpu().numpy(), 0.001)
    
    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_bf16_nz(self):
        x1 = torch.randint(-5, 5, (8192, 320), dtype=torch.int8)
        x2 = torch.randint(-5, 5, (320, 2560), dtype=torch.int8)
        x1_clone = x1.clone()
        x2_clone = x2.clone()
        scale = torch.randn(1, dtype=torch.float32)
        tensor_info = TensorInfo(x1.to(torch.int32), x2.to(torch.int32), scale)
        tensor_info.output_dtype = 27
        supported_output = cpu_golden_func(tensor_info)
        x2_nz = torch_npu.npu_format_cast(x2_clone.npu().contiguous(), 29)
        custom_output = torch_npu.npu_quant_matmul(x1_clone.npu(), x2_nz.npu(), scale.npu(), output_dtype=torch.bfloat16)
        self.assertRtolEqual(supported_output.float().cpu().numpy(), custom_output.float().cpu().numpy(), 0.001)
    
    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_fp16_nz(self):
        x1 = torch.randint(-5, 5, (8192, 320), dtype=torch.int8)
        x2 = torch.randint(-5, 5, (320, 2560), dtype=torch.int8)
        x1_clone = x1.clone()
        x2_clone = x2.clone()
        scale = torch.randn(1, dtype=torch.float32)
        tensor_info = TensorInfo(x1.to(torch.int32), x2.to(torch.int32), scale)
        tensor_info.output_dtype = 1
        supported_output = cpu_golden_func(tensor_info)
        x2_nz = torch_npu.npu_format_cast(x2_clone.npu().contiguous(), 29)
        custom_output = torch_npu.npu_quant_matmul(x1_clone.npu(), x2_nz.npu(), scale.npu(), output_dtype=torch.float16)
        self.assertRtolEqual(supported_output.float().cpu().numpy(), custom_output.float().cpu().numpy(), 0.001)
        
        
if __name__ == "__main__":
    run_tests()

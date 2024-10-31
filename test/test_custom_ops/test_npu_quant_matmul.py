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
    def deq_scale_generate(self, deq_scale_shape, out_dtype=None):
        fp32_deq_scale = np.random.uniform(low=2, high=3, size=deq_scale_shape).astype(np.float32)
        uint32_deq_scale = np.frombuffer(fp32_deq_scale, np.uint32).reshape(deq_scale_shape)
        #与高19位运算，模拟硬件
        uint32_deq_scale &= 0XFFFFE000

        if out_dtype != "int8":
            fp32_deq_scale = np.frombuffer(uint32_deq_scale, np.float32)
            uint64_deq_scale = np.zeros(deq_scale_shape, np.int64)
            uint64_deq_scale |= np.int64(uint32_deq_scale)
        elif out_dtype == "int8":
            temp_quant_tensor = np.random.randint(1, 3, deq_scale_shape).astype(np.float32)
            temp_quant_tensor_api = copy.deepcopy(temp_quant_tensor).astype(np.int64)
            for i, _ in enumerate(temp_quant_tensor_api):
                temp_quant_tensor_api[i] = struct.unpack('!I', struct.pack('!f', temp_quant_tensor_api[i]))[0]
                temp_quant_tensor_api[i] = temp_quant_tensor_api[i] | np.int64(0x400000000000)
            uint64_deq_scale = np.frombuffer(temp_quant_tensor_api, np.int64)
        return uint64_deq_scale

    def pertoken_scale_generate(self, pertoken_scale_shape):
        fp32_pertoken_scale = np.random.uniform(low=2, high=3, size=pertoken_scale_shape).astype(np.float32)
        return fp32_pertoken_scale

    def supported_op_exec(self, x1, x2, uint64_deq_scale, fp32_pertoken_scale, bias):
        x1 = torch.from_numpy(x1).to(torch.int32)
        x2 = torch.from_numpy(x2).to(torch.int32)
        mm_res = torch.matmul(x1, x2)
        uint64_deq_scale_slice = uint64_deq_scale.reshape(1, -1)[:, :mm_res.shape[-1]]
        uint64_deq_scale_slice = torch.from_numpy(uint64_deq_scale_slice)
        if fp32_pertoken_scale is None:
            output = (mm_res * uint64_deq_scale_slice).numpy().astype(np.float16)
        else:
            output = (mm_res * uint64_deq_scale_slice * fp32_pertoken_scale).numpy().astype(np.float16)
        return torch.from_numpy(output)

    def custom_op_exec(self, x1, x2, uint64_deq_scale, fp32_pertoken_scale, bias):
        return torch_npu.npu_quant_matmul(x1, x2, uint64_deq_scale, pertoken_scale=fp32_pertoken_scale, bias=bias, output_dtype=torch.float16)

    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_a8w8(self, device="npu"):
        torch.mannal_seed(0)
        x1 = torch.randint(-5, 5, (8192, 320), dtype=torch.int8)
        x2 = torch.randint(-5, 5, (320, 2560), dtype=torch.int8)
        deq_scale_shape = (1,)
        x1_clone = x1.clone()
        x2_clone = x2.clone()
        bias = torch.randint(-1, 1, (2560,), dtype=torch.int32)
        uint64_deq_scale = self.deq_scale_generate(deq_scale_shape, 'float16')

        supported_output = self.supported_op_exec(x1.numpy(), x2.numpy(), uint64_deq_scale, None, bias.numpy())
        custom_output = self.custom_op_exec(x1_clone.npu(), x2_clone.npu(),
                                            torch.from_numpy(uint64_deq_scale).npu(), None, bias.npu())
        self.assertRtolEqual(x1, x1_clone, 0.001)
        self.assertRtolEqual(supported_output, custom_output, 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_a4w4(self, device="npu"):
        torch.mannal_seed(0)
        x1_a4w4 = torch.randint(-5, 5, (8192, 40), dtype=torch.int32)
        x2_a4w4 = torch.randint(-5, 5, (320, 320), dtype=torch.int32)
        deq_scale_shape = (1,)
        x1_a4w4_clone = x1_a4w4.clone()
        x2_a4w4_clone = x2_a4w4.clone()
        bias = torch.randint(-1, 1, (320,), dtype=torch.int32).npu()
        uint64_deq_scale = self.deq_scale_generate(deq_scale_shape, 'float16')

        supported_output = self.supported_op_exec(x1_a4w4.numpy(), x2_a4w4.numpy(), uint64_deq_scale, None, bias.numpy())
        custom_output = self.custom_op_exec(x1_a4w4_clone.npu(), x2_a4w4_clone.npu(), uint64_deq_scale.npu(), None, bias.npu())
        self.assertRtolEqual(x1_a4w4, x1_a4w4_clone, 0.001)
        self.assertRtolEqual(supported_output, custom_output, 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_continuous_x2_tensor(self, device="npu"):
        torch.mannal_seed(0)
        x1 = torch.randint(-5, 5, (5, 20), dtype=torch.int32)
        x2 = torch.randint(-5, 5, (80, 20), dtype=torch.int32)
        deq_scale_shape = (80,)
        x1_clone = x1.clone()
        x2_clone = x2.clone()
        bias = torch.randint(-1, 1, (160,), dtype=torch.int32)
        uint64_deq_scale = self.deq_scale_generate(deq_scale_shape, 'float16')

        supported_output = self.supported_op_exec(x1.numpy(), x2.t().numpy(), uint64_deq_scale, None, bias.numpy())
        custom_output = self.custom_op_exec(x1_clone.npu(), x2_clone.t().npu(),
                                            torch.from_numpy(uint64_deq_scale).npu(), None, bias.npu())
        self.assertRtolEqual(x1, x1_clone, 0.001)
        self.assertRtolEqual(supported_output, custom_output, 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_bf16(self, device="npu"):
        x1 = torch.randint(-5, 5, (8192, 320), dtype=torch.int8)
        x2 = torch.randint(-5, 5, (320, 2560), dtype=torch.int8)
        x1_clone = x1.clone()
        x2_clone = x2.clone()
        scale = torch.randn(1, dtype=torch.float32)
        supported_output = torch.matmul(x1.to(torch.int32), x2.to(torch.int32)) * scale
        custom_output = torch_npu.npu_quant_matmul(x1_clone.npu(), x2_clone.npu(), scale.npu(), output_dtype=torch.bfloat16)
        self.assertRtolEqual(supported_output.numpy().astype(np.float32), custom_output.numpy().astype(np.float32), 0.5)
    
    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_bf16_nz(self, device="npu"):
        x1 = torch.randint(-5, 5, (8192, 320), dtype=torch.int8)
        x2 = torch.randint(-5, 5, (320, 2560), dtype=torch.int8)
        x1_clone = x1.clone()
        x2_clone = x2.clone()
        scale = torch.randn(1, dtype=torch.float32)
        supported_output = torch.matmul(x1.to(torch.int32), x2.to(torch.int32)) * scale
        x2_nz = torch_npu.npu_format_cast(x2_clone.npu().contiguous(), 29)
        custom_output = torch_npu.npu_quant_matmul(x1_clone.npu(), x2_nz.npu(), scale.npu(), output_dtype=torch.bfloat16)
        self.assertRtolEqual(supported_output.numpy().astype(np.float32), custom_output.numpy().astype(np.float32), 0.5)
    
    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_fp16_nz(self, device="npu"):
        x1 = torch.randint(-5, 5, (8192, 320), dtype=torch.int8)
        x2 = torch.randint(-5, 5, (320, 2560), dtype=torch.int8)
        x1_clone = x1.clone()
        x2_clone = x2.clone()
        scale = torch.randn(1, dtype=torch.float32)
        supported_output = torch.matmul(x1.to(torch.int32), x2.to(torch.int32)) * scale
        x2_nz = torch_npu.npu_format_cast(x2_clone.npu().contiguous(), 29)
        custom_output = torch_npu.npu_quant_matmul(x1_clone.npu(), x2_nz.npu(), scale.npu())
        self.assertRtolEqual(supported_output.numpy().astype(np.float32), custom_output.numpy().astype(np.float32), 0.5)
        
if __name__ == "__main__":
    run_tests()

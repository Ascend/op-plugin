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
            uint64_deq_scale = np.zeros(deq_scale_shape, np.uint64)
            uint64_deq_scale |= np.uint64(uint32_deq_scale)
        elif out_dtype == "int8":
            temp_quant_tensor = np.random.randint(1, 3, deq_scale_shape).astype(np.float32)
            temp_quant_tensor_api = copy.deepcopy(temp_quant_tensor).astype(np.uint64)
            for i, _ in enumerate(temp_quant_tensor_api):
                temp_quant_tensor_api[i] = struct.unpack('!I', struct.pack('!f', temp_quant_tensor_api[i]))[0]
                temp_quant_tensor_api[i] = temp_quant_tensor_api[i] | np.uint64(0x400000000000)
            uint64_deq_scale = np.frombuffer(temp_quant_tensor_api, np.uint64)
        return uint64_deq_scale

    def supported_op_exec(self, x1, x2, uint64_deq_scale, bias):
        x1 = torch.from_numpy(x1).to(torch.int32)
        x2 = torch.from_numpy(x2).to(torch.int32)
        mm_res = torch.matmul(x1, x2)
        uint64_deq_scale_slice = uint64_deq_scale.reshape(1, -1)[:, :mm_res.shape[-1]]
        uint64_deq_scale_slice = torch.from_numpy(uint64_deq_scale_slice)
        output = (mm_res * uint64_deq_scale_slice).numpy().astype(np.float16)
        output = torch.add(out, bias)
        return output

    def custom_op_exec(self, x1, x2, uint64_deq_scale, bias):
        return torch_npu.npu_quant_matmul(x1, x2, uint64_deq_scale, bias=bias)

    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul(self, device="npu"):
        torch.mannal_seed(0)
        x1 = torch.randn(1, 8192, 320, dtype=torch.int8).npu()
        x2 = torch.randn(1, 320, 2560, dtype=torch.int8).npu()
        deq_scale_shape = (1,)
        x1_clone = x1.clone()
        x2_clone = x2.clone()
        bias = torch.randint(-1, 1, (1, 1, 2560), dtype=torch.int32).npu()
        uint64_deq_scale = self.deq_scale_generate(deq_scale_shape, 'float16')

        supported_output = self.supported_op_exec(x1, x2, uint64_deq_scale, bias)
        custom_output = self.custom_op_exec(x1_clone, x2_clone, uint64_deq_scale, bias)
        self.assertRtolEqual(x1, x1_clone, 0.001)
        self.assertRtolEqual(supported_output, custom_output, 0.001)


if __name__ == "__main__":
    run_tests()

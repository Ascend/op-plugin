import math
import unittest
import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestFFN(TestCase):

    def deq_scale_generate(self, deq_scale_shape):
        fp32_deq_scale = np.random.uniform(low=0.01, high=0.05, size=deq_scale_shape).astype(np.float32)
        uint32_deq_scale = np.frombuffer(fp32_deq_scale, np.uint32).reshape(deq_scale_shape)
        # Perform AND operation with the upper 19 bits to simulate hardware
        uint32_deq_scale &= 0XFFFFE000
        fp32_deq_scale = np.frombuffer(uint32_deq_scale, np.float32)
        uint64_deq_scale = np.zeros(deq_scale_shape, np.uint64)
        uint64_deq_scale |= np.uint64(uint32_deq_scale)
        return fp32_deq_scale, uint64_deq_scale

    def supported_op_exec(self, x, weight1, weight2, activation, *, antiquant_scale1=None, antiquant_scale2=None,
                          antiquant_offset1=None, antiquant_offset2=None, scale=None, offset=None, deq_scale1=None,
                          deq_scale2=None):
        if antiquant_scale1 is not None:
            # mm1
            x = x.to(torch.float32).npu()
            weight1 = torch.from_numpy(weight1).to(torch.float16).npu()
            antiquant_offset1 = torch.broadcast_to(antiquant_offset1, weight1.size())
            antiquant_scale1 = torch.broadcast_to(antiquant_scale1, weight1.size())
            weight1 = (weight1 + antiquant_offset1) * antiquant_scale1
            weight1 = weight1.to(torch.float32)
            mm1_res = torch.matmul(x, weight1).to(torch.float16)
            # activation
            if activation == "relu":
                activation_res = torch.nn.functional.relu(mm1_res)
            elif activation == "gelu":
                activation_res = torch.nn.functional.gelu(mm1_res)
            elif activation == "silu":
                activation_res = torch.nn.functional.silu(mm1_res)
            activation_res = activation_res.to(torch.float32)
            # mm2
            weight2 = torch.from_numpy(weight2).to(torch.float16).npu()
            antiquant_offset2 = torch.broadcast_to(antiquant_offset2, weight2.size())
            antiquant_scale2 = torch.broadcast_to(antiquant_scale2, weight2.size())
            weight2 = (weight2 + antiquant_offset2) * antiquant_scale2
            weight2 = weight2.to(torch.float32)
            mm2_res = torch.matmul(activation_res, weight2).to(torch.float16)
        elif scale is not None:
            x = x.to(torch.int32)
            weight1 = torch.from_numpy(weight1).to(torch.int32)
            weight2 = torch.from_numpy(weight2).to(torch.int32)
            # mm1
            mm1_res = torch.matmul(x, weight1)
            deq_scale1 = deq_scale1.reshape(1, -1)[:, :mm1_res.shape[-1]]
            deq_scale1 = torch.from_numpy(deq_scale1)
            mm1_res = (mm1_res * deq_scale1).to(torch.float16)
            # activation
            scale = torch.from_numpy(scale).to(torch.float16)
            offset = torch.from_numpy(offset).to(torch.float16)
            if activation == "relu":
                activation_res = torch.nn.functional.relu(mm1_res)
            elif activation == "gelu":
                activation_res = torch.nn.functional.gelu(mm1_res)
            elif activation == "silu":
                activation_res = torch.nn.functional.silu(mm1_res)
            activation_res = activation_res * scale + offset
            activation_res = torch.round(activation_res, decimals=0)
            activation_res = activation_res.clamp(-128, 127).to(torch.int32)
            # mm2
            mm2_res = torch.matmul(activation_res, weight2)
            deq_scale2 = deq_scale2.reshape(1, -1)[:, :mm2_res.shape[-1]]
            deq_scale2 = torch.from_numpy(deq_scale2)    
            mm2_res = (mm2_res * deq_scale2).to(torch.float16) 
        else:
            mm1_res = torch.matmul(x, weight1)
            if activation == "relu":
                activation_res = torch.nn.functional.relu(mm1_res)
            elif activation == "gelu":
                activation_res = torch.nn.functional.gelu(mm1_res)
            elif activation == "silu":
                activation_res = torch.nn.functional.silu(mm1_res)
            mm2_res = torch.matmul(activation_res, weight2)
        return mm2_res

    def custom_op_exec(self, x, weight1, weight2, activation, *, antiquant_scale1=None, antiquant_scale2=None,
                       antiquant_offset1=None, antiquant_offset2=None, scale=None, offset=None, deq_scale1=None,
                       deq_scale2=None):
        if antiquant_scale1 is not None:
            return torch_npu.npu_ffn(x, weight1, weight2, activation, antiquant_scale1=antiquant_scale1,
                                     antiquant_scale2=antiquant_scale2, antiquant_offset1=antiquant_offset1,
                                     antiquant_offset2=antiquant_offset2, inner_precise=1)
        elif scale is not None:
            x = x.npu()
            weight1 = weight1.npu()
            weight2 = weight2.npu()
            scale = scale.npu()
            offset = offset.npu()
            deq_scale1 = torch.from_numpy(deq_scale1.astype(np.int64)).npu()
            deq_scale2 = torch.from_numpy(deq_scale2.astype(np.int64)).npu()
            return torch_npu.npu_ffn(x, weight1, weight2, activation, scale=scale, offset=offset,
                                     deq_scale1=deq_scale1, deq_scale2=deq_scale2, inner_precise=1)
        else:
            return torch_npu.npu_ffn(x, weight1, weight2, activation, inner_precise=1)

    @SupportedDevices(['Ascend910B'])
    def test_npu_ffn(self, device="npu"):
        torch.manual_seed(0)
        x = torch.randn(8192, 320, dtype=torch.float16).npu()
        weight1 = torch.randn(320, 2560, dtype=torch.float16).npu()
        weight2 = torch.randn(2560, 320, dtype=torch.float16).npu()
        x_clone = x.clone()
        weight1_clone = weight1.clone()
        weight2_clone = weight2.clone()
        activation = "silu"

        supported_output = self.supported_op_exec(x, weight1, weight2, activation)
        custom_output = self.custom_op_exec(x_clone, weight1_clone, weight2_clone, activation)
        self.assertRtolEqual(x, x_clone, 0.001)
        self.assertRtolEqual(supported_output, custom_output, 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_npu_ffn_antiquant(self, device="npu"):
        torch.manual_seed(0)
        np.random.seed(0)
        x = torch.normal(mean=0., std=0.2, size=(8192, 320), dtype=torch.float16).npu()
        weight1 = np.random.randint(-9, 9, size=(320, 2560), dtype=np.int8)
        weight2 = np.random.randint(-9, 9, size=(2560, 320), dtype=np.int8)
        antiquant_scale1 = torch.normal(mean=0., std=0.2, size=(1, 2560), dtype=torch.float16).npu()
        antiquant_scale2 = torch.normal(mean=0., std=0.2, size=(1, 320), dtype=torch.float16).npu()
        antiquant_offset1 = torch.normal(mean=0., std=0.2, size=(1, 2560), dtype=torch.float16).npu()
        antiquant_offset2 = torch.normal(mean=0., std=0.2, size=(1, 320), dtype=torch.float16).npu()
        x_clone = x.clone()
        weight1_clone = torch.from_numpy(weight1).npu()
        weight2_clone = torch.from_numpy(weight2).npu()
        antiquant_scale1_clone = antiquant_scale1.clone()
        antiquant_scale2_clone = antiquant_scale2.clone()
        antiquant_offset1_clone = antiquant_offset1.clone()
        antiquant_offset2_clone = antiquant_offset2.clone()
        activation = "relu"

        supported_output = self.supported_op_exec(x, weight1, weight2, activation,
                                                  antiquant_scale1=antiquant_scale1,
                                                  antiquant_scale2=antiquant_scale2,
                                                  antiquant_offset1=antiquant_offset1,
                                                  antiquant_offset2=antiquant_offset2)
        custom_output = self.custom_op_exec(x_clone, weight1_clone, weight2_clone, activation,
                                            antiquant_scale1=antiquant_scale1_clone,
                                            antiquant_scale2=antiquant_scale2_clone,
                                            antiquant_offset1=antiquant_offset1_clone,
                                            antiquant_offset2=antiquant_offset2_clone)
        self.assertRtolEqual(x, x_clone, 0.001)
        self.assertRtolEqual(supported_output, custom_output, 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_npu_ffn_quant(self, device="npu"):
        torch.manual_seed(0)
        np.random.seed(0)
        x = torch.from_numpy(np.random.randint(-128, 127, size=(8192, 320), dtype=np.int8))
        weight1 = np.random.randint(-128, 127, size=(320, 2560), dtype=np.int8)
        weight2 = np.random.randint(-128, 127, size=(2560, 320), dtype=np.int8)
        scale = np.ones(1, dtype=np.float32)
        offset = np.zeros(1, dtype=np.float32)
        deq_scale1_fp32, deq_scale1_uint64 = self.deq_scale_generate((1, 2560))
        deq_scale2_fp32, deq_scale2_uint64 = self.deq_scale_generate((1, 320))

        x_clone = x.clone().npu()
        weight1_clone = torch.from_numpy(weight1)
        weight2_clone = torch.from_numpy(weight2)
        scale_clone = torch.from_numpy(scale)
        offset_clone = torch.from_numpy(offset)
        activation = "gelu"

        supported_output = self.supported_op_exec(x, weight1, weight2, activation, scale=scale, offset=offset,
                                                  deq_scale1=deq_scale1_fp32, deq_scale2=deq_scale2_fp32)
        custom_output = self.custom_op_exec(x_clone, weight1_clone, weight2_clone, activation, scale=scale_clone,
                                            offset=offset_clone, deq_scale1=deq_scale1_uint64,
                                            deq_scale2=deq_scale2_uint64)
        self.assertRtolEqual(x, x_clone, 0.001)
        self.assertRtolEqual(supported_output, custom_output, 0.001)

if __name__ == "__main__":
    run_tests()

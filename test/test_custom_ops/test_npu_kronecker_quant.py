import unittest
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import get_npu_device, SupportedDevices


class DataInfo(object):
    def __init__(self, min_d, max_d, shape_x, shape_p1, shape_p2, dtype):
        self.min_d = min_d
        self.max_d = max_d
        self.shape_x = shape_x
        self.shape_p1 = shape_p1
        self.shape_p2 = shape_p2
        self.dtype = dtype


class TestNPUFlatQuant(TestCase):

    def generate_data_npu_quantize(self, datainfo):
        input_x = np.random.uniform(datainfo.min_d, datainfo.max_d, datainfo.shape_x).astype(np.float32)
        input_p1 = np.random.uniform(datainfo.min_d, datainfo.max_d, datainfo.shape_p1).astype(np.float32)
        input_p2 = np.random.uniform(datainfo.min_d, datainfo.max_d, datainfo.shape_p2).astype(np.float32)

        npu_input_x = torch.from_numpy(input_x).to(dtype = datainfo.dtype)
        npu_input_p1 = torch.from_numpy(input_p1).to(dtype = datainfo.dtype)
        npu_input_p2 = torch.from_numpy(input_p2).to(dtype = datainfo.dtype) 

        return [npu_input_x.to("npu"), npu_input_p1.to("npu"), npu_input_p2.to("npu")]

    def golden_op_exec_kronecker_quant(self, input_x, input_p1, input_p2, clip_ratio):
        if(clip_ratio is None):
            clip_ratio = 1.0
        K, M, N = input_x.shape
        x1 = input_x @ input_p2 
        x2 = input_p1 @ x1
        x2 = x2.flatten(-2, -1)
        qscale = torch.abs(x2).max(dim=-1, keepdim=True)[0].to(torch.float)
        ratio = torch.ones_like(qscale) * 7 * clip_ratio
        qscale2 = ratio / qscale
        golden_out = (x2.to(torch.float) * qscale2).to(torch.half).to(torch.int8).reshape(K, M, N)
        golden_scale = torch.flatten(qscale / ratio).reshape(K)
        return golden_out.to("cpu").numpy(), golden_scale.to("cpu").numpy()
    
    def tensor_int32_to_int8(self, tensor_int32):
        K, M, N = tensor_int32.shape
        tensor_array = tensor_int32.reshape(1, tensor_int32.numel())[0]
        int32_array = tensor_array.view(torch.int32).cpu().numpy()
        masks = np.array([0xF << (i * 4) for i in range(8)], dtype=np.uint32)
        shifted = (int32_array[:, None] & masks) >> np.arange(0, 32, 4)
        sign_extended = np.where(shifted & 0x8, shifted - 16, shifted)
        return torch.tensor(sign_extended.astype(np.int32)).to(torch.int8).reshape(K, M, N * 8)

    def npu_op_exec_kronecker_quant(self, input_x, input_p1, input_p2):
        out, quantScale = torch_npu.npu_kronecker_quant(input_x, input_p1, input_p2)
        return self.tensor_int32_to_int8(out.to("cpu")).numpy(), quantScale.to("cpu").numpy()
    
    def npu_op_exec_kronecker_quant_ratio(self, input_x, input_p1, input_p2, clip_ratio):
        out, quantScale = torch_npu.npu_kronecker_quant(input_x, input_p1, input_p2, clip_ratio)
        return self.tensor_int32_to_int8(out.to("cpu")).numpy(), quantScale.to("cpu").numpy()

    @unittest.skip("skip test_npu_kronecker_quant_float16 now")
    @SupportedDevices(['Ascend910B'])
    def test_npu_kronecker_quant_float16(self):
        datainfo = DataInfo(1, 1, (16, 7, 16), (7, 7), (16, 16), torch.float16)
        x, p1, p2 = self.generate_data_npu_quantize(datainfo)
        golden_out, golden_scale = self.golden_op_exec_kronecker_quant(x, p1, p2, None)
        npu_out, npu_scale = self.npu_op_exec_kronecker_quant(x, p1, p2)
        self.assertRtolEqual(golden_out, npu_out)
        self.assertRtolEqual(golden_scale, npu_scale)
        
    @unittest.skip("skip test_npu_kronecker_quant_bfloat16 now")
    @SupportedDevices(['Ascend910B'])
    def test_npu_kronecker_quant_bfloat16(self):
        datainfo = DataInfo(1, 1, (16, 56, 16), (56, 56), (16, 16), torch.bfloat16)
        x, p1, p2 = self.generate_data_npu_quantize(datainfo)
        golden_out, golden_scale = self.golden_op_exec_kronecker_quant(x, p1, p2, None)
        npu_out, npu_scale = self.npu_op_exec_kronecker_quant(x, p1, p2)
        self.assertRtolEqual(golden_out, npu_out)
        self.assertRtolEqual(golden_scale, npu_scale)

    @unittest.skip("skip test_npu_kronecker_quant_float16_ratio now")
    @SupportedDevices(['Ascend910B'])
    def test_npu_kronecker_quant_float16_ratio(self):
        datainfo = DataInfo(1, 1, (16, 8, 32), (8, 8), (32, 32), torch.float16)
        x, p1, p2 = self.generate_data_npu_quantize(datainfo)
        golden_out, golden_scale = self.golden_op_exec_kronecker_quant(x, p1, p2, 0.9063)
        npu_out, npu_scale = self.npu_op_exec_kronecker_quant_ratio(x, p1, p2, 0.9063)
        self.assertRtolEqual(golden_out, npu_out)
        self.assertRtolEqual(golden_scale, npu_scale)

    @unittest.skip("skip test_npu_kronecker_quant_bfloat16_ratio now")
    @SupportedDevices(['Ascend910B'])
    def test_npu_kronecker_quant_bfloat16_ratio(self):
        datainfo = DataInfo(1, 1, (16, 3, 64), (3, 3), (64, 64), torch.bfloat16)
        x, p1, p2 = self.generate_data_npu_quantize(datainfo)
        golden_out, golden_scale = self.golden_op_exec_kronecker_quant(x, p1, p2, 0.7848)
        npu_out, npu_scale = self.npu_op_exec_kronecker_quant_ratio(x, p1, p2, 0.7848)
        self.assertRtolEqual(golden_out, npu_out)
        self.assertRtolEqual(golden_scale, npu_scale)

if __name__ == "__main__":
    run_tests()

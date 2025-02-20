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

    def golden_op_exec_kronecker_quant(self, input_x, input_p1, input_p2):
        K = input_x.shape[0]
        M = input_x.shape[1]
        N = input_x.shape[2]

        x1 = input_x @ input_p2 
        x2 = input_p1 @ x1
        x2 = x2.flatten(-2, -1)
        quantScale = x2.abs().max(dim = -1, keepdim = True)[0] / 7
        out = x2 / quantScale
        out = torch.round(out).to(torch.int32).reshape(K, M, N)

        golden_scale = quantScale.to(dtype=torch.float).reshape(K)
        golden_out = torch.zeros(K, M, N // 8).to(torch.int32)
        for n in range(N // 8):
            golden_out[:, :, n] = (
                out[:, :, n * 8] << 28 |
                out[:, :, n * 8 + 1] << 24 |
                out[:, :, n * 8 + 2] << 20 |
                out[:, :, n * 8 + 3] << 16 |
                out[:, :, n * 8 + 4] << 12 |
                out[:, :, n * 8 + 5] << 8 |
                out[:, :, n * 8 + 6] << 4 |
                out[:, :, n * 8 + 7]
            )
        return golden_out.to("cpu").numpy(), golden_scale.to("cpu").numpy()

    def npu_op_exec_kronecker_quant(self, input_x, input_p1, input_p2):
        out, quantScale = torch_npu.npu_kronecker_quant(input_x, input_p1, input_p2)
        return out.to("cpu").numpy(), quantScale.to("cpu").numpy()

    @unittest.skip("skip test_npu_kronecker_quant_float16 now")
    @SupportedDevices(['Ascend910B'])
    def test_npu_kronecker_quant_float16(self):
        datainfo = DataInfo(1, 1, (16, 7, 16), (7, 7), (16, 16), torch.float16)
        x, p1, p2 = self.generate_data_npu_quantize(datainfo)
        golden_out, golden_scale = self.golden_op_exec_kronecker_quant(x, p1, p2)
        npu_out, npu_scale = self.npu_op_exec_kronecker_quant(x, p1, p2)
        self.assertRtolEqual(golden_out, npu_out)
        self.assertRtolEqual(golden_scale, npu_scale)
        
    @unittest.skip("skip test_npu_kronecker_quant_bfloat16 now")
    @SupportedDevices(['Ascend910B'])
    def test_npu_kronecker_quant_bfloat16(self):
        datainfo = DataInfo(1, 1, (16, 56, 16), (56, 56), (16, 16), torch.bfloat16)
        x, p1, p2 = self.generate_data_npu_quantize(datainfo)
        golden_out, golden_scale = self.golden_op_exec_kronecker_quant(x, p1, p2)
        npu_out, npu_scale = self.npu_op_exec_kronecker_quant(x, p1, p2)
        self.assertRtolEqual(golden_out, npu_out)
        self.assertRtolEqual(golden_scale, npu_scale)

if __name__ == "__main__":
    run_tests()

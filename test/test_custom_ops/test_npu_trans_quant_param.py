import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestTransQuantParam(TestCase):

    def supported_op_exec(self, deq_scale_shape, fp32_deq_scale):
        uint32__deq_scale = np.frombuffer(fp32_deq_scale, np.uint32).reshape(deq_scale_shape)
        # 与高19位运算，模拟硬件
        uint32__deq_scale &= 0XFFFFE000
        # output dtype: fp16
        fp32_deq_scale = np.frombuffer(uint32__deq_scale, np.float32)
        uint64_deq_scale = np.zeros(deq_scale_shape, np.uint64)
        uint64_deq_scale |= np.uint64(uint32__deq_scale)
        uint64_deq_scale |= (1 << 46)
        int64_deq_scale = np.int64(uint64_deq_scale)
        int64_deq_scale = torch.from_numpy(int64_deq_scale)
        return int64_deq_scale

    def custom_op_exec(self, fp32_deq_scale):
        fp32_deq_scale = torch.from_numpy(fp32_deq_scale)
        return torch_npu.npu_trans_quant_param(fp32_deq_scale.npu())

    @SupportedDevices(['Ascend910B'])
    def test_npu_transquantparam(self, device="npu"):
        deq_scale_shape = (1,)
        fp32_deq_scale = np.random.uniform(low=2, high=3, size=deq_scale_shape).astype(np.float32)
        supported_output = self.supported_op_exec(deq_scale_shape, fp32_deq_scale)
        custom_output = self.custom_op_exec(fp32_deq_scale)
        self.assertRtolEqual(supported_output, custom_output, 0.001)
        
        expect_ret = torch.randint(-1, 1, (4,), dtype=torch.int64).npu()
        scale = torch.randn(1, dtype=torch.float32).npu()
        offset = torch.randn(4, dtype=torch.float32).npu()
        res = torch_npu.npu_trans_quant_param(scale, offset)
        self.assertTrue(res.shape == expect_ret.shape)
        self.assertTrue(res.dtype == expect_ret.dtype)


if __name__ == "__main__":
    run_tests()

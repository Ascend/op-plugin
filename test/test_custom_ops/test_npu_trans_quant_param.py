import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


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
        return torch_npu.npu_trans_quant_param(fp32_deq_scale.npu)

    @unittest.skipIf(DEVICE_NAME != 'Ascend910B',
        "OP `TransQuantParam` is tested on 910B(support 910B/910C), skip this ut for other device type!")
    def test_npu_transquantparam(self, device="npu"):
        deq_scale_shape = (1,)
        fp32_deq_scale = np.random.uniform(low=2, high=3, size=deq_scale_shape).astype(np.float32)
        supported_output = self.supported_op_exec(deq_scale_shape, fp32_deq_scale)
        custom_output = self.custom_op_exec(fp32_deq_scale)
        self.assertRtolEqual(supported_output, custom_output, 0.001)


if __name__ == "__main__":
    run_tests()

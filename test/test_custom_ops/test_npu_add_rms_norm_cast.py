import unittest

import numpy as np
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import get_npu_device, SupportedDevices


class TestNPUAddRmsNormCast(TestCase):

    def supported_op_exec(self, x1, x2, gamma, epsilon=1e-6):
        ori_dtype = x1.dtype
        x = x1 + x2
        if ori_dtype == np.float16:
            x = x.astype(np.float32)
            gamma = gamma.astype(np.float32)
        variance = np.mean(np.power(x, 2), axis=-1, keepdims=True)
        std = np.sqrt(variance + epsilon)
        rstd = np.divide(1, std)
        result_mid = x * rstd
        y1 = result_mid * gamma
        if ori_dtype == np.float16:
            x = x.astype(np.float16)
            y2 = y1.astype(np.float16)
        return y1, y2, rstd, x

    def custom_op_exec(self, x1, x2, gamma, epsilon=1e-6):
        y1, y2, rstd, x = torch_npu.npu_add_rms_norm_cast(x1, x2, gamma, epsilon)
        return y1.cpu().numpy(), y2.cpu().numpy(), rstd.cpu().numpy(), x.cpu().numpy()

    @unittest.skip("skip test_add_rms_norm_cast_fp16 now")
    @SupportedDevices(['Ascend910B'])
    def test_add_rms_norm_cast_fp16(self):
        cpu_input0 = np.random.uniform(0, 100, [1024, 12288]).astype(np.float16)
        cpu_input1 = np.random.uniform(0, 100, [1024, 12288]).astype(np.float16)
        cpu_input2 = np.random.uniform(0, 100, [12288]).astype(np.float16)
        npu_input0 = torch.from_numpy(cpu_input0).npu()
        npu_input1 = torch.from_numpy(cpu_input1).npu()
        npu_input2 = torch.from_numpy(cpu_input2).npu()

        supported_output0, supported_output1, supported_output2, supported_output3 = self.supported_op_exec(cpu_input0, cpu_input1, cpu_input2)
        custom_output0, custom_output1, custom_output2, custom_output3 = self.custom_op_exec(npu_input0, npu_input1, npu_input2)
        self.assertRtolEqual(supported_output0, custom_output0, 0.001)
        self.assertRtolEqual(supported_output1, custom_output1, 0.001)
        self.assertRtolEqual(supported_output2, custom_output2, 0.001)
        self.assertRtolEqual(supported_output3, custom_output3, 0.001)


if __name__ == "__main__":
    run_tests()

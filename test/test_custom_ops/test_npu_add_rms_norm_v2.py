import unittest

import numpy as np
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import get_npu_device, SupportedDevices


class TestNPUAddRmsNormV2(TestCase):

    def supported_op_exec(self, x1, x2, gamma, epsilon=1e-6):
        ori_dtype = x1.dtype
        x = x1 + x2
        if ori_dtype == np.float16:
            x = x.astype(np.float32)
        variance = np.mean(np.power(x, 2), axis=-1, keepdims=True)
        std = np.sqrt(variance + epsilon)
        rstd = np.divide(1, std)
        result_mid = x * rstd
        result = result_mid * gamma
        if ori_dtype == np.float16:
            x = x.astype(np.float16)
            result = result.astype(np.float16)
        return rstd, result, x

    def custom_op_exec_functional(self, x1, x2, gamma, epsilon=1e-6):
        rstd, x1_inplace, x2_inplace = torch_npu.npu_add_rms_norm_v2_functional(x1, x2, gamma, epsilon)
        return rstd.cpu().numpy(), x1_inplace.cpu().numpy(), x2_inplace.cpu().numpy()
    
    def custom_op_exec(self, x1, x2, gamma, epsilon=1e-6):
        rstd = torch_npu.npu_add_rms_norm_v2(x1, x2, gamma, epsilon)
        return rstd.cpu().numpy()

    @SupportedDevices(['Ascend910B'])
    def test_add_rms_norm_v2_fp32(self, device="npu"):
        if device is None:
            device = get_npu_device()
        cpu_input0 = np.random.uniform(0, 100, [1024, 12288]).astype(np.float32)
        cpu_input1 = np.random.uniform(0, 100, [1024, 12288]).astype(np.float32)
        cpu_input2 = np.random.uniform(0, 100, [12288]).astype(np.float32)
        npu_input0 = torch.from_numpy(cpu_input0).npu()
        npu_input1 = torch.from_numpy(cpu_input1).npu()
        npu_input2 = torch.from_numpy(cpu_input2).npu()
        
        rstd_cpu, result_cpu, x_cpu = self.supported_op_exec(cpu_input0, cpu_input1, cpu_input2)
        
        rstd_func, x1_inplace, x2_inplace = self.custom_op_exec_functional(npu_input0, npu_input1, npu_input2)
        self.assertRtolEqual(rstd_func, rstd_cpu, 0.0001)
        self.assertRtolEqual(x1_inplace, result_cpu, 0.0001)
        self.assertRtolEqual(x2_inplace, x_cpu, 0.0001)

        rstd_npu = self.custom_op_exec(npu_input0, npu_input1, npu_input2)
        self.assertRtolEqual(rstd_npu, rstd_cpu, 0.0001)
        self.assertRtolEqual(npu_input0.cpu().numpy(), result_cpu, 0.0001)
        self.assertRtolEqual(npu_input1.cpu().numpy(), x_cpu, 0.0001)

    @SupportedDevices(['Ascend910B'])
    def test_add_rms_norm_v2_fp16(self):
        cpu_input0 = np.random.uniform(0, 100, [1024, 12288]).astype(np.float16)
        cpu_input1 = np.random.uniform(0, 100, [1024, 12288]).astype(np.float16)
        cpu_input2 = np.random.uniform(0, 100, [12288]).astype(np.float16)
        npu_input0 = torch.from_numpy(cpu_input0).npu()
        npu_input1 = torch.from_numpy(cpu_input1).npu()
        npu_input2 = torch.from_numpy(cpu_input2).npu()
        
        rstd_cpu, result_cpu, x_cpu = self.supported_op_exec(cpu_input0, cpu_input1, cpu_input2)
        
        rstd_func, x1_inplace, x2_inplace = self.custom_op_exec_functional(npu_input0, npu_input1, npu_input2)
        self.assertRtolEqual(rstd_func, rstd_cpu, 0.0001)
        self.assertRtolEqual(x1_inplace, result_cpu, 0.0001)
        self.assertRtolEqual(x2_inplace, x_cpu, 0.0001)

        rstd_npu = self.custom_op_exec(npu_input0, npu_input1, npu_input2)
        self.assertRtolEqual(rstd_npu, rstd_cpu, 0.0001)
        self.assertRtolEqual(npu_input0.cpu().numpy(), result_cpu, 0.0001)
        self.assertRtolEqual(npu_input1.cpu().numpy(), x_cpu, 0.0001)


if __name__ == "__main__":
    run_tests()

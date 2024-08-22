import unittest
import numpy as np
import torch_npu
import torch
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import get_npu_device, SupportedDevices


class TestNPURmsNorm(TestCase):

    def supported_op_exec(self, x, gamma):
        x_fp32 = np.array(x, dtype=np.float32)
        gamma_fp32 = np.array(gamma, dtype=np.float32)

        variance = np.mean(np.power(x_fp32, 2), axis=-1, keepdims=True)
        epsilon = 1e-6
        std = np.sqrt(variance + epsilon)
        rstd = 1 / std
        result_mid = x_fp32 * rstd
        result_fp32 = result_mid * gamma_fp32

        result = np.array(result_fp32, dtype=x.dtype)

        return result, rstd

    def custom_op_exec(self, x, gamma):
        y, rstd = torch_npu.npu_rms_norm(x, gamma)
        return y.cpu().numpy(), rstd.cpu().numpy()

    @SupportedDevices(['Ascend910B'])
    def test_rms_norm(self, device="npu"):
        if device is None:
            device = get_npu_device()
        cpu_input0 = np.random.uniform(0, 100, [1024, 12288]).astype(np.float32)
        cpu_input1 = np.random.uniform(0, 100, [12288]).astype(np.float32)
        npu_input0 = torch.from_numpy(cpu_input0).to(device)
        npu_input1 = torch.from_numpy(cpu_input1).to(device)

        supported_output0, supported_output1 = self.supported_op_exec(cpu_input0, cpu_input1)
        custom_output0, custom_output1 = self.custom_op_exec(npu_input0, npu_input1)
        self.assertRtolEqual(supported_output0, custom_output0)
        self.assertRtolEqual(supported_output1, custom_output1)

    @SupportedDevices(['Ascend910B'])
    def test_rms_norm_mix_dtype(self, device="npu"):
        if device is None:
            device = get_npu_device()
        cpu_input0 = np.random.uniform(0, 100, [1024, 12288]).astype(np.float16)
        cpu_input1 = np.random.uniform(0, 100, [12288]).astype(np.float32)
        npu_input0 = torch.from_numpy(cpu_input0).to(device)
        npu_input1 = torch.from_numpy(cpu_input1).to(device)

        supported_output0, supported_output1 = self.supported_op_exec(cpu_input0, cpu_input1)
        custom_output0, custom_output1 = self.custom_op_exec(npu_input0, npu_input1)
        self.assertRtolEqual(supported_output0, custom_output0)
        self.assertRtolEqual(supported_output1, custom_output1)

if __name__ == "__main__":
    run_tests()

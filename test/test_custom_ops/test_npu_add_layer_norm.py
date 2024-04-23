import unittest
import numpy as np
import torch_npu
import torch
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import get_npu_device, SupportedDevices
torch.npu.set_compile_mode(jit_compile=False)


class AddLayerNormOutputParams:
    def __init__(self, y, mean, rstd, x):
        self.y = y
        self.mean = mean
        self.rstd = rstd
        self.x = x


class TestNPUAddLayerNorm(TestCase):
    def supported_op_exec(self, x1, x2, gamma, beta):
        if x1.dtype is not torch.float and x2.dtype is torch.float:
            x1_tensor = x1.to(torch.float)
        else:
            x1_tensor = x1
        if x1.dtype is torch.float and x2.dtype is not torch.float:
            x2_tensor = x2.to(torch.float)
        else:
            x2_tensor = x2

        epsilon = 1e-5
        x = x1_tensor + x2_tensor
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        try:
            rstd = 1 / np.sqrt(variance + epsilon)
        except ZeroDivisionError as err:
            raise err
        y = ((x - mean) * rstd) * gamma + beta
        return AddLayerNormOutputParams(y, mean, rstd, x)

    def custom_op_exec(self, x1, x2, gamma, beta):
        y, mean, rstd, x = torch_npu.npu_add_layer_norm(x1, x2, gamma, beta, 1e-5, True)
        return AddLayerNormOutputParams(y.cpu().numpy(), mean.cpu().numpy(), rstd.cpu().numpy(), x.cpu().numpy())

    @SupportedDevices(['Ascend910B'])
    def test_add_layer_norm(self, device="npu"):
        if device is None:
            device = get_npu_device()
        cpu_input_x1 = np.random.uniform(0, 100, [1, 1024, 8192]).astype(np.float32)
        cpu_input_x2 = np.random.uniform(0, 100, [1, 1024, 8192]).astype(np.float32)
        cpu_input_gamma = np.random.uniform(0, 1, [8192]).astype(np.float32)
        cpu_input_beta = np.random.uniform(0, 1, [8192]).astype(np.float32)

        npu_input_x1 = torch.from_numpy(cpu_input_x1).to(device)
        npu_input_x2 = torch.from_numpy(cpu_input_x2).to(device)
        npu_input_gamma = torch.from_numpy(cpu_input_gamma).to(device)
        npu_input_beta = torch.from_numpy(cpu_input_beta).to(device)

        supported_output = self.supported_op_exec(cpu_input_x1, cpu_input_x2, cpu_input_gamma, cpu_input_beta)
        custom_output = self.custom_op_exec(npu_input_x1, npu_input_x2, npu_input_gamma, npu_input_beta)

        self.assertRtolEqual(supported_output.y, custom_output.y)
        self.assertRtolEqual(supported_output.mean, custom_output.mean)
        self.assertRtolEqual(supported_output.rstd, custom_output.rstd)
        self.assertRtolEqual(supported_output.x, custom_output.x)

    @SupportedDevices(['Ascend910B'])
    def test_add_layer_norm_x1x2_different_dtype(self, device="npu"):
        if device is None:
            device = get_npu_device()
        cpu_input_x1 = np.random.uniform(0, 100, [1, 1024, 8192]).astype(np.float16)
        cpu_input_x2 = np.random.uniform(0, 100, [1, 1024, 8192]).astype(np.float32)
        cpu_input_gamma = np.random.uniform(0, 1, [8192]).astype(np.float32)
        cpu_input_beta = np.random.uniform(0, 1, [8192]).astype(np.float32)

        npu_input_x1 = torch.from_numpy(cpu_input_x1).to(device)
        npu_input_x2 = torch.from_numpy(cpu_input_x2).to(device)
        npu_input_gamma = torch.from_numpy(cpu_input_gamma).to(device)
        npu_input_beta = torch.from_numpy(cpu_input_beta).to(device)

        supported_output = self.supported_op_exec(cpu_input_x1, cpu_input_x2, cpu_input_gamma, cpu_input_beta)
        custom_output = self.custom_op_exec(npu_input_x1, npu_input_x2, npu_input_gamma, npu_input_beta)

        self.assertEqual(supported_output.y.dtype, "float32")
        self.assertRtolEqual(supported_output.y, custom_output.y)
        self.assertRtolEqual(supported_output.mean, custom_output.mean)
        self.assertRtolEqual(supported_output.rstd, custom_output.rstd)
        self.assertRtolEqual(supported_output.x, custom_output.x)

if __name__ == "__main__":
    run_tests()

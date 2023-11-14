import numpy as np
import torch_npu
import torch
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import get_npu_device
torch.npu.set_compile_mode(jit_compile=False)


class TestNPUDeepNorm(TestCase):
    def supported_op_exec(self, x, gx, beta, gamma):
        alpha = 0.3
        epsilon = 1e-6
        
        new_x = alpha * x + gx
        mean = np.mean(new_x, axis=-1, keepdims=True)
        diff = new_x - mean
        variance = np.mean(np.power(diff, 2), axis=-1, keepdims=True)
        std = np.sqrt(variance + epsilon)
        rstd = 1 / std
        result_mid = diff * rstd
        y = result_mid * gamma + beta
        return mean, rstd, y

    def custom_op_exec(self, x, gx, beta, gamma):
        mean, rstd, y = torch_npu.npu_deep_norm(x, gx, beta, gamma, float(0.3), 1e-6)
        return mean.cpu().numpy(), rstd.cpu().numpy(), y.cpu().numpy()

    def test_deep_norm(self, device="npu"):
        if device is None:
            device = get_npu_device()
        cpu_input_x = np.random.uniform(0, 100, [1024, 12288]).astype(np.float32)
        cpu_input_gx = np.random.uniform(0, 100, [1024, 12288]).astype(np.float32)
        cpu_input_beta = np.random.uniform(0, 100, [12288]).astype(np.float32)
        cpu_input_gamma = np.random.uniform(0, 100, [12288]).astype(np.float32)
        
        npu_input_x = torch.from_numpy(cpu_input_x).to(device)
        npu_input_gx = torch.from_numpy(cpu_input_gx).to(device)
        npu_input_beta = torch.from_numpy(cpu_input_beta).to(device)
        npu_input_gamma = torch.from_numpy(cpu_input_gamma).to(device)

        supported_mean, supported_rstd, supported_y = self.supported_op_exec(cpu_input_x, cpu_input_gx, cpu_input_beta, cpu_input_gamma)
        custom_mean, custom_rstd, custom_y = self.custom_op_exec(npu_input_x, npu_input_gx, npu_input_beta, npu_input_gamma)
        
        self.assertRtolEqual(supported_mean, custom_mean)
        self.assertRtolEqual(supported_rstd, custom_rstd)
        self.assertRtolEqual(supported_y, custom_y)

if __name__ == "__main__":
    run_tests()

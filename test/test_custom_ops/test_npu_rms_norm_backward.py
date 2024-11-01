import unittest
import numpy as np
import torch_npu
import torch
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import get_npu_device, SupportedDevices


class TestNPURmsNormBackward(TestCase):

    def supported_op_exec(self, x, gamma, grad_y):
        variance = np.mean(np.power(x, 2), axis=-1, keepdims=True)
        epsilon = 1e-6
        std = np.sqrt(variance + epsilon)
        for _, e in enumerate(std):  
            if e == 0:
                e = epsilon
        rstd = 1 / std
        np.broadcast(rstd, grad_y)
        np.broadcast(gamma, grad_y)
        dgamma = np.sum(grad_y * x * rstd, 0, keepdims=True)
        dxp1 = grad_y * gamma * rstd
        dxp2 = x * np.sum(rstd * rstd * rstd * grad_y * gamma * x, 1, keepdims=True) / 128
        dx = dxp1 - dxp2
        dgamma = np.reshape(dgamma, (128))
        return dx, dgamma

    def custom_op_exec(self, npu_input0, npu_input1, npu_grad_y):
        npu_input0.requires_grad = True
        npu_input1.requires_grad = True
        setattr(npu_input1, 'sequence_parallel', False)
        out = torch_npu.npu_rms_norm(npu_input0, npu_input1)[0]
        out.backward(npu_grad_y)
        dx = npu_input0.grad
        dw = npu_input1.grad
        out = out.to(torch.float32).float().cpu()
        dx = dx.float().cpu()
        dw = dw.float().cpu()
        return dx.numpy(), dw.numpy()

    @SupportedDevices(['Ascend910B'])
    def test_npu_rms_norm_backward(self, device="npu"):
        if device is None:
            device = get_npu_device()

        cpu_input0 = np.random.uniform(0, 100, [10, 128]).astype(np.float32)
        cpu_input1 = np.random.uniform(0, 100, [128]).astype(np.float32)
        grad_y = np.random.uniform(0, 100, [10, 128]).astype(np.float32)
        npu_input0 = torch.from_numpy(cpu_input0).to(device)
        npu_input1 = torch.from_numpy(cpu_input1).to(device)
        npu_grad_y = torch.from_numpy(grad_y).to(device)

        supported_output0, supported_output1 = self.supported_op_exec(cpu_input0, cpu_input1, grad_y)
        custom_output0, custom_output1 = self.custom_op_exec(npu_input0, npu_input1, npu_grad_y)
        self.assertRtolEqual(supported_output0, custom_output0)
        self.assertRtolEqual(supported_output1, custom_output1)


if __name__ == "__main__":
    run_tests()

import unittest
import numpy as np
import torch_npu
import torch
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import get_npu_device, SupportedDevices
torch.npu.set_compile_mode(jit_compile=False)


class DeepNormGradInputParams:
    def __init__(self, dy, x, gx, gamma):
        self.dy = dy
        self.x = x
        self.gx = gx
        self.gamma = gamma


class DeepNormGradOutputParams:
    def __init__(self, dx, dgx, dbeta, dgamma):
        self.dx = dx
        self.dgx = dgx
        self.dbeta = dbeta
        self.dgamma = dgamma


class TestNPUDeepNormBackward(TestCase):
    def supported_op_exec(self, dy, x, gx, gamma, alpha):
        epsilon = 1e-6

        reduce_axis = len(dy.shape) - 1
        value_D = dy.shape[-1]

        # pre cal
        x_sum = alpha * x + gx
        input_var = np.var(x_sum, axis=-1, keepdims=True)
        mean = np.mean(x_sum, axis=-1, keepdims=True).astype(np.float32)
        rstd = np.power((input_var + epsilon), (-0.5)).astype(np.float32)

        # main cal
        pd_xl = dy * gamma
        x2_tensor = x_sum - mean

        pd_var_first_part = (-0.5) * pd_xl * x2_tensor * np.power(rstd, 3)
        pd_var = np.sum(pd_var_first_part, reduce_axis, keepdims=True)

        pd_mean = np.sum((-1.0) * pd_xl * rstd, reduce_axis, keepdims=True)

        pd_x_first_part = pd_xl * rstd
        try:
            pd_x_second_part = pd_var * (2.0 / value_D) * x2_tensor
            pd_x_thrid_part = pd_mean * (1.0 / value_D)
        except ZeroDivisionError as err:
            raise err

        pd_gx = pd_x_first_part + pd_x_second_part + pd_x_thrid_part
        pd_x = alpha * pd_gx
        pd_gamma = np.sum(dy * x2_tensor * rstd, axis=0, keepdims=True)
        pd_beta = np.sum(dy, axis=0, keepdims=True)

        for n in range(len(pd_gamma.shape) - 1):
            pd_gamma = np.sum(pd_gamma, axis=n, keepdims=False)
            pd_beta = np.sum(pd_beta, axis=n, keepdims=False)

        return DeepNormGradOutputParams(pd_x, pd_gx, pd_beta, pd_gamma)


    def custom_op_exec(self, beta, alpha, deepnormgrad_input: DeepNormGradInputParams):
        dy = deepnormgrad_input.dy
        x = deepnormgrad_input.x
        gx = deepnormgrad_input.gx
        gamma = deepnormgrad_input.gamma

        x.requires_grad = True
        gx.requires_grad = True
        beta.requires_grad = True
        gamma.requires_grad = True
        setattr(beta, 'sequence_parallel', False)
        setattr(gamma, 'sequence_parallel', False)
        _, _, y = torch_npu.npu_deep_norm(x, gx, beta, gamma, alpha, 1e-6)

        y.backward(dy)
        dx = x.grad
        dgx = gx.grad
        dbeta = beta.grad
        dgamma = gamma.grad

        y = y.to(torch.float32).float().cpu()
        dx = dx.float().cpu()
        dgx = dgx.float().cpu()
        dbeta = dbeta.float().cpu()
        dgamma = dgamma.float().cpu()

        return DeepNormGradOutputParams(dx.numpy(), dgx.numpy(),
                                        dbeta.numpy(), dgamma.numpy())

    @SupportedDevices(['Ascend910B'])
    def test_deep_norm_backward_base(self, device="npu"):
        if device is None:
            device = get_npu_device()

        cpu_input_dy = np.random.uniform(0, 1, [48, 2048]).astype(np.float32)
        cpu_input_x = np.random.uniform(0, 1, [48, 2048]).astype(np.float32)
        cpu_input_gx = np.random.uniform(0, 1, [48, 2048]).astype(np.float32)
        cpu_input_beta = np.random.uniform(0, 1, [2048]).astype(np.float32)
        cpu_input_gamma = np.random.uniform(0, 1, [2048]).astype(np.float32)

        npu_input_dy = torch.from_numpy(cpu_input_dy).to(device)
        npu_input_x = torch.from_numpy(cpu_input_x).to(device)
        npu_input_gx = torch.from_numpy(cpu_input_gx).to(device)
        npu_input_beta = torch.from_numpy(cpu_input_beta).to(device)
        npu_input_gamma = torch.from_numpy(cpu_input_gamma).to(device)

        alpha = 0.3

        supported_output = self.supported_op_exec(cpu_input_dy, cpu_input_x,
                                                  cpu_input_gx, cpu_input_gamma, alpha)

        deepnormgrad_input = DeepNormGradInputParams(npu_input_dy, npu_input_x,
                                                     npu_input_gx, npu_input_gamma)

        custom_output = self.custom_op_exec(npu_input_beta, alpha, deepnormgrad_input)

        self.assertRtolEqual(supported_output.dx, custom_output.dx)
        self.assertRtolEqual(supported_output.dgx, custom_output.dgx)
        self.assertRtolEqual(supported_output.dbeta, custom_output.dbeta)
        self.assertRtolEqual(supported_output.dgamma, custom_output.dgamma)

    @SupportedDevices(['Ascend910B'])
    def test_deep_norm_backward_different_alpha(self, device="npu"):
        if device is None:
            device = get_npu_device()

        cpu_input_dy = np.random.uniform(0, 1, [48, 2048]).astype(np.float32)
        cpu_input_x = np.random.uniform(0, 1, [48, 2048]).astype(np.float32)
        cpu_input_gx = np.random.uniform(0, 1, [48, 2048]).astype(np.float32)
        cpu_input_beta = np.random.uniform(0, 1, [2048]).astype(np.float32)
        cpu_input_gamma = np.random.uniform(0, 1, [2048]).astype(np.float32)

        npu_input_dy = torch.from_numpy(cpu_input_dy).to(device)
        npu_input_x = torch.from_numpy(cpu_input_x).to(device)
        npu_input_gx = torch.from_numpy(cpu_input_gx).to(device)
        npu_input_beta = torch.from_numpy(cpu_input_beta).to(device)
        npu_input_gamma = torch.from_numpy(cpu_input_gamma).to(device)

        alpha = 1

        supported_output = self.supported_op_exec(cpu_input_dy, cpu_input_x,
                                                  cpu_input_gx, cpu_input_gamma, alpha)

        deepnormgrad_input = DeepNormGradInputParams(npu_input_dy, npu_input_x,
                                                     npu_input_gx, npu_input_gamma)

        custom_output = self.custom_op_exec(npu_input_beta, alpha, deepnormgrad_input)

        self.assertRtolEqual(supported_output.dx, custom_output.dx)
        self.assertRtolEqual(supported_output.dgx, custom_output.dgx)
        self.assertRtolEqual(supported_output.dbeta, custom_output.dbeta)
        self.assertRtolEqual(supported_output.dgamma, custom_output.dgamma)

if __name__ == "__main__":
    run_tests()


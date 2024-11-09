import unittest
import numpy as np
import torch_npu
import torch
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import get_npu_device, SupportedDevices


class TestNPUAddLayerNormBackward(TestCase):
    def supported_op_exec(self, input_x1, input_x2, gamma_fp32, beta_fp32, grad_y):
        reduce_axis = -1
        eps = 1e-6
        x_fp32 = input_x1.astype(np.float32) + input_x2.astype(np.float32)
        mean_fp32 = np.mean(x_fp32, reduce_axis, keepdims=True)
        var_fp32 = np.var(x_fp32, reduce_axis, keepdims=True)
        var_fp32_sqrt = np.sqrt(var_fp32 + eps)
        var_fp32_sqrt = np.where(var_fp32_sqrt == 0, 1, var_fp32_sqrt)
        rstd_fp32 = 1 / var_fp32_sqrt
        y_fp32 = gamma_fp32 * ((x_fp32 - mean_fp32) * rstd_fp32) + beta_fp32

        # rstd = np.power((inputVariace + EPSLON), (-0.5))

        N, _, D = grad_y.shape

        pd_xl = grad_y * gamma_fp32
        x2_tensor = x_fp32 - mean_fp32

        pd_var_first_part = (-0.5) * pd_xl * x2_tensor * np.power(rstd_fp32, 3)
        pd_var = np.sum(pd_var_first_part, reduce_axis, keepdims=True)

        pd_mean_first_part = np.sum(((-1.0) * pd_xl * rstd_fp32), reduce_axis, keepdims=True)
        pd_mean = pd_mean_first_part

        pd_x_first_part = pd_xl * rstd_fp32
        if (D == 0):
            pd_x_second_part = pd_var * 2.0 * x2_tensor
            pd_x_thrid_part = pd_mean * 1.0
        else:
            pd_x_second_part = pd_var * (2.0 / D) * x2_tensor
            pd_x_thrid_part = pd_mean * (1.0 / D)
        pd_x = pd_x_first_part + pd_x_second_part + pd_x_thrid_part
        # res_for_gamma = x2Tensor * rstd
        dgamma = np.sum(grad_y * x2_tensor * rstd_fp32, axis=(0, 1), keepdims=True)
        dbeta = np.sum(grad_y, axis=(0, 1), keepdims=True)

        # pd_x = pd_x_first_part  #
        return pd_x, dgamma.flatten(), dbeta.flatten()

    def custom_op_exec(self, x0, x1, gamma, beta, npu_grad_y):
        x0.requires_grad = True
        x1.requires_grad = True
        gamma.requires_grad = True
        beta.requires_grad = True
        # setattr(npu_input1, 'sequence_parallel', False)
        out = torch_npu.npu_add_layer_norm(x0, x1, gamma, beta)[0]
        out.backward(npu_grad_y)
        dx0 = x0.grad
        dx1 = x1.grad
        dgamma = gamma.grad
        dbeta = beta.grad

        out = out.to(torch.float32).float().cpu()
        dx0 = dx0.float().cpu()
        dx1 = dx1.float().cpu()
        dgamma = dgamma.float().cpu()
        dbeta = dbeta.float().cpu()
        return dx0.numpy(), dgamma.numpy(), dbeta.numpy()

    @SupportedDevices(['Ascend910B'])
    def test_npu_add_layer_norm_backward(self, device="npu"):
        if device is None:
            device = get_npu_device()

        cpu_input0 = np.random.uniform(0, 100, [10, 2, 128]).astype(np.float32)
        cpu_input1 = np.random.uniform(0, 100, [10, 2, 128]).astype(np.float32)
        cpu_input2 = np.random.uniform(0, 100, [128]).astype(np.float32)
        cpu_input3 = np.random.uniform(0, 100, [128]).astype(np.float32)
        grad_y = np.random.uniform(0, 100, [10, 2, 128]).astype(np.float32)
        npu_input0 = torch.from_numpy(cpu_input0).to(device)
        npu_input1 = torch.from_numpy(cpu_input1).to(device)
        npu_input2 = torch.from_numpy(cpu_input2).to(device)
        npu_input3 = torch.from_numpy(cpu_input3).to(device)
        npu_grad_y = torch.from_numpy(grad_y).to(device)

        (
            supported_output0,
            supported_output1,
            supported_output2,
        ) = self.supported_op_exec(cpu_input0, cpu_input1, cpu_input2, cpu_input3, grad_y)
        custom_output0, custom_output1, custom_output2 = self.custom_op_exec(
            npu_input0, npu_input1, npu_input2, npu_input3, npu_grad_y
        )
        self.assertRtolEqual(supported_output0, custom_output0)
        self.assertRtolEqual(supported_output1, custom_output1)
        self.assertRtolEqual(supported_output2, custom_output2)


if __name__ == "__main__":
    run_tests()

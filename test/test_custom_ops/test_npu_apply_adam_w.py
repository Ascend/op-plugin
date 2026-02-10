import math
import unittest
import copy
import struct
from struct import pack, unpack
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

def gen_apply_adam_w(var, m, v, grad, max_grad_norm, beta1_power, beta2_power, lr, weight_decay, 
                     beta1, beta2, epsilon, amsgrad=True, maximize=False):
    if maximize:
        gt = -1 * grad
    else:
        gt = grad
        m_out = m * beta1 - (beta1 + (-1)) * gt
        v_out = v * beta2 - (beta2 + (-1)) * gt * gt

    var_t = var * (1 + (-lr * weight_decay))
    beta1_power_out = beta1_power * beta1
    beta2_power_out = beta2_power * beta2

    if amsgrad:
        max_grad_norm_out = np.maximum(max_grad_norm, v_out)
        denom = np.sqrt(max_grad_norm_out / (1 - beta2_power_out)) + epsilon
    else:
        max_grad_norm_out = None
        denom = np.sqrt(v_out / (1 - beta2_power_out)) + epsilon
    var_out = var_t + (-lr * m_out / (1 - beta1_power_out)) / denom
    return var_out, m_out, v_out

class TestQuantScatter(TestCase):
    @SupportedDevices(['Ascend950'])
    def test_npu_apply_adam_w(self, device="npu"):
        var_data = np.random.uniform(0, 1, [20000]).astype(np.float32)
        var1 = torch.from_numpy(var_data).to(torch.float32).npu()

        m_data = np.random.uniform(0, 1, [20000]).astype(np.float32)
        m1 = torch.from_numpy(m_data).to(torch.float32).npu()

        v_data = np.random.uniform(0, 1, [20000]).astype(np.float32)
        v1 = torch.from_numpy(v_data).to(torch.float32).npu()

        grad_data = np.random.uniform(0, 1, [20000]).astype(np.float32)
        grad1 = torch.from_numpy(grad_data).to(torch.float32).npu()

        max_grad_norm = np.random.uniform(0, 1, [20000]).astype(np.float32)
        max_grad_norm1 = torch.from_numpy(max_grad_norm).to(torch.float32).npu()

        beta1_power = np.array([0.00001]).astype(np.float32)
        beta2_power = np.array([0.00001]).astype(np.float32)
        lr = np.array([0.00001]).astype(np.float32)
        weight_decay = np.array([0.00001]).astype(np.float32)
        beta1 = np.array([0.00001]).astype(np.float32)
        beta2 = np.array([0.00001]).astype(np.float32)
        epsilon = np.array([0.00001]).astype(np.float32)
        beta1_power1 = torch.from_numpy(beta1_power).to(torch.float32).npu()
        beta2_power1 = torch.from_numpy(beta2_power).to(torch.float32).npu()
        lr1 = torch.from_numpy(lr).to(torch.float32).npu()
        weight_decay1 = torch.from_numpy(weight_decay).to(torch.float32).npu()
        beta11 = torch.from_numpy(beta1).to(torch.float32).npu()
        beta21 = torch.from_numpy(beta2).to(torch.float32).npu()
        epsilon1 = torch.from_numpy(epsilon).to(torch.float32).npu()


        supported_output, m_result,  v_result= gen_apply_adam_w(var_data, m_data, v_data, grad_data, max_grad_norm, beta1_power,
                                            beta2_power, lr, weight_decay, beta1, beta2, epsilon)
        torch_npu.npu_apply_adam_w_out(beta1_power1, beta2_power1, lr1, weight_decay1, beta11, beta21, epsilon1,
                                                       grad1, max_grad_norm1, True, False, var1, m1, v1)
        self.assertRtolEqual(supported_output.cpu().numpy(), var1.cpu().numpy(), 0.001)

if __name__ == "__main__":
    run_tests()
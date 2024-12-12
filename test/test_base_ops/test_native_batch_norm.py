import torch
import numpy as np

import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestNativeBatchNormLegit(TestCase):
    # pylint:disable = huawei-too-many-arguments
    def cpu_op_exec(self, input1, weight, bias, running_mean, running_var, training, momentum, eps):
        cpu_output = torch._native_batch_norm_legit(input1, weight, bias, running_mean, running_var, training, momentum,
            eps=eps)
        return cpu_output

    # pylint:disable = huawei-too-many-arguments
    def npu_op_exec(self, input1, weight, bias, running_mean, running_var, training, momentum, eps):
        npu_output = torch._native_batch_norm_legit(input1, weight, bias, running_mean, running_var, training, momentum,
            eps)
        return npu_output

    def test_native_batch_norm_legit(self):
        input1 = torch.randn(5, 3, 10, 10)
        weight = torch.ones(3)
        bias = torch.zeros(3)
        running_mean = torch.zeros(3)
        running_var = torch.ones(3)
        training = True
        momentum = 0.1
        eps = 1e-5
        cpu_output = self.cpu_op_exec(input1, weight, bias, running_mean, running_var, training, momentum, eps)
        npu_output = self.npu_op_exec(input1.npu(), weight.npu(), bias.npu(), running_mean.npu(), running_var.npu(), training, momentum, eps)
        self.assertRtolEqual(cpu_output[0], npu_output[0])
        self.assertRtolEqual(cpu_output[1], npu_output[1])


if __name__ == "__main__":
    run_tests()

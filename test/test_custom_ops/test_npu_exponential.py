import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices
from scipy.stats import kstest


class TestExponential(TestCase):
    def cal_reject_num(self, alpha, n):
        z = -3.0902
        rate = float((1 - alpha) + z * pow((1 - alpha) * np.divide(alpha, n, where=n != 0), 0.5))
        reject_num = (1 - rate) * n
        return reject_num

    def supported_op_exec(self, tensor_cpu):
        np_cpu = tensor_cpu.exponential_().numpy()
        return np_cpu

    def custom_op_exec(self, tensor_cpu):
        tensor_npu = tensor_cpu.npu()
        tensor_npu = tensor_npu.exponential_()
        return tensor_npu

    @SupportedDevices(['Ascend910B'])
    def test_npu_exponential(self, device="npu"):
        N = 100
        alpha = 0.01
        count = 0
        for i in range(N):
            k = np.random.randint(1, 5)
            shape = tuple(np.random.randint(10, 100, size=(k, )))
            tensor_cpu = torch.rand(size=shape)
            np_cpu = self.supported_op_exec(tensor_cpu)
            tensor_npu = self.custom_op_exec(tensor_cpu)
            np_npu = tensor_npu.cpu().numpy()

            test_output = kstest(np_cpu.flatten(), np_npu.flatten())
            if test_output.pvalue < alpha:
                count += 1
    
        reject_num = self.cal_reject_num(alpha, N)
        if count <= reject_num:
            return True
        else:
            return False
    
    @SupportedDevices(['Ascend910B'])
    def test_exponential_no_zero(self):
        for _ in range(100):
            x = torch.empty(500000000, device="npu", dtype=torch.float32).exponential_()
            self.assertTrue(x.min() > 0)

    @SupportedDevices(['Ascend910B'])
    def test_exponential_negative_lambda_fails(self):
        with self.assertRaises(RuntimeError):
            torch.empty((1,), device="npu", dtype=torch.float32).exponential_(-0.5)


if __name__ == "__main__":
    run_tests()

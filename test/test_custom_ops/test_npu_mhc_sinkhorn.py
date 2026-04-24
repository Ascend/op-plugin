import itertools
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices
import itertools


class TestNpuMhcSinkhorn(TestCase):
   
    def cpu_op_exec(self, x, eps, num_iters, out_flag):
        y = torch.softmax(x, dim=-1) + eps
        y = y / (y.sum(dim=-2, keepdim=True) + eps)
        for i in range(max(num_iters - 1, 0)):
            # Row Norm
            y = y / (y.sum(dim=-1, keepdim=True) + eps)
            # Col Norm
            y = y / (y.sum(dim=-2, keepdim=True) + eps)
        return y
   
    def custom_op_exec(self, x, eps, num_iters, out_flag):
        y, norm_out, sum_out = torch_npu.npu_mhc_sinkhorn(x, eps=eps, num_iters=num_iters, out_flag=out_flag)
        return y
    
    @SupportedDevices(['Ascend950'])
    def test_npu_mhc_sinkhorn(self, device="npu"):
        x_shape = [1, 128, 4, 4]
        x = torch.randn(x_shape, dtype=torch.float32)
        eps = 1e-6
        num_iters = 20
        out_flag = 0

        expected_output = self.cpu_op_exec(x, eps, num_iters, out_flag)
        custom_output = self.custom_op_exec(x.npu(), eps, num_iters, out_flag)
        self.assertRtolEqual(expected_output.numpy(), custom_output.cpu().numpy())


if __name__ == "__main__":
    run_tests()        
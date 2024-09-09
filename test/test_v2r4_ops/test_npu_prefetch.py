import os
import unittest
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class TestNpuPrefetch(TestCase):

    @SupportedDevices(['Ascend910B'])
    def npu_op_exec(self, src, ctrl, max_size):
        torch_npu.npu_prefetch(src, ctrl, max_size)
        output = src.numpy()
        return output

    @SupportedDevices(['Ascend910B'])
    def test_npu_prefetch_base(self):
        input_shape = [np.float16, -1, (8, 8)]
        original_input1, npu_input1 = create_common_tensor(input_shape, -100, 100)
        original_input2, npu_input2 = create_common_tensor(input_shape, -100, 100)
        original_output1 = original_input1.numpy()
        original_output2 = original_input2.numpy()
        npu_output1 = self.npu_op_exec(npu_input1, None, 256)
        npu_output2 = self.npu_op_exec(npu_input2, None, 1000)
        self.assertRtolEqual(original_output1, npu_output1)
        self.assertRtolEqual(original_output2, npu_output2)
   
    @SupportedDevices(['Ascend910B'])
    def test_npu_prefetch_dependency(self):
        input_shape = [np.float16, -1, (8, 8)]
        original_input1, npu_input1 = create_common_tensor(input_shape, -100, 100)
        original_input2, npu_input2 = create_common_tensor(input_shape, -100, 100)
        _, dependency = create_common_tensor(input_shape, -100, 100)
        original_output1 = original_input1.numpy()
        original_output2 = original_input2.numpy()
        npu_output1 = self.npu_op_exec(npu_input1, dependency, 256)
        npu_output2 = self.npu_op_exec(npu_input2, dependency, 1000)
        self.assertRtolEqual(original_output1, npu_output1)
        self.assertRtolEqual(original_output2, npu_output2)

    @SupportedDevices(['Ascend910B'])
    def test_npu_prefetch_max_size_is_negative(self):
        input_shape = [np.float16, -1, (8, 8)]
        _, npu_input = create_common_tensor(input_shape, -100, 100)
        with self.assertRaises(RuntimeError) as cm:
            npu_output = self.npu_op_exec(npu_input, None, -1)
        exception = cm.exception
        self.assertEqual(str(exception), "kernel size should be greater than zero, but got -1")

if __name__ == "__main__":
    run_tests()

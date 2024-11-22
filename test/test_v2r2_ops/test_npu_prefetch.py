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
    def npu_op_exec(self, src, ctrl, max_size, offset=0):
        torch_npu.npu_prefetch(src, ctrl, max_size, offset)
        output = src.cpu().numpy()
        return output

    @SupportedDevices(['Ascend910B'])
    def test_npu_prefetch_base(self):
        input_shape = [np.float16, -1, (8, 8)]
        original_input1, npu_input1 = create_common_tensor(input_shape, -100, 100)
        original_input2, npu_input2 = create_common_tensor(input_shape, -100, 100)
        original_input3, npu_input3 = create_common_tensor(input_shape, -100, 100)
        original_output1 = original_input1.numpy()
        original_output2 = original_input2.numpy()
        original_output3 = original_input3.numpy()
        npu_output1 = self.npu_op_exec(npu_input1, None, 256)
        npu_output2 = self.npu_op_exec(npu_input2, None, 1000)
        npu_output3 = self.npu_op_exec(npu_input3, None, 256, 10)
        self.assertRtolEqual(original_output1, npu_output1)
        self.assertRtolEqual(original_output2, npu_output2)
        self.assertRtolEqual(original_output3, npu_output3)
   
    @SupportedDevices(['Ascend910B'])
    def test_npu_prefetch_dependency(self):
        input_shape = [np.float16, -1, (8, 8)]
        original_input1, npu_input1 = create_common_tensor(input_shape, -100, 100)
        original_input2, npu_input2 = create_common_tensor(input_shape, -100, 100)
        original_input3, npu_input3 = create_common_tensor(input_shape, -100, 100)
        _, dependency = create_common_tensor(input_shape, -100, 100)
        original_output1 = original_input1.numpy()
        original_output2 = original_input2.numpy()
        original_output3 = original_input3.numpy()
        npu_output1 = self.npu_op_exec(npu_input1, dependency, 256)
        npu_output2 = self.npu_op_exec(npu_input2, dependency, 1000)
        npu_output3 = self.npu_op_exec(npu_input3, dependency, 256, 10)
        self.assertRtolEqual(original_output1, npu_output1)
        self.assertRtolEqual(original_output2, npu_output2)
        self.assertRtolEqual(original_output3, npu_output3)

        # test NZ, sizes: [8, 8], storage_sizes: [1, 1, 16, 16]
        input_shape_nz = [np.float32, 29, (8, 8)]
        original_input, npu_input = create_common_tensor(input_shape_nz, -100, 100)
        original_output = original_input.numpy()
        npu_output = self.npu_op_exec(npu_input, None, 256, 512)
        self.assertRtolEqual(original_output, npu_output)

    @SupportedDevices(['Ascend910B'])
    def test_npu_prefetch_max_size_is_negative(self):
        input_shape = [np.float16, -1, (8, 8)]
        _, npu_input = create_common_tensor(input_shape, -100, 100)
        with self.assertRaises(RuntimeError) as cm:
            npu_output = self.npu_op_exec(npu_input, None, -1)
        exception = cm.exception
        self.assertTrue("max_size should be greater than zero, but got -1" in str(exception))

    @SupportedDevices(['Ascend910B'])
    def test_npu_prefetch_offset_exception(self):
        input_shape = [np.float16, -1, (8, 8)]
        _, npu_input = create_common_tensor(input_shape, -100, 100)
        with self.assertRaises(RuntimeError) as cm:
            npu_output = self.npu_op_exec(npu_input, None, 1000, -1)
        exception = cm.exception
        self.assertTrue("offset should not be smaller than zero, but got -1" in str(exception))
        with self.assertRaises(RuntimeError) as cm:
            npu_output = self.npu_op_exec(npu_input, None, 1000, 1000)
        exception = cm.exception
        self.assertTrue("offset out of range of tensor size, tensor size: 128, offset: 1000" in str(exception))

if __name__ == "__main__":
    run_tests()

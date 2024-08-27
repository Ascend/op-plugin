import unittest
import random
import torch
import torch_npu
import hypothesis
import numpy as np
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestForeachTrunc(TestCase):

    torch_dtypes = {
        "float16" : torch.float16,
        "float32" : torch.float32,
        "bfloat16" : torch.bfloat16,
    }

    def assert_equal(self, cpu_outs, npu_outs):
        for cpu_out, npu_out in zip(cpu_outs, npu_outs):
            if (cpu_out.shape != npu_out.shape):
                self.fail("shape error")
            if (cpu_out.dtype != npu_out.dtype):
                self.fail("dtype error!")
            result = torch.allclose(cpu_out, npu_out.cpu(), rtol=0.001, atol=0.001)
            if not result:
                self.fail("result error!")
        return True
    
    def create_tensors(self, tensor_nums, dtype):
        cpu_tensors = []
        npu_tensors = []
        for i in range(tensor_nums):
            m = random.randint(1, 100)
            n = random.randint(1, 100)
            t = torch.randn((m, n), dtype=self.torch_dtypes.get(dtype))
            cpu_tensors.append(t)
            npu_tensors.append(t.npu())
        return tuple(cpu_tensors), tuple(npu_tensors)

    
    def test_foreach_trunc_out_float32_shpae_tensor_num(self):
        tensor_num_list = [12, 62]
        for tensor_num in tensor_num_list :
            cpu_tensors, npu_tensors = self.create_tensors(tensor_num, "float32")
            cpu_output = torch._foreach_trunc(cpu_tensors)

            npu_output = torch._foreach_trunc(npu_tensors)

            self.assertRtolEqual(cpu_output, npu_output)
    
    
    def test_foreach_trunc_out_float16_shpae_tensor_num(self):
        tensor_num_list = [12, 62]
        for tensor_num in tensor_num_list :
            cpu_tensors, npu_tensors = self.create_tensors(tensor_num, "float16")
            cpu_tensors = [cpu_tensor.numpy() for cpu_tensor in cpu_tensors]
            cpu_output = [torch.from_numpy(np.trunc(cpu_tensors[i])) for i in range(len(cpu_tensors))]
            npu_output = torch._foreach_trunc(npu_tensors)

            self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(['Ascend910B'])
    def test_foreach_trunc_out_bfloat16_shpae_tensor_num(self):
        tensor_num_list = [12, 62]
        for tensor_num in tensor_num_list :
            cpu_tensors, npu_tensors = self.create_tensors(tensor_num, "bfloat16")
            cpu_output = torch._foreach_trunc(cpu_tensors)
            npu_output = torch._foreach_trunc(npu_tensors)

            self.assert_equal(cpu_output, npu_output)

    
    def test_foreach_trunc_inplace_float32_shpae_tensor_num(self):
        tensor_num_list = [12, 62]
        for tensor_num in tensor_num_list :
            cpu_tensors, npu_tensors = self.create_tensors(tensor_num, "float32")
            torch._foreach_trunc_(cpu_tensors)
            torch._foreach_trunc_(npu_tensors)

            self.assertRtolEqual(cpu_tensors, npu_tensors)
    
    
    def test_foreach_trunc_inplace_float16_shpae_tensor_num(self):
        tensor_num_list = [12, 62]
        for tensor_num in tensor_num_list :
            cpu_tensors, npu_tensors = self.create_tensors(tensor_num, "float16")
            cpu_tensors = [cpu_tensor.numpy() for cpu_tensor in cpu_tensors]
            cpu_output = [torch.from_numpy(np.trunc(cpu_tensors[i])) for i in range(len(cpu_tensors))]
            torch._foreach_trunc_(npu_tensors)

            self.assertRtolEqual(cpu_output, npu_tensors)

    @SupportedDevices(['Ascend910B'])
    def test_foreach_trunc_inplace_bfloat16_shpae_tensor_num(self):
        tensor_num_list = [12, 62]
        for tensor_num in tensor_num_list :
            cpu_tensors, npu_tensors = self.create_tensors(tensor_num, "bfloat16")
            torch._foreach_trunc_(cpu_tensors)
            torch._foreach_trunc_(npu_tensors)

            self.assert_equal(cpu_tensors, npu_tensors)


if __name__ == "__main__":
    run_tests()

import unittest
import random
import torch
import torch_npu
import hypothesis
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestForeachNeg(TestCase):

    torch_dtypes = {
        "float16" : torch.float16,
        "float32" : torch.float32,
        "bfloat16" : torch.bfloat16,
        "int32" : torch.int32
    }

    def create_tensors(self, tensor_nums, dtype):
        cpu_tensors = []
        npu_tensors = []
        for i in range(tensor_nums):
            m = random.randint(1, 100)
            n = random.randint(1, 100)
            if (dtype == "int32"):
                t = torch.randint(low=-100, high=100, size=(m, n), dtype=self.torch_dtypes.get(dtype))
            else:
                t = torch.randn((m, n), dtype=self.torch_dtypes.get(dtype))
            cpu_tensors.append(t)
            npu_tensors.append(t.npu())
        return tuple(cpu_tensors), tuple(npu_tensors)

    
    def test_foreach_neg_out_float32_shpae_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list :
            cpu_tensors, npu_tensors = self.create_tensors(tensor_num, "float32")
            cpu_output = torch._foreach_neg(cpu_tensors)
            npu_output = torch._foreach_neg(npu_tensors)

            self.assertRtolEqual(cpu_output, npu_output)
    
    
    def test_foreach_neg_out_float16_shpae_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list :
            cpu_tensors, npu_tensors = self.create_tensors(tensor_num, "float16")
            cpu_output = torch._foreach_neg(cpu_tensors)
            npu_output = torch._foreach_neg(npu_tensors)

            self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(['Ascend910B'])
    def test_foreach_neg_out_bfloat16_shpae_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list :
            cpu_tensors, npu_tensors = self.create_tensors(tensor_num, "bfloat16")
            cpu_output = torch._foreach_neg(cpu_tensors)
            npu_output = torch._foreach_neg(npu_tensors)

            self.assertRtolEqual(cpu_output, npu_output)
            
    
    def test_foreach_neg_out_int32_shpae_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list :
            cpu_tensors, npu_tensors = self.create_tensors(tensor_num, "int32")
            cpu_output = torch._foreach_neg(cpu_tensors)
            npu_output = torch._foreach_neg(npu_tensors)

            self.assertRtolEqual(cpu_output, npu_output)

    
    def test_foreach_neg_inplace_float32_shpae_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list :
            cpu_tensors, npu_tensors = self.create_tensors(tensor_num, "float32")
            torch._foreach_neg_(cpu_tensors)
            torch._foreach_neg_(npu_tensors)

            self.assertRtolEqual(cpu_tensors, npu_tensors)
    
    
    def test_foreach_neg_inplace_float16_shpae_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list :
            cpu_tensors, npu_tensors = self.create_tensors(tensor_num, "float16")
            torch._foreach_neg_(cpu_tensors)
            torch._foreach_neg_(npu_tensors)

            self.assertRtolEqual(cpu_tensors, npu_tensors)

    @SupportedDevices(['Ascend910B'])
    def test_foreach_neg_inplace_bfloat16_shpae_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list :
            cpu_tensors, npu_tensors = self.create_tensors(tensor_num, "bfloat16")
            torch._foreach_neg_(cpu_tensors)
            torch._foreach_neg_(npu_tensors)

            self.assertRtolEqual(cpu_tensors, npu_tensors)
    
    
    def test_foreach_neg_inplace_int32_shpae_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list :
            cpu_tensors, npu_tensors = self.create_tensors(tensor_num, "int32")
            torch._foreach_neg_(cpu_tensors)
            torch._foreach_neg_(npu_tensors)

            self.assertRtolEqual(cpu_tensors, npu_tensors)


if __name__ == "__main__":
    run_tests()

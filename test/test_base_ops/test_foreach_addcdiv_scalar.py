import unittest
import random
import torch
import torch_npu
import hypothesis
import numpy as np
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestForeachAddcdivScalar(TestCase):

    torch_dtypes = {
        "float16" : torch.float16,
        "float32" : torch.float32,
        "bfloat16" : torch.bfloat16
    }

    def create_tensors(self, dtype, shapes):
        cpu_tensors = []
        npu_tensors = []
        for shape in shapes:
            t = torch.randn((shape[0], shape[1]), dtype=self.torch_dtypes.get(dtype))
            t[t == 0] = 2.3
            cpu_tensors.append(t)
            npu_tensors.append(t.npu())
        return tuple(cpu_tensors), tuple(npu_tensors)
    
    def create_input_tensors(self, tensor_num, dtype):
        input_nums = 3
        cpu_inputs = []
        npu_inputs = []
        shapes = []
        for i in range(tensor_num):
            m = random.randint(1, 100)
            n = random.randint(1, 100)
            shapes.append([m, n])
        for i in range(input_nums) :
            cpu_tensors, npu_tensors = self.create_tensors(dtype, shapes)
            cpu_inputs.append(cpu_tensors)
            npu_inputs.append(npu_tensors)
        return cpu_inputs, npu_inputs
    
    
    def test_foreach_addcdiv_scalar_out_float32_shpae_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list :
            cpu_tensors, npu_tensors = self.create_input_tensors(tensor_num, "float32")
            cpu_output = torch._foreach_addcdiv(cpu_tensors[0], cpu_tensors[1], cpu_tensors[2])
            npu_output = torch._foreach_addcdiv(npu_tensors[0], npu_tensors[1], npu_tensors[2])
            
            self.assertRtolEqual(cpu_output, npu_output)
    
    
    def test_foreach_addcdiv_scalar_out_float16_shpae_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list :
            cpu_tensors, npu_tensors = self.create_input_tensors(tensor_num, "float16")
            cpu_tensors_1 = [cpu_tensor.numpy() for cpu_tensor in cpu_tensors[0]]
            cpu_tensors_2 = [cpu_tensor.numpy() for cpu_tensor in cpu_tensors[1]]
            cpu_tensors_3 = [cpu_tensor.numpy() for cpu_tensor in cpu_tensors[2]]
            cpu_output = [torch.from_numpy(cpu_tensors_1[i] + cpu_tensors_2[i] / cpu_tensors_3[i]) for i in range(len(cpu_tensors_1))]

            npu_output = torch._foreach_addcdiv(npu_tensors[0], npu_tensors[1], npu_tensors[2])

            self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(['Ascend910B'])
    def test_foreach_addcdiv_scalar_out_bfloat16_shpae_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list :
            cpu_tensors, npu_tensors = self.create_input_tensors(tensor_num, "bfloat16")
            cpu_output = torch._foreach_addcdiv(cpu_tensors[0], cpu_tensors[1], cpu_tensors[2])
            npu_output = torch._foreach_addcdiv(npu_tensors[0], npu_tensors[1], npu_tensors[2])

            self.assertRtolEqual(cpu_output, npu_output)

    def test_foreach_addcdiv_scalar_inplace_float32_shpae_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list :
            cpu_tensors, npu_tensors = self.create_input_tensors(tensor_num, "float32")
            torch._foreach_addcdiv_(cpu_tensors[0], cpu_tensors[1], cpu_tensors[2])
            torch._foreach_addcdiv_(npu_tensors[0], npu_tensors[1], npu_tensors[2])

            self.assertRtolEqual(cpu_tensors[0], npu_tensors[0])
    
    def test_foreach_addcdiv_scalar_inplace_float16_shpae_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list :
            cpu_tensors, npu_tensors = self.create_input_tensors(tensor_num, "float16")
            cpu_tensors_1 = [cpu_tensor.numpy() for cpu_tensor in cpu_tensors[0]]
            cpu_tensors_2 = [cpu_tensor.numpy() for cpu_tensor in cpu_tensors[1]]
            cpu_tensors_3 = [cpu_tensor.numpy() for cpu_tensor in cpu_tensors[2]]
            cpu_output = [torch.from_numpy(cpu_tensors_1[i] + cpu_tensors_2[i] / cpu_tensors_3[i]) for i in range(len(cpu_tensors_1))]

            torch._foreach_addcdiv_(npu_tensors[0], npu_tensors[1], npu_tensors[2])

            self.assertRtolEqual(cpu_output, npu_tensors[0])
            
    @SupportedDevices(['Ascend910B'])
    def test_foreach_addcdiv_scalar_inplace_bfloat16_shpae_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list :
            cpu_tensors, npu_tensors = self.create_input_tensors(tensor_num, "bfloat16")
            torch._foreach_addcdiv_(cpu_tensors[0], cpu_tensors[1], cpu_tensors[2])
            torch._foreach_addcdiv_(npu_tensors[0], npu_tensors[1], npu_tensors[2])

            self.assertRtolEqual(cpu_tensors[0], npu_tensors[0])


if __name__ == "__main__":
    run_tests()

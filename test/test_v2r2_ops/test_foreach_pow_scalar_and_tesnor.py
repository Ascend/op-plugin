import unittest
import random
import torch
import torch_npu
import hypothesis
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestForeachPowScalarAndTensor(TestCase):

    torch_dtypes = {
        "float16" : torch.float16,
        "float32" : torch.float32,
        "int32" : torch.int32,
        "bfloat16" : torch.bfloat16
    }
    
    def create_tensors(self, dtype, shapes):
        cpu_tensors = []
        npu_tensors = []
        if dtype == "int32":
            for shape in shapes:
                t = torch.randint(low=-20, high=50, size=(shape[0], shape[1]), dtype=self.torch_dtypes.get(dtype))
                cpu_tensors.append(t)
                npu_tensors.append(t.npu())
        else:
            for shape in shapes:
                t = torch.rand((shape[0], shape[1]), dtype=self.torch_dtypes.get(dtype)) * 10 - 5
                cpu_tensors.append(t)
                npu_tensors.append(t.npu())
        return tuple(cpu_tensors), tuple(npu_tensors)
    
    def create_input_tensors(self, tensor_num, dtype):
        input_nums = 1
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

    @SupportedDevices(['Ascend910B'])
    def test_foreach_pow_scalar_and_tensor_out_float32_shpae_tensor_num(self):
        tensor_num_list = [15, 51]
        for tensor_num in tensor_num_list :
            cpu_tensors, npu_tensors = self.create_input_tensors(tensor_num, "float32")
            scalar = 5.0
            cpu_output = torch._foreach_pow(scalar, cpu_tensors[0])
            npu_output = torch._foreach_pow(scalar, npu_tensors[0])

            self.assertRtolEqual(cpu_output, npu_output)
    
    @SupportedDevices(['Ascend910B'])
    def test_foreach_pow_scalar_and_tensor_out_float16_shpae_tensor_num(self):
        tensor_num_list = [12, 52]
        for tensor_num in tensor_num_list :
            cpu_tensors, npu_tensors = self.create_input_tensors(tensor_num, "float16")
            scalar = 5.0
            cpu_output = torch._foreach_pow(scalar, cpu_tensors[0])
            npu_output = torch._foreach_pow(scalar, npu_tensors[0])

            self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(['Ascend910B'])
    def test_foreach_pow_scalar_and_tensor_out_bfloat16_shpae_tensor_num(self):
        tensor_num_list = [12, 52]
        for tensor_num in tensor_num_list :
            cpu_tensors, npu_tensors = self.create_input_tensors(tensor_num, "bfloat16")
            scalar = 5.0
            cpu_output = torch._foreach_pow(scalar, cpu_tensors[0])
            npu_output = torch._foreach_pow(scalar, npu_tensors[0])

            self.assertRtolEqual(cpu_output, npu_output)
            
    @SupportedDevices(['Ascend910B'])
    def test_foreach_pow_scalar_and_tensor_out_int32_shpae_tensor_num(self):
        tensor_num_list = [12, 52]
        for tensor_num in tensor_num_list :
            cpu_tensors, npu_tensors = self.create_input_tensors(tensor_num, "int32")
            scalar = 5
            cpu_output = torch._foreach_pow(scalar, cpu_tensors[0])
            npu_output = torch._foreach_pow(scalar, npu_tensors[0])

            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()

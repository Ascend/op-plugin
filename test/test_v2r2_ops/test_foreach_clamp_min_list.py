import unittest
import random
import torch
import torch_npu
import hypothesis
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestForeachClampMinList(TestCase):

    torch_dtypes = {
        "float16" : torch.float16,
        "float32" : torch.float32,
        "bfloat16" : torch.bfloat16,
        "int32" : torch.int32
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
    
    def create_tensors(self, dtype, shapes):
        cpu_tensors = []
        npu_tensors = []
        if dtype == "int32":
            for shape in shapes:
                t = torch.randint(low=0, high=100, size=(shape[0], shape[1]), dtype=self.torch_dtypes.get(dtype))
                cpu_tensors.append(t)
                npu_tensors.append(t)
        else:
            for shape in shapes:
                t = torch.randn((shape[0], shape[1]), dtype=self.torch_dtypes.get(dtype))
                cpu_tensors.append(t)
                npu_tensors.append(t)
        return tuple(cpu_tensors), tuple(npu_tensors)
    
    def create_input_tensors(self, tensor_num, dtype):
        input_nums = 2
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
    def test_foreach_clamp_min_list_out_float32_shpae_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list :
            cpu_tensors, npu_tensors = self.create_input_tensors(tensor_num, "float32")
            
            cpu_output = torch._foreach_clamp_min(cpu_tensors[0], cpu_tensors[1])
            npu_output = torch._foreach_clamp_min(npu_tensors[0], npu_tensors[1])

            self.assertRtolEqual(cpu_output, npu_output)
    
    @SupportedDevices(['Ascend910B'])
    def test_foreach_clamp_min_list_out_float16_shpae_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list :
            cpu_tensors, npu_tensors = self.create_input_tensors(tensor_num, "float16")
            
            cpu_output = torch._foreach_clamp_min(cpu_tensors[0], cpu_tensors[1])
            npu_output = torch._foreach_clamp_min(npu_tensors[0], npu_tensors[1])

            self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(['Ascend910B'])
    def test_foreach_clamp_min_list_out_bfloat16_shpae_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list :
            cpu_tensors, npu_tensors = self.create_input_tensors(tensor_num, "bfloat16")
            
            cpu_output = torch._foreach_clamp_min(cpu_tensors[0], cpu_tensors[1])
            npu_output = torch._foreach_clamp_min(npu_tensors[0], npu_tensors[1])

            self.assert_equal(cpu_output, npu_output)
            
    @SupportedDevices(['Ascend910B'])
    def test_foreach_clamp_min_list_out_int32_shpae_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list :
            cpu_tensors, npu_tensors = self.create_input_tensors(tensor_num, "int32")
            
            cpu_output = torch._foreach_clamp_min(cpu_tensors[0], cpu_tensors[1])
            npu_output = torch._foreach_clamp_min(npu_tensors[0], npu_tensors[1])

            self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(['Ascend910B'])
    def test_foreach_clamp_min_list_inplace_float32_shpae_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list :
            cpu_tensors, npu_tensors = self.create_input_tensors(tensor_num, "float32")
            
            torch._foreach_clamp_min_(cpu_tensors[0], cpu_tensors[1])
            torch._foreach_clamp_min_(npu_tensors[0], npu_tensors[1])

            self.assertRtolEqual(cpu_tensors[0], npu_tensors[0])
    
    @SupportedDevices(['Ascend910B'])
    def test_foreach_clamp_min_list_inplace_float16_shpae_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list :
            cpu_tensors, npu_tensors = self.create_input_tensors(tensor_num, "float16")
            
            torch._foreach_clamp_min_(cpu_tensors[0], cpu_tensors[1])
            torch._foreach_clamp_min_(npu_tensors[0], npu_tensors[1])

            self.assertRtolEqual(cpu_tensors[0], npu_tensors[0])
            
    @SupportedDevices(['Ascend910B'])
    def test_foreach_clamp_min_list_inplace_bfloat16_shpae_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list :
            cpu_tensors, npu_tensors = self.create_input_tensors(tensor_num, "bfloat16")
            
            torch._foreach_clamp_min_(cpu_tensors[0], cpu_tensors[1])
            torch._foreach_clamp_min_(npu_tensors[0], npu_tensors[1])

            self.assert_equal(cpu_tensors[0], npu_tensors[0])

    @SupportedDevices(['Ascend910B'])
    def test_foreach_clamp_min_list_inplace_int32_shpae_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list :
            cpu_tensors, npu_tensors = self.create_input_tensors(tensor_num, "int32")
            
            torch._foreach_clamp_min_(cpu_tensors[0], cpu_tensors[1])
            torch._foreach_clamp_min_(npu_tensors[0], npu_tensors[1])

            self.assertRtolEqual(cpu_tensors[0], npu_tensors[0])

if __name__ == "__main__":
    run_tests()

import unittest
import random
import torch
import torch_npu
import hypothesis
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestForeachCopy(TestCase):

    torch_dtypes = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
        "int64": torch.int64,
        "double": torch.double,
        "bool": torch.bool
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
            if (dtype == "int32" or dtype == "int64"):
                t = torch.randint(low=-100, high=100, size=(1, 1), dtype=self.torch_dtypes.get(dtype))
            elif (dtype == "double"):
                t = torch.rand(size=(1, 1), dtype=self.torch_dtypes.get(dtype))
                t = t * 200 - 100
            elif (dtype == "bool"):
                t = torch.randint(low=0, high=2, size=(1, 1), dtype=self.torch_dtypes.get(dtype))
                t = t.bool()
            else:
                t = torch.randn((1, 1), dtype=self.torch_dtypes.get(dtype))
            cpu_tensors.append(t)
            npu_tensors.append(t.npu())
        return tuple(cpu_tensors), tuple(npu_tensors)

    @SupportedDevices(['Ascend910B'])
    def test_foreach_copy_float32_shpae_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list:
            cpu_tensors_x, npu_tensors_x = self.create_tensors(tensor_num, "float32")
            cpu_tensors_y, npu_tensors_y = self.create_tensors(tensor_num, "float32")
            torch._foreach_copy_(cpu_tensors_y, cpu_tensors_x)
            torch._foreach_copy_(npu_tensors_y, npu_tensors_x)

            self.assertRtolEqual(cpu_tensors_y, npu_tensors_y)
    
    @SupportedDevices(['Ascend910B'])
    def test_foreach_copy_float16_shpae_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list:
            cpu_tensors_x, npu_tensors_x = self.create_tensors(tensor_num, "float16")
            cpu_tensors_y, npu_tensors_y = self.create_tensors(tensor_num, "float16")
            torch._foreach_copy_(cpu_tensors_y, cpu_tensors_x)
            torch._foreach_copy_(npu_tensors_y, npu_tensors_x)
            
            self.assertRtolEqual(cpu_tensors_y, npu_tensors_y)

    @SupportedDevices(['Ascend910B'])
    def test_foreach_copy_bfloat16_shpae_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list:
            cpu_tensors_x, npu_tensors_x = self.create_tensors(tensor_num, "bfloat16")
            cpu_tensors_y, npu_tensors_y = self.create_tensors(tensor_num, "bfloat16")
            torch._foreach_copy_(cpu_tensors_y, cpu_tensors_x)
            torch._foreach_copy_(npu_tensors_y, npu_tensors_x)
            
            self.assert_equal(cpu_tensors_y, npu_tensors_y)
    
    @SupportedDevices(['Ascend910B'])
    def test_foreach_copy_int32_shpae_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list:
            cpu_tensors_x, npu_tensors_x = self.create_tensors(tensor_num, "int32")
            cpu_tensors_y, npu_tensors_y = self.create_tensors(tensor_num, "int32")
            torch._foreach_copy_(cpu_tensors_y, cpu_tensors_x)
            torch._foreach_copy_(npu_tensors_y, npu_tensors_x)
            
            self.assertRtolEqual(cpu_tensors_y, npu_tensors_y)

    @SupportedDevices(['Ascend910B'])
    def test_foreach_copy_int64_shape_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list:
            cpu_tensors_x, npu_tensors_x = self.create_tensors(tensor_num, "int64")
            cpu_tensors_y, npu_tensors_y = self.create_tensors(tensor_num, "int64")
            torch._foreach_copy_(cpu_tensors_y, cpu_tensors_x)
            torch._foreach_copy_(npu_tensors_y, npu_tensors_x)
            
            self.assertRtolEqual(cpu_tensors_y, npu_tensors_y)
    
    @SupportedDevices(['Ascend910B'])
    def test_foreach_copy_double_shape_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list:
            cpu_tensors_x, npu_tensors_x = self.create_tensors(tensor_num, "double")
            cpu_tensors_y, npu_tensors_y = self.create_tensors(tensor_num, "double")
            torch._foreach_copy_(cpu_tensors_y, cpu_tensors_x)
            torch._foreach_copy_(npu_tensors_y, npu_tensors_x)
            
            self.assertRtolEqual(cpu_tensors_y, npu_tensors_y)
    
    @SupportedDevices(['Ascend910B'])
    def test_foreach_copy_bool_shape_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list:
            cpu_tensors_x, npu_tensors_x = self.create_tensors(tensor_num, "bool")
            cpu_tensors_y, npu_tensors_y = self.create_tensors(tensor_num, "bool")
            torch._foreach_copy_(cpu_tensors_y, cpu_tensors_x)
            torch._foreach_copy_(npu_tensors_y, npu_tensors_x)
            
            self.assertRtolEqual(cpu_tensors_y, npu_tensors_y)

    @SupportedDevices(['Ascend910B'])
    def test_foreach_copy_int32_different_xpu(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list:
            cpu_tensors_x, npu_tensors_x = self.create_tensors(tensor_num, "int32")
            cpu_tensors_y, npu_tensors_y = self.create_tensors(tensor_num, "int32")
            torch._foreach_copy_(npu_tensors_y, cpu_tensors_x)
            self.assertRtolEqual(npu_tensors_y, cpu_tensors_x)

if __name__ == "__main__":
    run_tests()

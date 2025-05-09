import unittest
import random
import torch
import torch_npu
import hypothesis
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestForeachSign(TestCase):

    torch_dtypes = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
        "int8": torch.int8,
        "int64": torch.int64,
    }

    def assert_equal(self, cpu_outs, npu_outs):
        for cpu_out, npu_out in zip(cpu_outs, npu_outs):
            if cpu_out.shape != npu_out.shape:
                self.fail("shape error")
            if cpu_out.dtype != npu_out.dtype:
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
            if dtype == "int32" or dtype == "int8" or dtype == "int64":
                t = torch.randint(low=-100, high=100, size=(m, n), dtype=self.torch_dtypes.get(dtype))
            else:
                t = torch.randn((m, n), dtype=self.torch_dtypes.get(dtype))
            cpu_tensors.append(t)
            npu_tensors.append(t.npu())
        return tuple(cpu_tensors), tuple(npu_tensors)

    @SupportedDevices(['Ascend910B'])
    def test_foreach_sign_out_float32_shape_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list:
            cpu_tensors, npu_tensors = self.create_tensors(tensor_num, "float32")
            cpu_output = torch._foreach_sign(cpu_tensors)
            npu_output = torch._foreach_sign(npu_tensors)

            self.assertRtolEqual(cpu_output, npu_output)
    
    @SupportedDevices(['Ascend910B'])
    def test_foreach_sign_out_float16_shape_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list:
            cpu_tensors, npu_tensors = self.create_tensors(tensor_num, "float16")
            cpu_output = torch._foreach_sign(cpu_tensors)
            npu_output = torch._foreach_sign(npu_tensors)

            self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(['Ascend910B'])
    def test_foreach_sign_out_bfloat16_shape_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list:
            cpu_tensors, npu_tensors = self.create_tensors(tensor_num, "bfloat16")
            cpu_output = torch._foreach_sign(cpu_tensors)
            npu_output = torch._foreach_sign(npu_tensors)

            self.assert_equal(cpu_output, npu_output)
            
    @SupportedDevices(['Ascend910B'])
    def test_foreach_sign_out_int32_shape_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list:
            cpu_tensors, npu_tensors = self.create_tensors(tensor_num, "int32")
            cpu_output = torch._foreach_sign(cpu_tensors)
            npu_output = torch._foreach_sign(npu_tensors)

            self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(['Ascend910B'])
    def test_foreach_sign_out_int32_shape_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list:
            cpu_tensors, npu_tensors = self.create_tensors(tensor_num, "int8")
            cpu_output = torch._foreach_sign(cpu_tensors)
            npu_output = torch._foreach_sign(npu_tensors)

            self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(['Ascend910B'])
    def test_foreach_sign_out_int32_shape_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list:
            cpu_tensors, npu_tensors = self.create_tensors(tensor_num, "int64")
            cpu_output = torch._foreach_sign(cpu_tensors)
            npu_output = torch._foreach_sign(npu_tensors)

            self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(['Ascend910B'])
    def test_foreach_sign_inplace_float32_shpae_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list:
            cpu_tensors, npu_tensors = self.create_tensors(tensor_num, "float32")
            torch._foreach_sign_(cpu_tensors)
            torch._foreach_sign_(npu_tensors)

            self.assertRtolEqual(cpu_tensors, npu_tensors)
    
    @SupportedDevices(['Ascend910B'])
    def test_foreach_sign_inplace_float16_shape_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list:
            cpu_tensors, npu_tensors = self.create_tensors(tensor_num, "float16")
            torch._foreach_sign_(cpu_tensors)
            torch._foreach_sign_(npu_tensors)

            self.assertRtolEqual(cpu_tensors, npu_tensors)

    @SupportedDevices(['Ascend910B'])
    def test_foreach_sign_inplace_bfloat16_shape_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list:
            cpu_tensors, npu_tensors = self.create_tensors(tensor_num, "bfloat16")
            torch._foreach_sign_(cpu_tensors)
            torch._foreach_sign_(npu_tensors)

            self.assert_equal(cpu_tensors, npu_tensors)
    
    @SupportedDevices(['Ascend910B'])
    def test_foreach_sign_inplace_int32_shape_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list:
            cpu_tensors, npu_tensors = self.create_tensors(tensor_num, "int32")
            torch._foreach_sign_(cpu_tensors)
            torch._foreach_sign_(npu_tensors)

            self.assertRtolEqual(cpu_tensors, npu_tensors)

    @SupportedDevices(['Ascend910B'])
    def test_foreach_sign_inplace_int32_shape_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list:
            cpu_tensors, npu_tensors = self.create_tensors(tensor_num, "int8")
            torch._foreach_sign_(cpu_tensors)
            torch._foreach_sign_(npu_tensors)

            self.assertRtolEqual(cpu_tensors, npu_tensors)

    @SupportedDevices(['Ascend910B'])
    def test_foreach_sign_inplace_int32_shape_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list:
            cpu_tensors, npu_tensors = self.create_tensors(tensor_num, "int64")
            torch._foreach_sign_(cpu_tensors)
            torch._foreach_sign_(npu_tensors)

            self.assertRtolEqual(cpu_tensors, npu_tensors)

    @SupportedDevices(['Ascend910B'])
    def test_foreach_sign_inplace_float64_shape_tensor_num(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list:
            cpu_tensors, npu_tensors = self.create_tensors(tensor_num, "float64")
            torch._foreach_sign_(cpu_tensors)
            torch._foreach_sign_(npu_tensors)

            self.assertRtolEqual(cpu_tensors, npu_tensors)

    @SupportedDevices(['Ascend910B'])
    def test_foreach_sign_inplace_int32_different_xpu(self):
        tensor_num_list = [20, 50]
        for tensor_num in tensor_num_list:
            cpu_tensors, npu_tensors = self.create_tensors(tensor_num, "int32")
            torch._foreach_sign_([cpu_tensors[0], npu_tensors[0]])

            self.assertRtolEqual(cpu_tensors[0], npu_tensors[0])


if __name__ == "__main__":
    run_tests()

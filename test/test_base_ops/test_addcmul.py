import copy
import torch
import numpy as np

import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestAddCMul(TestCase):
    def generate_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input2 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input3 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)
        npu_input3 = torch.from_numpy(input3)

        return npu_input1, npu_input2, npu_input3

    def generate_single_data(self, min1, max1, shape, dtype):
        inputs = np.random.uniform(min1, max1, shape).astype(dtype)
        npu_input = torch.from_numpy(inputs)
        return npu_input

    def generate_output_data(self, min_d, max_d, shape, dtype):
        output_y = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_output_y = torch.from_numpy(output_y)
        return npu_output_y

    def generate_scalar(self, min1, max1):
        scalar = np.random.uniform(min1, max1)
        return scalar

    def generate_int_scalar(self, min1, max1):
        scalar = np.random.randint(min1, max1)
        return scalar

    def cpu_op_exec(self, input1, input2, input3, scalar):
        output = torch.addcmul(input1, input2, input3, value=scalar)
        output = output.numpy()
        return output

    def cpu_op_exec_out(self, input1, input2, input3, scalar, output_y):
        output = output_y
        torch.addcmul(input1, input2, input3, value=scalar, out=output_y)
        output = output.numpy()
        return output

    def cpu_op_inp_contiguous_exec(self, input1, input2, input3, scalar):
        input1.addcmul_(input2, input3, value=scalar)
        output = input1.numpy()
        return output

    def npu_op_exec(self, input1, input2, input3, scalar):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        input3 = input3.to("npu")
        output = torch.addcmul(input1, input2, input3, value=scalar)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, input2, input3, scalar, output_y):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        input3 = input3.to("npu")
        output = output_y.to("npu")
        torch.addcmul(input1, input2, input3, value=scalar, out=output)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_inp_contiguous_exec(self, input1, input2, input3, scalar):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        input3 = input3.to("npu")
        input1.addcmul_(input2, input3, value=scalar)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def test_addcmul_3_3_float32(self):
        input1, input2, input3 = self.generate_data(0, 100, (3, 3), np.float32)
        cpu_output = self.cpu_op_exec(input1, input2, input3, 0.5)
        npu_output = self.npu_op_exec(input1, input2, input3, 0.5)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_addcmul_10_10_float32(self):
        input1, input2, input3 = self.generate_data(0, 100, (10, 10), np.float32)
        cpu_output = self.cpu_op_exec(input1, input2, input3, 0.5)
        npu_output = self.npu_op_exec(input1, input2, input3, 0.5)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_addcmul_3_3_float16(self):
        input1, input2, input3 = self.generate_data(0, 100, (3, 3), np.float16)
        input1_cpu = input1.float()
        input2_cpu = input2.float()
        input3_cpu = input3.float()
        cpu_output = self.cpu_op_exec(input1_cpu, input2_cpu, input3_cpu, 0.5).astype(
            np.float16
        )
        npu_output = self.npu_op_exec(input1, input2, input3, 0.5)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_addcmul_10_10_float16(self):
        input1, input2, input3 = self.generate_data(0, 100, (10, 10), np.float16)
        input1_cpu = input1.float()
        input2_cpu = input2.float()
        input3_cpu = input3.float()
        cpu_output = self.cpu_op_exec(input1_cpu, input2_cpu, input3_cpu, 0.5).astype(
            np.float16
        )
        npu_output = self.npu_op_exec(input1, input2, input3, 0.5)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_addcmul_10_23_float32(self):
        input1, input2, input3 = self.generate_data(0, 100, (10, 23), np.float32)
        cpu_output = self.cpu_op_exec(input1, input2, input3, 0.5)
        npu_output = self.npu_op_exec(input1, input2, input3, 0.5)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_tensor_addcmul_3_3_float32(self):
        input1, input2, input3 = self.generate_data(0, 100, (3, 3), np.float32)
        input1_npu = input1.npu()
        input2_npu = input2.npu()
        input3_npu = input3.npu()
        input1.addcmul_(input2, input3, value=0.5)
        input1_npu.addcmul_(input2_npu, input3_npu, value=0.5)
        self.assertRtolEqual(input1, input1_npu)

    def test_tensor_addcmul_10_10_float32(self):
        input1, input2, input3 = self.generate_data(0, 100, (10, 10), np.float32)
        input1_npu = input1.npu()
        input2_npu = input2.npu()
        input3_npu = input3.npu()
        input1.addcmul_(input2, input3, value=0.5)
        input1_npu.addcmul_(input2_npu, input3_npu, value=0.5)
        self.assertRtolEqual(input1, input1_npu)

    def test_tensor_addcmul_3_3_float16(self):
        input1, input2, input3 = self.generate_data(0, 100, (3, 3), np.float16)
        input1_npu = input1.npu()
        input2_npu = input2.npu()
        input3_npu = input3.npu()
        input1.addcmul_(input2, input3, value=0.5)
        input1_npu.addcmul_(input2_npu, input3_npu, value=0.5)
        self.assertRtolEqual(input1, input1_npu)

    def test_tensor_addcmul_10_10_float16(self):
        input1, input2, input3 = self.generate_data(0, 100, (10, 10), np.float16)
        input1_npu = input1.npu()
        input2_npu = input2.npu()
        input3_npu = input3.npu()
        input1.addcmul_(input2, input3, value=0.5)
        input1_npu.addcmul_(input2_npu, input3_npu, value=0.5)
        self.assertRtolEqual(input1, input1_npu)

    def test_tensor_addcmul_float32(self):
        input1 = torch.randn(3, 5, 5).npu()
        input2 = torch.randn(5, 5).npu()
        input3 = torch.ones(3, 5).npu()
        with self.assertRaises(RuntimeError) as cm:
            npu_output = self.npu_op_exec(input1, input2, input3, 0.5)
        exception = cm.exception
        self.assertTrue("The size of tensor a (5) must match the size of tensor b (3) at non-singleton dimension 1" in str(exception))

    @SupportedDevices(['Ascend910B'])
    def test_addcmul_high_type_cast(self):
        cpu_input1, cpu_input2, cpu_input3 = self.generate_data(
            1, 100, (5, 3), np.float32
        )
        cpu_input1 = cpu_input1.to(torch.int8)
        cpu_input3 = cpu_input3.to(torch.int8)
        scalar = self.generate_scalar(1, 10)

        cpu_output = self.cpu_op_exec(
            cpu_input1.float(), cpu_input2.float(), cpu_input3.float(), scalar
        )
        npu_output = self.npu_op_exec(cpu_input1, cpu_input2, cpu_input3, scalar)
        self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(['Ascend910B'])
    def test_addcmul_high_type_cast_out(self):
        npu_input1, npu_input2, npu_input3 = self.generate_data(
            1, 100, (5, 3), np.float32
        )
        npu_input1 = npu_input1.to(torch.int8)
        npu_input3 = npu_input3.to(torch.int8)
        scalar = self.generate_scalar(1, 10)
        npu_input4 = self.generate_single_data(1, 100, (5, 3), np.float32)

        cpu_output = self.cpu_op_exec_out(
            npu_input1, npu_input2, npu_input3, scalar, npu_input4
        )
        npu_output = self.npu_op_exec_out(
            npu_input1, npu_input2, npu_input3, scalar, npu_input4
        )
        self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(['Ascend910B'])
    def test_addcmul_high_type_cast_inp(self):
        npu_input1, npu_input2, npu_input3 = self.generate_data(
            1, 100, (5, 3), np.float32
        )
        npu_input3 = npu_input3.to(torch.int8)
        cpu_input1 = copy.deepcopy(npu_input1)
        cpu_input2 = copy.deepcopy(npu_input2)
        cpu_input3 = copy.deepcopy(npu_input3)
        scalar = self.generate_int_scalar(1, 10)

        cpu_output = self.cpu_op_inp_contiguous_exec(
            cpu_input1, cpu_input2, cpu_input3, scalar
        )
        npu_output = self.npu_op_inp_contiguous_exec(
            npu_input1, npu_input2, npu_input3, scalar
        )
        self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(['Ascend910B'])
    def test_addcmul_high_type_cast_out_expect_error(self):
        npu_input1, npu_input2, npu_input3 = self.generate_data(
            1, 100, (5, 3), np.float32
        )
        npu_input1 = npu_input1.to(torch.int8)
        npu_input3 = npu_input3.to(torch.int8)
        scalar = self.generate_scalar(1, 10)
        npu_input4 = self.generate_single_data(1, 100, (5, 3), np.float32)
        npu_input4 = npu_input4.to(torch.int8)

        with self.assertRaises(RuntimeError) as cpu_err:
            self.cpu_op_exec_out(
                npu_input1, npu_input2, npu_input3, scalar, npu_input4
            )
        self.assertTrue("result type Float can't be cast to the desired output type Char" in str(cpu_err.exception))

        with self.assertRaises(RuntimeError) as npu_err:
            self.npu_op_exec_out(
                npu_input1, npu_input2, npu_input3, scalar, npu_input4
            )
        self.assertTrue("result type Float can't be cast to the desired output type Char" in str(npu_err.exception))

    @SupportedDevices(['Ascend910B'])
    def test_addcmul_high_type_cast_inp_expect_error(self):
        npu_input1, npu_input2, npu_input3 = self.generate_data(
            1, 100, (5, 3), np.float32
        )
        npu_input1 = npu_input1.to(torch.int8)
        npu_input3 = npu_input3.to(torch.int8)
        cpu_input1 = copy.deepcopy(npu_input1)
        cpu_input2 = copy.deepcopy(npu_input2)
        cpu_input3 = copy.deepcopy(npu_input3)
        scalar = self.generate_int_scalar(1, 10)

        with self.assertRaises(RuntimeError) as cpu_err:
            self.cpu_op_inp_contiguous_exec(
                cpu_input1, cpu_input2, cpu_input3, scalar
            )
        self.assertTrue("result type Float can't be cast to the desired output type Char" in str(cpu_err.exception))

        with self.assertRaises(RuntimeError) as npu_err:
            self.npu_op_inp_contiguous_exec(
                npu_input1, npu_input2, npu_input3, scalar
            )
        self.assertTrue("result type Float can't be cast to the desired output type Char" in str(npu_err.exception))

    def test_addcmul_out_resize(self):
        npu_input1, npu_input2, npu_input3 = self.generate_data(
            1, 100, (3, 3), np.float32
        )
        scalar = self.generate_scalar(1, 10)
        # the shape is different from input1
        npu_input4 = self.generate_single_data(1, 100, (4, 4), np.float32)
        cpu_output = self.cpu_op_exec_out(
            npu_input1, npu_input2, npu_input3, scalar, npu_input4
        )
        npu_output = self.npu_op_exec_out(
            npu_input1, npu_input2, npu_input3, scalar, npu_input4
        )
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()

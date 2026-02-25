import copy
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestGelu(TestCase):
    def generate_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)

        # modify from numpy.ndarray to torch.tensor
        npu_input1 = torch.from_numpy(input1)
        return npu_input1

    def cpu_op_exec(self, input1):
        torch.nn.functional.gelu(input1,out=input1)
        input1 = input1.numpy()
        return input1

    def npu_op_exec(self, input1):
        input1_npu = input1.to('npu')
        torch.nn.functional.gelu(input1_npu,out = input1_npu)
        input1_npu = input1_npu.to("cpu")
        input1_npu = input1_npu.numpy()
        return input1_npu

    def cpu_op_exec_fp16(self, input1):
        input1 = input1.to(torch.float32)
        torch.nn.functional.gelu(input1,out=input1)
        input1 = input1.numpy()
        input1 = input1.astype(np.float16)
        return input1

    def npu_op_exec_fp16(self, input1):
        input1_npu = input1.to(torch.float32).to('npu')
        torch.nn.functional.gelu(input1_npu,out=input1_npu)
        input1_npu = input1_npu.to("cpu")
        input1_npu = input1_npu.numpy().astype(np.float16)
        return input1_npu

    def test_gelu_float32_1(self):
        input1 = self.generate_data(0, 100, (4, 3), np.float32)
        cpu_input1 = copy.deepcopy(input1)
        cpu_output = self.cpu_op_exec(cpu_input1)
        npu_output = self.npu_op_exec(input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_gelu_float32_2(self):
        input1 = self.generate_data(0, 1000, (4, 3), np.float32)
        cpu_input1 = copy.deepcopy(input1)
        cpu_output = self.cpu_op_exec(cpu_input1)
        npu_output = self.npu_op_exec(input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_gelu_float16_1(self):
        npu_input1 = self.generate_data(0, 100, (5, 3), np.float16)
        cpu_input1 = copy.deepcopy(npu_input1)
        cpu_output = self.cpu_op_exec_fp16(cpu_input1)
        npu_output = self.npu_op_exec_fp16(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_gelu_float16_2(self):
        npu_input1 = self.generate_data(0, 1000, (5, 3), np.float16)
        cpu_input1 = copy.deepcopy(npu_input1)
        cpu_output = self.cpu_op_exec_fp16(cpu_input1)
        npu_output = self.npu_op_exec_fp16(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_gelu_float16_3(self):
        npu_input1 = self.generate_data(0, 1000, (3, 3), np.float16)
        cpu_input1 = copy.deepcopy(npu_input1)
        cpu_output = self.cpu_op_exec_fp16(cpu_input1)
        npu_output = self.npu_op_exec_fp16(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)
    
    def test_gelu_negative_input(self):
        npu_input1 = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
        cpu_input1 = copy.deepcopy(npu_input1)
        cpu_output = self.cpu_op_exec(cpu_input1)
        npu_output = self.npu_op_exec(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output, prec=1e-03, prec16=1e-03)


if __name__ == "__main__":
    np.random.seed(1234)
    run_tests()
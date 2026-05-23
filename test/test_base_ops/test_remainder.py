import copy
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestRemainder(TestCase):
    def cpu_op_exec(self, input1, input2):
        output = torch.remainder(input1, input2)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        output = torch.remainder(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, input2, out):
        output = torch.remainder(input1, input2, out=out)
        output = out.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_inplace_exec(self, input1, input2):
        output = input1.remainder_(input2)
        output = output.numpy()
        return output

    def npu_op_inplace_exec(self, input1, input2):
        output = input1.remainder_(input2)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_scalar(self, input1, input2):
        output = torch.remainder(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def remainder_out_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 0, 100)
            npu_input3 = torch.randn(6).to("npu")
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_input2 = cpu_input2.to(torch.float32)

            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output_out = self.npu_op_exec_out(npu_input1, npu_input2, npu_input3)
            self.assertRtolEqual(cpu_output, npu_output_out)

    def remainder_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 0, 100)
            npu_input3 = copy.deepcopy(cpu_input1).to("npu")
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            npu_output_out = self.npu_op_exec_out(npu_input1, npu_input2, npu_input3)
            cpu_output_inplace = self.cpu_op_inplace_exec(cpu_input1, cpu_input2)
            npu_output_inplace = self.npu_op_inplace_exec(npu_input1, npu_input2)

            cpu_output = cpu_output.astype(npu_output.dtype)
            cpu_output_inplace = cpu_output_inplace.astype(npu_output_inplace.dtype)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output_inplace, npu_output_inplace)
            self.assertRtolEqual(cpu_output, npu_output_out)

    def remainder_scalar_result(self, shape_format):
        for item in shape_format:
            scalar = np.random.uniform(0, 100)
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 2)
            npu_input3 = copy.deepcopy(cpu_input1).to("npu")
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, scalar)
            npu_output_scalar = self.npu_op_exec_scalar(npu_input1, scalar)
            npu_output_out = self.npu_op_exec_out(npu_input1, scalar, npu_input3)

            cpu_output = cpu_output.astype(npu_output_scalar.dtype)
            self.assertRtolEqual(cpu_output, npu_output_scalar)
            self.assertRtolEqual(cpu_output, npu_output_out)

    def test_remainder_shape_format_fp16_1d(self):
        format_list = [0, 3]
        shape_format = [[np.float16, i, [4]] for i in format_list
                        ]
        self.remainder_result(shape_format)

    def test_remainder_shape_format_fp32_1d(self):
        format_list = [0, 3]
        shape_format = [[np.float32, i, [4]] for i in format_list
                        ]
        self.remainder_result(shape_format)

    def test_remainder_shape_format_fp16_2d(self):
        format_list = [0, 3, 29]
        shape_format = [[np.float16, i, [4, 18]] for i in format_list
                        ]
        self.remainder_result(shape_format)

    def test_remainder_shape_format_fp32_2d(self):
        format_list = [0, 3, 29]
        shape_format = [[np.float32, i, [4, 18]] for i in format_list
                        ]
        self.remainder_result(shape_format)

    def test_remainder_shape_format_fp16_3d(self):
        format_list = [0, 3, 29]
        shape_format = [[np.float16, i, [4, 18, 32]] for i in format_list
                        ]
        self.remainder_result(shape_format)

    def test_remainder_shape_format_fp32_3d(self):
        format_list = [0, 3, 29]
        shape_format = [[np.float32, i, [4, 18, 32]] for i in format_list
                        ]
        self.remainder_result(shape_format)

    def test_remainder_shape_format_fp16_4d(self):
        format_list = [0, 3, 29]
        shape_format = [[np.float16, i, [4, 18, 32, 128]] for i in format_list
                        ]
        self.remainder_result(shape_format)

    def test_remainder_shape_format_fp32_4d(self):
        format_list = [0, 3, 29]
        shape_format = [[np.float32, i, [4, 18, 32, 128]] for i in format_list
                        ]
        self.remainder_result(shape_format)

    # scalar----------------------------------------------------------
    def test_remainder_scalar_shape_format_fp16_1d(self):
        format_list = [0, 3]
        shape_format = [[np.float16, i, [4]] for i in format_list
                        ]
        self.remainder_scalar_result(shape_format)

    def test_remainder_scalar_shape_format_fp32_1d(self):
        format_list = [0, 3]
        shape_format = [[np.float32, i, [4]] for i in format_list
                        ]
        self.remainder_scalar_result(shape_format)

    def test_remainder_scalar_shape_format_fp16_2d(self):
        format_list = [0, 3, 29]
        shape_format = [[np.float16, i, [4, 18]] for i in format_list
                        ]
        self.remainder_scalar_result(shape_format)

    def test_remainder_scalar_shape_format_fp32_2d(self):
        format_list = [0, 3, 29]
        shape_format = [[np.float32, i, [4, 18]] for i in format_list
                        ]
        self.remainder_scalar_result(shape_format)

    def test_remainder_scalar_shape_format_fp16_3d(self):
        format_list = [0, 3, 29]
        shape_format = [[np.float16, i, [4, 18, 32]] for i in format_list
                        ]
        self.remainder_scalar_result(shape_format)

    def test_remainder_scalar_shape_format_fp32_3d(self):
        format_list = [0, 3, 29]
        shape_format = [[np.float32, i, [4, 18, 32]] for i in format_list
                        ]
        self.remainder_scalar_result(shape_format)

    def test_remainder_scalar_shape_format_fp16_4d(self):
        format_list = [0, 3, 29]
        shape_format = [[np.float16, i, [4, 18, 32, 128]] for i in format_list
                        ]
        self.remainder_scalar_result(shape_format)

    def test_remainder_scalar_shape_format_fp32_4d(self):
        format_list = [0, 3, 29]
        shape_format = [[np.float32, i, [4, 18, 32, 128]] for i in format_list
                        ]
        self.remainder_scalar_result(shape_format)

    def test_remainder_mix_dtype_1(self):
        npu_input1, npu_input2 = create_common_tensor([np.int32, 0, (2, 3)], 1, 100)
        npu_input3, npu_input4 = create_common_tensor([np.float32, 0, (2, 3)], 1, 100)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input3)
        npu_output = self.npu_op_exec(npu_input1, npu_input3)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_remainder_mix_dtype_2(self):
        npu_input1, npu_input2 = create_common_tensor([np.float32, 0, (2, 3)], 1, 100)
        npu_input3 = torch.tensor(3).int()
        cpu_output = self.cpu_op_exec(npu_input1, npu_input3)
        npu_output = self.npu_op_exec(npu_input1, npu_input3)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_remainder_scalar_shape_format_fp32_out_4d(self):
        format_list = [0]
        shape_format = [[np.float32, i, [4, 18, 32, 128]] for i in format_list
                        ]
        self.remainder_out_result(shape_format)

    def test_remainder_second_arg_0d_cpu_tensor(self):
        cpu_a = torch.tensor([5.0, 7.0, 9.0])
        cpu_b = torch.tensor(3.0)
        npu_a = cpu_a.npu()
        cpu_output = torch.remainder(cpu_a, cpu_b)
        npu_output = torch.remainder(npu_a, cpu_b)
        self.assertRtolEqual(cpu_output, npu_output.cpu())

    def test_remainder_first_arg_0d_cpu_tensor(self):
        cpu_a = torch.tensor(7.0)
        cpu_b = torch.tensor([3.0, 4.0, 5.0])
        npu_b = cpu_b.npu()
        cpu_output = torch.remainder(cpu_a, cpu_b)
        npu_output = torch.remainder(cpu_a, npu_b)
        self.assertRtolEqual(cpu_output, npu_output.cpu())

    def test_remainder_inplace_second_arg_0d_cpu_tensor(self):
        cpu_a = torch.tensor([5.0, 7.0, 9.0])
        cpu_b = torch.tensor(3.0)
        npu_a = cpu_a.clone().npu()
        cpu_a.remainder_(cpu_b)
        npu_a.remainder_(cpu_b)
        self.assertRtolEqual(cpu_a, npu_a.cpu())


if __name__ == "__main__":
    run_tests()

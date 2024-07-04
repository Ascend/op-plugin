import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices


class TestAngle(TestCase):
    def cpu_op_exec(self, input1):
        output = torch.angle(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        output = torch.angle(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, input2):
        torch.angle(input1, out=input2)
        output = input2.to("cpu")
        output = output.numpy()
        return output

    def test_angle_common_shape_format_fp32(self):
        format_list = [0]
        shape_list = [[2, 9], [2, 13, 4], [32, 64, 128, 3]]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, -1, 1)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)
    
    def test_angle_common_shape_format_fp16(self):
        format_list = [0]
        shape_list = [[2, 9], [2, 13, 4], [32, 64, 128, 3]]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, -1, 1)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)
    
    @SupportedDevices(['Ascend910B'])
    def test_angle_common_shape_format_complex64(self):
        format_list = [0]
        shape_list = [[2, 9], [2, 13, 4], [32, 64, 128, 3]]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, -1, 1)
            cpu_input2, npu_input2 = create_common_tensor(item, -1, 1)
            cpu_input = torch.complex(cpu_input1, cpu_input2)
            npu_input = torch.complex(npu_input1, npu_input2)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_angle_out_common_shape_format_fp32(self):
        format_list = [0]
        shape_list = [[2, 9], [2, 13, 4], [32, 64, 128, 3]]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, -1, 1)
            cpu_input2, npu_input2 = create_common_tensor(item, -1, 1)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec_out(npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_angle_out_common_shape_format_fp16(self):
        format_list = [0]
        shape_list = [[2, 9], [2, 13, 4], [32, 64, 128, 3]]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, -1, 1)
            cpu_input2, npu_input2 = create_common_tensor(item, -1, 1)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec_out(npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(['Ascend910B'])
    def test_angle_out_common_shape_format_complex64(self):
        format_list = [0]
        shape_list = [[2, 9], [2, 13, 4], [32, 64, 128, 3]]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, -1, 1)
            cpu_input2, npu_input2 = create_common_tensor(item, -1, 1)
            cpu_input = torch.complex(cpu_input1, cpu_input2)
            npu_input = torch.complex(npu_input1, npu_input2)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec_out(npu_input, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)

if __name__ == "__main__":
    run_tests()

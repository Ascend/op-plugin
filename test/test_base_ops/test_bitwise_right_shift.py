import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestBitwiseRightShift(TestCase):
    def cpu_op_exec(self, input1, input2):
        output = torch.bitwise_right_shift(input1, input2)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        output = torch.bitwise_right_shift(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_out(self, input1, input2, out):
        torch.bitwise_right_shift(input1, input2, out=out)
        out = out.numpy()
        return out

    def npu_op_exec_out(self, input1, input2, out):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        out = out.to("npu")
        torch.bitwise_right_shift(input1, input2, out=out)
        out = out.to("cpu")
        out = out.numpy()
        return out

    def test_bitwise_right_shift_tensor(self, device="npu"):
        format_list = [0]
        shape_list = [(256, 32, 56)]
        shape_format = [[np.int32, i, j] for i in format_list for j in shape_list]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input2 = torch.tensor([1]).to(torch.int32)
            npu_input2 = cpu_input2.npu()
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_bitwise_right_shift_tensor_out(self, device="npu"):
        shape_format = [
            [[np.int32, 0, [256, 128, 7, 7]], [np.int32, 0, [256, 128, 7, 7]]],
            [[np.int32, 0, [2, 3, 3, 3]], [np.int32, 0, [2, 3, 3, 3]]],
            [[np.int32, 0, [128, 232, 7, 7]], [np.int32, 0, [128, 232, 7, 7]]],
            [[np.int16, 0, [128, 3, 224, 224]], [np.int16, 0, [128, 3, 224, 224]]],
            [[np.int16, 0, [128, 116, 14, 14]], [np.int16, 0, [128, 116, 14, 14]]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], 0, 5)
            cpu_out, npu_out = create_common_tensor(item[1], 0, 1)
            cpu_output = self.cpu_op_exec_out(cpu_input1, cpu_input2, cpu_out)
            npu_output = self.npu_op_exec_out(npu_input1, npu_input2, npu_out)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()

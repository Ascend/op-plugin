import numpy as np
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices


class TestRshift(TestCase):
    def cpu_op_exec(self, input1, input2):
        output = input1.__rshift__(input2)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        output = input1.__rshift__(input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    @SupportedDevices(['Ascend910A'])
    def test_rshift_tensor(self):
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

    def test_rshift_scalar(self):
        format_list = [0]
        shape_list = [(256, 32, 56)]
        shape_format = [[np.int32, i, j] for i in format_list for j in shape_list]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input2 = torch.tensor(1).to(torch.int32)
            npu_input2 = cpu_input2.npu()
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()

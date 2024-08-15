import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestGcd(TestCase):

    def cpu_op_exec(self, input1, input2, output):
        torch.gcd(input1, input2, out=output)
        return output.detach().numpy()

    def npu_op_exec(self, input1, input2, output):
        torch.gcd(input1, input2, out=output)
        output = output.cpu()
        return output.detach().numpy()

    def test_gcd(self):
        shape = (4, 4)
        cpu_input1 = torch.randint(0, 1000, size=shape)
        cpu_input2 = torch.randint(0, 1000, size=shape)
        npu_input1 = cpu_input1.npu()
        npu_input2 = cpu_input2.npu()
        cpu_output = torch.empty(shape, dtype=torch.int32)
        npu_output = cpu_output.npu()

        cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2, cpu_output)
        npu_output = self.npu_op_exec(npu_input1, npu_input2, npu_output)
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()

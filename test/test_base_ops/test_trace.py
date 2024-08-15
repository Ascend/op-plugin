import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestTrace(TestCase):
    def cpu_op_exec(self, input1):
        output = torch.trace(input1)
        output = output.numpy()
        return output
    
    def npu_op_exec(self, input1):
        output = torch.trace(input1)
        output = output.to('cpu')
        output = output.numpy()
        return output
    
    def test_trace_float32(self):
        cpu_input = torch.rand(3, 3)
        npu_input = cpu_input.to('npu')
        cpu_output = self.cpu_op_exec(cpu_input)
        npu_output = self.npu_op_exec(npu_input)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_trace_int(self):
        cpu_input = torch.randint(0, 1024, (3, 3))
        npu_input = cpu_input.to('npu')
        cpu_output = self.cpu_op_exec(cpu_input)
        npu_output = self.npu_op_exec(npu_input)
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()


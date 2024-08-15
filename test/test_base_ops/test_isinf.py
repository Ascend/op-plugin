import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestIsinf(TestCase):

    def test_isneginf(self):
        cpu_input = torch.tensor([1.0, -float('inf'), 2.0, float('inf'), -3.0])
        cpu_output = torch.isneginf(cpu_input)

        npu_input = cpu_input.npu()
        npu_output = torch.isneginf(npu_input)
        self.assertEqual(npu_output, cpu_output)

    def test_isposinf(self):
        cpu_input = torch.tensor([1.0, -float('inf'), 2.0, float('inf'), -3.0])
        cpu_output = torch.isposinf(cpu_input)

        npu_input = cpu_input.npu()
        npu_output = torch.isposinf(npu_input)
        self.assertEqual(npu_output, cpu_output)


if __name__ == "__main__":
    run_tests()

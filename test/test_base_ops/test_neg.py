import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestNeg(TestCase):

    def test_neg(self):
        shape = (4, 4)
        cpu_input = torch.randn(shape, dtype=torch.float32)
        cpu_output = torch.neg(cpu_input)

        npu_input = cpu_input.npu()
        npu_output = torch.neg(npu_input)
        self.assertEqual(npu_output, cpu_output)


if __name__ == "__main__":
    run_tests()

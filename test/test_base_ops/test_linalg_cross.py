import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestLinalgcross(TestCase):

    def test_linalg_cross(self):
        shape = (4, 3)
        cpu_input1 = torch.randn(shape, dtype=torch.float32)
        cpu_input2 = torch.randn(shape, dtype=torch.float32)
        npu_input1, npu_input2 = cpu_input1.npu(), cpu_input2.npu()

        cpu_output = torch.linalg.cross(cpu_input1, cpu_input2)
        npu_output = torch.linalg.cross(npu_input1, npu_input2)

        self.assertEqual(npu_output, cpu_output)


if __name__ == "__main__":
    run_tests()

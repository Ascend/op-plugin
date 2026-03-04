import torch
from torch_npu.testing.testcase import TestCase, run_tests
import op_extension


class TestCustomAdd(TestCase):

    def test_add_custom_ops(self):
        length = [8, 2048]
        x = torch.randint(low=1, high=100, size=length, device='cpu', dtype=torch.int)
        y = torch.randint(low=1, high=100, size=length, device='cpu', dtype=torch.int)

        x_npu = x.npu()
        y_npu = y.npu()
        output = op_extension.ops.custom_add(x_npu, y_npu)
        cpuout = torch.add(x, y)

        self.assertRtolEqual(output, cpuout)

if __name__ == "__main__":
    run_tests()

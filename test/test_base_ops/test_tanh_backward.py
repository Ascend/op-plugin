import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestTanhBackward(TestCase):

    def test_tanh_backward_diff_shape(self):
        gradoutput = torch.randn(16, 16)
        a = torch.randn(16)
        cpu_result = torch.ops.aten.tanh_backward(gradoutput, a)
        npu_result = torch.ops.aten.tanh_backward(gradoutput.npu(), a.npu())
        self.assertRtolEqual(cpu_result, npu_result)

if __name__ == "__main__":
    run_tests()

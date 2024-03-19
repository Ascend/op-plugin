import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestMaxPool2d(TestCase):

    def test_sum_dim_none(self):
        data = torch.tensor(-6.6523, dtype=torch.float16)
        data_npu = data.npu()
        dim = None
        keepdim = False
        cpu_output = data.sum(dim=dim, keepdim=keepdim)
        npu_output = data_npu.sum(dim=dim, keepdim=keepdim)
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()

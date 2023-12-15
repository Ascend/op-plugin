import torch
import torch.nn.functional as F

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestMaxPool2d(TestCase):

    def test_max_pool2d_one_tuple(self):
        data = torch.ones([19, 19, 16])
        cpu_output = F.max_pool2d(data, (14,), (13,), (3, 4), ceil_mode=True)
        npu_output = F.max_pool2d(data.npu(), (14,), (13,), (3, 4), ceil_mode=True)
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()

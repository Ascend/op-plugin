import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestHistc(TestCase):

    def test_histc(self):
        case_list = [
            [np.float32, -1, [2]],
            [np.float32, -1, [2]],
            [np.float32, -1, [2, 3]],
            [np.float32, -1, [2, 3]],
            [np.float32, -1, [2, 3, 4]],
            [np.float32, -1, [2, 3, 4]],
            [np.float32, -1, [2, 3, 4, 5]],
            [np.float32, -1, [2, 3, 4, 5]],
        ]
        for item in case_list:
            for bins in (10, 2048):
                cpu_input, npu_input = create_common_tensor(item, -2, 2)
                cpu_output = torch.histc(cpu_input, bins, -2, 2)
                npu_output = torch.histc(npu_input, bins, -2, 2)
                self.assertRtolEqual(cpu_output, npu_output)

    def test_histc_out(self):
        case_list = [
            [np.float32, -1, [2]],
            [np.float32, -1, [2]],
            [np.float32, -1, [2, 3]],
            [np.float32, -1, [2, 3]],
            [np.float32, -1, [2, 3, 4]],
            [np.float32, -1, [2, 3, 4]],
            [np.float32, -1, [2, 3, 4, 5]],
            [np.float32, -1, [2, 3, 4, 5]],
        ]
        for item in case_list:
            for bins in (10, 2048):
                cpu_input, npu_input = create_common_tensor(item, -2, 2)
                cpu_output = torch.from_numpy(np.zeros([bins], dtype=item[0]))
                npu_output = cpu_output.npu()
                torch.histc(cpu_input, bins, -2, 2, out=cpu_output)
                torch.histc(npu_input, bins, -2, 2, out=npu_output)
                self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()

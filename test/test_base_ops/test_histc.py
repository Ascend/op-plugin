import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices

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

    @SupportedDevices(['Ascend910B'])
    def test_histc_check_logic(self):
        device = "npu"

        # negative bins throws
        with self.assertRaisesRegex(RuntimeError, 'bins must be > 0'):
            torch.histc(torch.tensor([1], dtype=torch.float, device=device), bins=-1)

        # min > max throws
        with self.assertRaisesRegex(RuntimeError, "max must be larger than min"):
            torch.histc(torch.tensor([1., 2., 3.], dtype=torch.float, device=device), bins=4, min=5, max=1)

        # default range with inf should throw
        with self.assertRaisesRegex(RuntimeError, r'range of \[[\w,+\-\.\ ]+\] is not finite'):
            torch.histc(torch.tensor([float("inf")], dtype=torch.float, device=device))

        # min == max with non-empty input falls back to input range
        actual = torch.histc(torch.tensor([1., 2., 1.], dtype=torch.float, device=device), bins=4, min=5, max=5)
        self.assertEqual(torch.tensor([2, 0, 0, 1], dtype=torch.float, device=device), actual)

        # nan is ignored when explicit finite range is provided
        self.assertEqual(torch.histc(torch.tensor([1., 2., float("nan")], dtype=torch.float, device=device), bins=4, max=3), torch.tensor([0, 1, 1, 0], dtype=torch.float, device=device))


if __name__ == "__main__":
    run_tests()

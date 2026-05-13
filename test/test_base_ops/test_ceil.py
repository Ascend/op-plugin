import torch
import numpy as np

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestCeil(TestCase):
    def test_ceil(self):
        cpu_input = torch.randn(10, 10)
        npu_input = cpu_input.to("npu")
        cpu_output = torch.ceil_(cpu_input)
        npu_output = torch.ceil_(npu_input)
        npu_output = npu_output.to("cpu")

        self.assertRtolEqual(cpu_output, npu_output)

    def cpu_op_exec(self, input1):
        output = torch.ceil(input1)
        return output

    def npu_op_exec(self, input1):
        output = torch.ceil(input1)
        output = output.to("cpu")
        return output

    def test_ceil_shape_format(self, device="npu"):
        shape_format = [
            [np.float32, 0, 10],
            [np.float32, 0, (64, 10)],
            [np.float32, 3, (256, 2048, 7, 7)],
            [np.float32, 4, (32, 1, 3, 3)],
            [np.float32, 29, (10, 128)],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_ceil_integer_identity_npu(self):
        """Integer ceil/ceil_ is identity on NPU (no aclnnCeil / aclnnInplaceCeil for integral dtypes)."""
        dtypes = [
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.int64,
        ]
        for dt in dtypes:
            cpu_x = torch.tensor([[1, -2, 7], [-3, 0, 42]], dtype=dt)
            npu_x = cpu_x.npu()

            self.assertEqual(torch.ceil(cpu_x), cpu_x)
            self.assertEqual(torch.ceil(npu_x).cpu(), cpu_x)

            npu_inplace = cpu_x.clone().npu()
            npu_inplace.ceil_()
            self.assertEqual(npu_inplace.cpu(), cpu_x)


if __name__ == "__main__":
    run_tests()

import unittest
import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.common_utils import SupportedDevices


@SupportedDevices(['Ascend910B'])
class TestArgSort(TestCase):
    def cpu_op_exec(self, input1, dim, descending):
        output = torch.argsort(input1, dim=dim, descending=descending)
        return output.numpy()

    def npu_op_exec(self, input1, dim, descending):
        output = torch.argsort(input1, dim=dim, descending=descending)
        return output.cpu().numpy()

    def cpu_default_op_exec(self, input1):
        output = torch.argsort(input1)
        return output.numpy()

    def npu_default_op_exec(self, input1):
        output = torch.argsort(input1)
        return output.cpu().numpy()

    def cpu_op_exec_stable(self, input1, stable, dim, descending):
        output = torch.argsort(input1, stable=stable, dim=dim, descending=descending)
        return output.numpy()

    def npu_op_exec_stable(self, input1, stable, dim, descending):
        output = torch.argsort(input1, stable=stable, dim=dim, descending=descending)
        return output.cpu().numpy()

    def test_sort_shape_format_fp32(self):
        shape_format = [
            [[np.float32, 0, (8, 4, 3, 9)], 2, False],
            [[np.float32, 0, (2, 3)]],
            [[np.float32, 0, (1, 7)], 0, True],
            [[np.float32, 0, (1, 5, 6)], 1, False],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            if len(item) > 1:
                cpu_output = self.cpu_op_exec(cpu_input1, item[1], item[2])
                npu_output = self.npu_op_exec(npu_input1, item[1], item[2])
            else:
                cpu_output = self.cpu_default_op_exec(cpu_input1)
                npu_output = self.npu_default_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_sort_shape_format_fp16(self):
        shape_format = [
            [[np.float16, 0, (8, 4, 3, 9)], 2, False],
            [[np.float16, 0, (2, 3)]],
            [[np.float16, 0, (1, 7)], 0, True],
            [[np.float16, 0, (1, 5, 6)], 1, False],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            if len(item) > 1:
                cpu_output = self.cpu_op_exec(cpu_input1, item[1], item[2])
                npu_output = self.npu_op_exec(npu_input1, item[1], item[2])
            else:
                cpu_output = self.cpu_default_op_exec(cpu_input1)
                npu_output = self.npu_default_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

    @unittest.skipIf("1.11.0" in torch.__version__,
                "OP `argsort.stable` is not supported on torch v1.11.0, skip this ut for this torch version")
    def test_sort_stable_shape_format_fp32(self):
        shape_format = [
            [[np.float32, 0, (8, 4, 3, 9)], True, 2, False],
            [[np.float32, 0, (1, 7)], False, 0, True],
            [[np.float32, 0, (1, 5, 6)], True, 1, False],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_output = self.cpu_op_exec_stable(cpu_input1, item[1], item[2], item[3])
            npu_output = self.npu_op_exec_stable(npu_input1, item[1], item[2], item[3])
            self.assertRtolEqual(cpu_output, npu_output)

    @unittest.skipIf("1.11.0" in torch.__version__,
                "OP `argsort.stable` is not supported on torch v1.11.0, skip this ut for this torch version")
    def test_sort_stable_shape_format_fp16(self):
        shape_format = [
            [[np.float16, 0, (8, 4, 3, 9)], True, 2, False],
            [[np.float16, 0, (1, 7)], False, 0, True],
            [[np.float16, 0, (1, 5, 6)], True, 1, False],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_output = self.cpu_op_exec_stable(cpu_input1, item[1], item[2], item[3])
            npu_output = self.npu_op_exec_stable(npu_input1, item[1], item[2], item[3])
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()

import unittest

import torch
from torch.nn import functional as F
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


# Upstream pytorch#173895 removed named-tensor support in PyTorch 2.13.
# torch.Tensor.refine_names -- the entry point to naming tensor dims that
# these tests rely on -- is gone. Skip the dimname test methods on 2.13+.
_TORCH_HAS_NAMED_TENSOR = hasattr(torch.Tensor, "refine_names")


class TestOnes(TestCase):
    def cpu_op_exec(self, shape, dtype):
        output = torch.ones(size=shape, dtype=dtype)
        output = output.detach().numpy()
        return output

    def npu_op_exec(self, shape, dtype):
        output = torch.ones(size=shape, device='npu', dtype=dtype)
        output = output.to("cpu")
        output = output.detach().numpy()
        return output

    def cpu_op_name_exec(self, shape, name, dtype):
        output = torch.ones(size=shape, names=name, dtype=dtype)
        output = output.detach().numpy()
        return output

    def npu_op_name_exec(self, shape, name, dtype):
        output = torch.ones(size=shape, names=name, device='npu', dtype=dtype)
        output = output.to("cpu")
        output = output.detach().numpy()
        return output

    def cpu_op_out_exec(self, shape, output, dtype):
        torch.ones(size=shape, dtype=dtype, out=output)
        return output

    def npu_op_out_exec(self, shape, output, dtype):
        torch.ones(size=shape, dtype=dtype, device='npu', out=output)
        output = output.to("cpu")
        return output

    def test_ones_format(self):
        shape_format = [
            [(2, 3, 4, 1, 5), torch.float32],
            [(1, 100, 7), torch.int32],
            [(10, 1, 7), torch.int8],
            [(1, 2, 7), torch.uint8],
            [(33, 44, 55), torch.float16],
        ]
        for item in shape_format:
            cpu_output = self.cpu_op_exec(item[0], item[1])
            npu_output = self.npu_op_exec(item[0], item[1])

            self.assertRtolEqual(cpu_output, npu_output)

    def test_ones_out_format(self):
        shape_format = [
            [(2, 3, 4, 1, 5), torch.float32],
            [(1, 100, 7), torch.int32],
            [(10, 1, 7), torch.int8],
            [(1, 2, 7), torch.uint8],
            [(33, 44, 55), torch.float16],
        ]
        for item in shape_format:
            cpu_out = torch.randn(item[0], dtype=torch.float32)
            cpu_out = cpu_out.to(item[1])
            npu_out = cpu_out.to('npu')
            cpu_output = self.cpu_op_out_exec(item[0], cpu_out, item[1])
            npu_output = self.npu_op_out_exec(item[0], npu_out, item[1])

            self.assertRtolEqual(cpu_output, npu_output)

    @unittest.skipUnless(_TORCH_HAS_NAMED_TENSOR, "Named tensor removed in PyTorch 2.13 (pytorch#173895)")
    def test_ones_name_format(self):
        shape_format = [
            [(2, 3, 4, 1, 5), ('A', 'B', 'C', 'D', 'E'), torch.float32],
            [(1, 100, 7), ('C', 'H', 'W'), torch.int32],
            [(10, 1, 7), ('C', 'H', 'W'), torch.int8],
            [(1, 2, 7), ('C', 'H', 'W'), torch.uint8],
            [(33, 44, 55), ('C', 'H', 'W'), torch.float16],
        ]
        for item in shape_format:
            cpu_output = self.cpu_op_name_exec(item[0], item[1], item[2])
            npu_output = self.npu_op_name_exec(item[0], item[1], item[2])

            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()

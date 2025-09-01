import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestDimArange(TestCase):
    def test_dim_arange(self):
        shape_format = [
            [(100,), 0, torch.float32],
            [(100, 20), 0, torch.bfloat16],
            [(20, 100), 1, torch.float16],
            [(200, 10), 1, torch.float64],
        ]

        for item in shape_format:
            like_cpu = torch.randn(item[0], dtype=item[2], device="cpu")
            cpu_output = torch._dim_arange(like_cpu, item[1])

            like_npu = like_cpu.to("npu")
            npu_output = torch._dim_arange(like_npu, item[1]).npu()
            self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())

    def test_dim_arange_int(self):
        shape_format = [
            [(100,), 0, torch.int32],
            [(20, 100), 1, torch.int64]
        ]

        for item in shape_format:
            like_cpu = torch.randint(low=0, high=100, size=item[0], dtype=item[2], device="cpu")
            cpu_output = torch._dim_arange(like_cpu, item[1])

            like_npu = like_cpu.to("npu")
            npu_output = torch._dim_arange(like_npu, item[1]).npu()
            self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())

    def test_dim_arange_1(self):
        shape_format = [
            [(1,), 0, torch.float32],
            [(0,), 0, torch.float32],
            [(1024,), 0, torch.float32],
            [(1, 1), 0, torch.bfloat16],
            [(1, 1000), 1, torch.float16],
            [(200, 1), 0, torch.float64],
            [(0, 100), 0, torch.float32],
            [(2, 3, 4), 0, torch.float32],
            [(2, 3, 4), 1, torch.float32],
            [(2, 3, 4), 2, torch.float32],
        ]

        for item in shape_format:
            like_cpu = torch.randn(item[0], dtype=item[2], device="cpu")
            cpu_output = torch._dim_arange(like_cpu, item[1])
            like_npu = like_cpu.to("npu")
            npu_output = torch._dim_arange(like_npu, item[1]).npu()
            self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())

if __name__ == "__main__":
    run_tests()

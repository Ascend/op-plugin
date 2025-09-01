import torch
import numpy as np
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestSplitWithSizesCopy(TestCase):
    def generate_data(self, shape, dtype):
        data = np.random.rand(*shape).astype(dtype)
        return torch.from_numpy(data)

    def cpu_op_exec_out(self, input_tensor, split_sizes, dim):
        views = torch.split_with_sizes(input_tensor, split_sizes, dim=dim)
        output_tensors = [torch.zeros_like(v) for v in views]
        torch.split_with_sizes_copy(input_tensor, split_sizes, dim=dim, out=output_tensors)
        output_cpu = [out.to("cpu").numpy() for out in output_tensors]
        return output_cpu

    def npu_op_exec_out(self, input_tensor, split_sizes, dim):
        input_tensor = input_tensor.to("npu")
        views = torch.split_with_sizes(input_tensor, split_sizes, dim=dim)
        output_tensors = [torch.zeros_like(v) for v in views]
        torch.split_with_sizes_copy(input_tensor, split_sizes, dim=dim, out=output_tensors)
        output_cpu = [out.to("cpu").numpy() for out in output_tensors]
        return output_cpu

    def cpu_op_exec(self, input_tensor, split_sizes, dim):
        output_tensors = torch.split_with_sizes_copy(input_tensor, split_sizes, dim=dim)
        output_cpu = [out.to("cpu").numpy() for out in output_tensors]
        return output_cpu

    def npu_op_exec(self, input_tensor, split_sizes, dim):
        input_tensor = input_tensor.to("npu")
        output_tensors = torch.split_with_sizes_copy(input_tensor, split_sizes, dim=dim)
        output_cpu = [out.to("cpu").numpy() for out in output_tensors]
        return output_cpu

    def test_split_with_sizes_copy(self):
        shape = (30, 40, 50)
        dtype = np.float32
        x_cpu = self.generate_data(shape, dtype)
        cases = [
            (0, [3, 7, 8, 12]),
            (1, [3, 7, 10, 20]),
            (-2, [3, 7, 10, 20]),
            (2, [3, 7, 10, 12, 18]),
            (-1, [3, 7, 10, 12, 18]),
            (2, [3, 7, 10, 0, 30]),
        ]

        for dim, split_sizes in cases:
            cpu_outs = self.cpu_op_exec_out(x_cpu, split_sizes, dim)
            npu_outs = self.npu_op_exec_out(x_cpu, split_sizes, dim)

            for cpu, npu in zip(cpu_outs, npu_outs):
                self.assertRtolEqual(cpu, npu)

            cpu_outs = self.cpu_op_exec(x_cpu, split_sizes, dim)
            npu_outs = self.npu_op_exec(x_cpu, split_sizes, dim)

            for cpu, npu in zip(cpu_outs, npu_outs):
                self.assertRtolEqual(cpu, npu)


if __name__ == "__main__":
    run_tests()

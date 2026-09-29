import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestNestedToPaddedTensor(TestCase):
    def gen_nested_tensor(self, sub_shapes, dtype, device):
        tensors = []
        for shape in sub_shapes:
            numel = int(np.prod(shape))
            tensors.append(torch.arange(numel, dtype=dtype).reshape(shape).to(device))
        return torch.nested.nested_tensor(tensors, dtype=dtype, device=device)

    def cpu_op_exec(self, nested, padding, output_size=None):
        output = nested.to_padded_tensor(padding, output_size=output_size)
        return output.numpy()

    def npu_op_exec(self, nested, padding, output_size=None):
        output = nested.to_padded_tensor(padding, output_size=output_size)
        return output.cpu().numpy()

    def to_padded_tensor_result(self, sub_shapes, dtype, padding=0.0, output_size=None):
        cpu_nested = self.gen_nested_tensor(sub_shapes, dtype, "cpu")
        npu_nested = self.gen_nested_tensor(sub_shapes, dtype, "npu")
        cpu_output = self.cpu_op_exec(cpu_nested, padding, output_size)
        npu_output = self.npu_op_exec(npu_nested, padding, output_size)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_to_padded_tensor_fp32_32aligned(self):
        sub_shapes = [(32, 32), (32, 32), (32, 32)]
        self.to_padded_tensor_result(sub_shapes, torch.float32)

    def test_to_padded_tensor_fp32_non32aligned(self):
        sub_shapes = [(5, 7), (5, 7), (5, 7)]
        self.to_padded_tensor_result(sub_shapes, torch.float32)

    def test_to_padded_tensor_fp16(self):
        sub_shapes = [(32, 16), (32, 16)]
        self.to_padded_tensor_result(sub_shapes, torch.float16)

    def test_to_padded_tensor_padding(self):
        sub_shapes = [(4, 6), (4, 6), (4, 6)]
        self.to_padded_tensor_result(sub_shapes, torch.float32, padding=1.5)

    def test_to_padded_tensor_output_size(self):
        sub_shapes = [(4, 6), (4, 6)]
        self.to_padded_tensor_result(sub_shapes, torch.float32, output_size=[3, 8, 8])

    def test_to_padded_tensor_3d(self):
        sub_shapes = [(2, 3, 4), (2, 3, 4)]
        self.to_padded_tensor_result(sub_shapes, torch.float32)

    def test_to_padded_tensor_irregular_shapes(self):
        sub_shapes = [(2, 3), (3, 4), (1, 5)]
        self.to_padded_tensor_result(sub_shapes, torch.float32)

    def test_to_padded_tensor_irregular_shapes_padding(self):
        sub_shapes = [(2, 3), (3, 4), (1, 5)]
        self.to_padded_tensor_result(sub_shapes, torch.float32, padding=0.5)

    def test_to_padded_tensor_1d(self):
        sub_shapes = [(3,), (5,), (7,)]
        self.to_padded_tensor_result(sub_shapes, torch.float32)


if __name__ == "__main__":
    run_tests()

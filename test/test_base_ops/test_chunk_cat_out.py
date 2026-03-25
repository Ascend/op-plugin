import copy
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SkipIfNotGteCANNVersion

@SkipIfNotGteCANNVersion("9.0.0")
class TestChunkCat(TestCase):
    def generate_data(self, shape, dtype, device='cpu'):
        return torch.randn(shape, dtype=dtype, device=device)

    def cpu_op_exec(self, x, y, dim, num_chunks, out_dtype):
        out_shape = [2,7]
        out = torch.zeros(out_shape, dtype=out_dtype, device='cpu')
        output = torch._chunk_cat([x, y], dim, num_chunks, out=out)
        return output.numpy()

    def npu_op_exec(self, x, y, dim, num_chunks, out_dtype):
        x_npu = x.to('npu')
        y_npu = y.to('npu')
        out_shape = [2,7]
        out = torch.zeros(out_shape, dtype=out_dtype, device='npu')
        output = torch._chunk_cat([x_npu, y_npu], dim, num_chunks, out=out)
        return output.to("cpu").numpy()

    def test_chunk_cat_float32_dim0_out_fp32(self):
        x = self.generate_data((2, 3, 4), torch.float32)
        y = self.generate_data((2, 3, 4), torch.float32)
        cpu_x = copy.deepcopy(x)
        cpu_y = copy.deepcopy(y)
        cpu_output = self.cpu_op_exec(cpu_x, cpu_y, 0, 2, torch.float32)
        npu_output = self.npu_op_exec(x, y, 0, 2, torch.float32)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_chunk_cat_float32_dim0_out_fp16(self):
        x = self.generate_data((2, 3, 4), torch.float32)
        y = self.generate_data((2, 3, 4), torch.float32)
        cpu_x = copy.deepcopy(x)
        cpu_y = copy.deepcopy(y)
        cpu_output = self.cpu_op_exec(cpu_x, cpu_y, 0, 2, torch.float16)
        npu_output = self.npu_op_exec(x, y, 0, 2, torch.float16)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_chunk_cat_float32_dim0_out_bf16(self):
        x = self.generate_data((2, 3, 4), torch.float32)
        y = self.generate_data((2, 3, 4), torch.float32)
        cpu_x = copy.deepcopy(x)
        cpu_y = copy.deepcopy(y)
        cpu_output = self.cpu_op_exec(cpu_x, cpu_y, 0, 2, torch.bfloat16)
        npu_output = self.npu_op_exec(x, y, 0, 2, torch.bfloat16)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_chunk_cat_float32_dim1_out_fp32(self):
        x = self.generate_data((2, 3, 4), torch.float32)
        y = self.generate_data((2, 3, 4), torch.float32)
        cpu_x = copy.deepcopy(x)
        cpu_y = copy.deepcopy(y)
        cpu_output = self.cpu_op_exec(cpu_x, cpu_y, 1, 2, torch.float32)
        npu_output = self.npu_op_exec(x, y, 1, 2, torch.float32)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_chunk_cat_float16_dim0_out_fp32(self):
        x = self.generate_data((2, 3, 4), torch.float16)
        y = self.generate_data((2, 3, 4), torch.float16)
        cpu_x = copy.deepcopy(x)
        cpu_y = copy.deepcopy(y)
        cpu_output = self.cpu_op_exec(cpu_x, cpu_y, 0, 2, torch.float32)
        npu_output = self.npu_op_exec(x, y, 0, 2, torch.float32)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_chunk_cat_float16_dim0_out_fp16(self):
        x = self.generate_data((2, 3, 4), torch.float16)
        y = self.generate_data((2, 3, 4), torch.float16)
        cpu_x = copy.deepcopy(x)
        cpu_y = copy.deepcopy(y)
        cpu_output = self.cpu_op_exec(cpu_x, cpu_y, 0, 2, torch.float16)
        npu_output = self.npu_op_exec(x, y, 0, 2, torch.float16)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_chunk_cat_float16_dim0_out_bf16(self):
        x = self.generate_data((2, 3, 4), torch.float16)
        y = self.generate_data((2, 3, 4), torch.float16)
        cpu_x = copy.deepcopy(x)
        cpu_y = copy.deepcopy(y)
        cpu_output = self.cpu_op_exec(cpu_x, cpu_y, 0, 2, torch.bfloat16)
        npu_output = self.npu_op_exec(x, y, 0, 2, torch.bfloat16)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_chunk_cat_float16_dim1_out_fp32(self):
        x = self.generate_data((2, 3, 4), torch.float16)
        y = self.generate_data((2, 3, 4), torch.float16)
        cpu_x = copy.deepcopy(x)
        cpu_y = copy.deepcopy(y)
        cpu_output = self.cpu_op_exec(cpu_x, cpu_y, 1, 2, torch.float32)
        npu_output = self.npu_op_exec(x, y, 1, 2, torch.float32)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_chunk_cat_bfloat16_dim0_out_fp32(self):
        x = self.generate_data((2, 3, 4), torch.bfloat16)
        y = self.generate_data((2, 3, 4), torch.bfloat16)
        cpu_x = copy.deepcopy(x)
        cpu_y = copy.deepcopy(y)
        cpu_output = self.cpu_op_exec(cpu_x, cpu_y, 0, 2, torch.float32)
        npu_output = self.npu_op_exec(x, y, 0, 2, torch.float32)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_chunk_cat_bfloat16_dim0_out_fp16(self):
        x = self.generate_data((2, 3, 4), torch.bfloat16)
        y = self.generate_data((2, 3, 4), torch.bfloat16)
        cpu_x = copy.deepcopy(x)
        cpu_y = copy.deepcopy(y)
        cpu_output = self.cpu_op_exec(cpu_x, cpu_y, 0, 2, torch.float16)
        npu_output = self.npu_op_exec(x, y, 0, 2, torch.float16)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_chunk_cat_bfloat16_dim0_out_bf16(self):
        x = self.generate_data((2, 3, 4), torch.bfloat16)
        y = self.generate_data((2, 3, 4), torch.bfloat16)
        cpu_x = copy.deepcopy(x)
        cpu_y = copy.deepcopy(y)
        cpu_output = self.cpu_op_exec(cpu_x, cpu_y, 0, 2, torch.bfloat16)
        npu_output = self.npu_op_exec(x, y, 0, 2, torch.bfloat16)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_chunk_cat_bfloat16_dim1_out_bf16(self):
        x = self.generate_data((2, 3, 4), torch.bfloat16)
        y = self.generate_data((2, 3, 4), torch.bfloat16)
        cpu_x = copy.deepcopy(x)
        cpu_y = copy.deepcopy(y)
        cpu_output = self.cpu_op_exec(cpu_x, cpu_y, 1, 2, torch.bfloat16)
        npu_output = self.npu_op_exec(x, y, 1, 2, torch.bfloat16)
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    np.random.seed(1234)
    run_tests()
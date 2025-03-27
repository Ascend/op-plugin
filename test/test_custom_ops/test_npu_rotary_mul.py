import unittest
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestRotaryMul(TestCase):
    def rotary_mul(self, x, r1, r2):
        x1, x2 = torch.chunk(x, 2, -1)
        x_new = torch.cat((-x2, x1), dim=-1)
        output = r1 * x + r2 * x_new
        return output
    
    def rotary_mul_mode(self, x, r1, r2):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x_new = torch.stack((-x2, x1), dim=-1)
        x_new = x_new.reshape(x.shape)
        res = x * r1 + x_new * r2
        return res

    def gen_data(self, shape, dtype):
        cpu_input = torch.rand(shape, dtype=dtype)
        npu_input = cpu_input.npu()
        return cpu_input, npu_input

    def cpu_to_exec(self, x, r1, r2, mode='half'):
        if mode == 'half':
            out = self.rotary_mul(x, r1, r2)
        else:
            out = self.rotary_mul_mode(x, r1, r2)
        return out.cpu()

    def npu_to_exec(self, x, r1, r2, mode='half'):
        if mode == 'half':
            out = torch_npu.npu_rotary_mul(x, r1, r2)
        else:
            out = torch_npu.npu_rotary_mul(x, r1, r2, rotary_mode=mode)
        return out.cpu()

    @SupportedDevices(['Ascend910B'])
    def test_rotary_mul(self):
        dtype_list = [torch.float16, torch.float32]
        shape_list = [
            [[2, 8192, 5, 128], [1, 8192, 1, 128], [1, 8192, 1, 128]],
            [[8192, 2, 5, 128], [8192, 1, 1, 128], [8192, 1, 1, 128]],
            [[2048, 4, 32, 64], [2048, 4, 1, 64], [2048, 4, 1, 64]],
        ]
        items = [
            [shape, dtype]
            for shape in shape_list
            for dtype in dtype_list
        ]
        for shape, dtype in items:
            cpu_x, npu_x = self.gen_data(shape[0], dtype)
            cpu_r1, npu_r1 = self.gen_data(shape[1], dtype)
            cpu_r2, npu_r2 = self.gen_data(shape[2], dtype)
            cpu_out = self.cpu_to_exec(cpu_x, cpu_r1, cpu_r2)
            npu_out = self.npu_to_exec(npu_x, npu_r1, npu_r2)
            self.assertRtolEqual(cpu_out, npu_out)

    @SupportedDevices(['Ascend910B'])
    def test_rotary_mul_mode(self):
        dtype_list = [torch.float16, torch.float32]
        shape_list = [
            [[2, 2, 5, 128], [1, 2, 1, 128], [1, 2, 1, 128]],
            [[2, 24, 5, 128], [1, 1, 5, 128], [1, 1, 5, 128]],
            [[128, 4, 4, 256], [1, 1, 4, 256], [1, 1, 4, 256]],
            [[64, 8, 8, 512], [1, 1, 8, 512], [1, 1, 8, 512]],
        ]
        items = [
            [shape, dtype]
            for shape in shape_list
            for dtype in dtype_list
        ]
        for shape, dtype in items:
            cpu_x, npu_x = self.gen_data(shape[0], dtype)
            cpu_r1, npu_r1 = self.gen_data(shape[1], dtype)
            cpu_r2, npu_r2 = self.gen_data(shape[2], dtype)
            cpu_out = self.cpu_to_exec(cpu_x, cpu_r1, cpu_r2, mode='interleave')
            npu_out = self.npu_to_exec(npu_x, npu_r1, npu_r2, mode='interleave')
            self.assertRtolEqual(cpu_out, npu_out)

    @SupportedDevices(['Ascend910B'])
    def test_rotary_mul_error_param(self):
        x = torch.rand(2, 2, 5, 128).npu()
        r1 = torch.rand(1, 2, 1, 128).npu()
        r2 = torch.rand(1, 2, 1, 128).npu()
        msg = "The rotary_mode of npu_rotary_mul should be half or interleave, but got "
        with self.assertRaisesRegex(RuntimeError, msg):
            torch_npu.npu_rotary_mul(x, r1, r2, 'quarter')

if __name__ == '__main__':
    run_tests()

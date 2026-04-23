import unittest
import itertools
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices


class TestNpuRotateQuant(TestCase):
    def compare_tensor_nibbles(self, a: torch.Tensor, b: torch.Tensor) -> bool:
        """
        比较两个形状相同的张量，每个元素是 int32，由 8 个 int4 拼成。
        要求对应位置的每个半字节（视为有符号 int4）差值绝对值 ≤ 1。
        """
        if a.shape != b.shape:
            return False

        # 转换为无符号 32 位整数（用 int64 防止溢出）
        a_uint = a.to(torch.int64) & 0xFFFFFFFF
        b_uint = b.to(torch.int64) & 0xFFFFFFFF

        # 准备所有移位偏移（从高位到低位：28, 24, 20, ..., 0）
        shifts = torch.tensor([28 - 4 * i for i in range(8)], dtype=torch.int64, device=a.device)

        # 提取所有半字节：a_nibbles shape = (8, *a.shape)，每个值为 0~15
        a_nibbles = (a_uint.unsqueeze(0) >> shifts.view(-1, *([1] * a.dim()))) & 0xF
        b_nibbles = (b_uint.unsqueeze(0) >> shifts.view(-1, *([1] * b.dim()))) & 0xF

        # 将无符号半字节转换为有符号 int4（范围 -8 ~ 7）
        a_int4 = torch.where(a_nibbles >= 8, a_nibbles - 16, a_nibbles)
        b_int4 = torch.where(b_nibbles >= 8, b_nibbles - 16, b_nibbles)

        # 逐半字节比较差值绝对值
        diff = torch.abs(a_int4 - b_int4)
        mask = diff > 1

        if mask.any():
            # 获取所有错误位置的索引
            indices = torch.nonzero(mask, as_tuple=True)
            nibble_indices = indices[0]          # 半字节位置 (0~7)
            tensor_indices = indices[1:]         # 每个维度的坐标

            print("Found mismatches (|diff| > 1) with int4 interpretation:")
            for i in range(len(nibble_indices)):
                nibble_pos = nibble_indices[i].item()
                tensor_pos = tuple(idx[i].item() for idx in tensor_indices)
                a_val = a_int4[(nibble_pos,) + tensor_pos].item()
                b_val = b_int4[(nibble_pos,) + tensor_pos].item()
                diff_val = abs(a_val - b_val)
                print(f"  - nibble index {nibble_pos}, tensor index {tensor_pos}: a={a_val}, b={b_val}, diff={diff_val}")
            return False
        else:
            return True

    def conv_rot(self,  input_matrix, rot_matrix):
        stride = rot_matrix.shape[0]
        h, w = input_matrix.shape
        num_blocks = w // stride
        input_matrix = input_matrix.view(h, num_blocks, stride).reshape(-1, stride)
        out = torch.matmul(input_matrix, rot_matrix)
        mat_rot = out.reshape(h, w)
        return mat_rot

    def rotate_quant(self, x, rot_matrix, dst_dtype):
        x_rot = self.conv_rot(x, rot_matrix)
        xdtype = x.dtype
        torch_npu.npu.synchronize()
        x_rot_int8, x_rot_scale = torch_npu.npu_dynamic_quant(x_rot.to(xdtype), dst_type=dst_dtype)
        return x_rot_int8, x_rot_scale

    def gen_input_data(self, M, N, K):
        x = torch.randn(M, N, dtype=torch.bfloat16)
        rotation = torch.randn(K, K, dtype=torch.bfloat16)
        # 归一化旋转矩阵
        rotation = rotation / (torch.norm(rotation, dim=1, keepdim=True) + 1e-6)
        return x, rotation

    @unittest.skip("Skipping test_npu_rotate_quant until CANN is updated to support aclnnRotateQuant.")
    @SupportedDevices(['Ascend910B'])
    def test_npu_rotate_quant_int8(self, device="npu"):
        # 生成数据
        M = 512
        N = 1024
        K = 1024
        dst_dtype = torch.int8  # int8
        x, rotation = self.gen_input_data(M, N, K)
        output0, output1 = self.rotate_quant(x.npu(), rotation.npu(), dst_dtype=dst_dtype)
        output0_npu, output1_npu = torch_npu.npu_rotate_quant(x.npu(), rotation.npu(), alpha=0.0, dst_dtype=dst_dtype)
        self.assertEqual(output0, output0_npu.cpu(), 1)
        self.assertRtolEqual(output1, output1_npu.cpu())

    @unittest.skip("Skipping test_npu_rotate_quant until CANN is updated to support aclnnRotateQuant.")
    @SupportedDevices(['Ascend910B'])
    def test_npu_rotate_quant_int4(self, device="npu"):
        # 生成数据
        M = 512
        N = 1024
        K = 1024
        dst_dtype = torch.quint4x2  # int4
        x, rotation = self.gen_input_data(M, N, K)
        output0, output1 = self.rotate_quant(x.npu(), rotation.npu(), dst_dtype=dst_dtype)
        output0_npu, output1_npu = torch_npu.npu_rotate_quant(x.npu(), rotation.npu(), alpha=0.0, dst_dtype=dst_dtype)
        self.compare_tensor_nibbles(output0.cpu(), output0_npu.cpu())
        self.assertRtolEqual(output1, output1_npu.cpu())

if __name__ == "__main__":
    run_tests()

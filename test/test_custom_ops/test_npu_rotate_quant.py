import unittest
import math
import itertools
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices


class TestNpuRotateQuant(TestCase):
    def compare_tensor_nibbles(self, a: torch.Tensor, b: torch.Tensor) -> bool:
        """
        Compare two tensors of the same shape, where each element is int32
        packed from 8 int4 values. Requires that the absolute difference of
        each corresponding nibble (interpreted as signed int4) is <= 1.
        """
        if a.shape != b.shape:
            return False

        # Convert to unsigned 32-bit integers (use int64 to prevent overflow)
        a_uint = a.to(torch.int64) & 0xFFFFFFFF
        b_uint = b.to(torch.int64) & 0xFFFFFFFF

        # Prepare shift offsets (from high to low: 28, 24, 20, ..., 0)
        shifts = torch.tensor([28 - 4 * i for i in range(8)], dtype=torch.int64, device=a.device)

        # Extract all nibbles: a_nibbles shape = (8, *a.shape), each value is 0~15
        a_nibbles = (a_uint.unsqueeze(0) >> shifts.view(-1, *([1] * a.dim()))) & 0xF
        b_nibbles = (b_uint.unsqueeze(0) >> shifts.view(-1, *([1] * b.dim()))) & 0xF

        # Convert unsigned nibbles to signed int4 (range -8 ~ 7)
        a_int4 = torch.where(a_nibbles >= 8, a_nibbles - 16, a_nibbles)
        b_int4 = torch.where(b_nibbles >= 8, b_nibbles - 16, b_nibbles)

        # Compare absolute difference per nibble
        diff = torch.abs(a_int4 - b_int4)
        mask = diff > 1

        if mask.any():
            # Get indices of all mismatched positions
            indices = torch.nonzero(mask, as_tuple=True)
            nibble_indices = indices[0]          # nibble position (0~7)
            tensor_indices = indices[1:]         # coordinates in each dimension

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
        # Normalize the rotation matrix
        rotation = rotation / (torch.norm(rotation, dim=1, keepdim=True) + 1e-6)
        return x, rotation

    @unittest.skip("Skipping test_npu_rotate_quant until CANN is updated to support aclnnRotateQuant.")
    @SupportedDevices(['Ascend910B'])
    def test_npu_rotate_quant_int8(self, device="npu"):
        # Generate input data
        M = 512
        N = 1024
        K = 1024
        dst_dtype = torch.int8
        x, rotation = self.gen_input_data(M, N, K)
        output0, output1 = self.rotate_quant(x.npu(), rotation.npu(), dst_dtype=dst_dtype)
        output0_npu, output1_npu = torch_npu.npu_rotate_quant(x.npu(), rotation.npu(), dst_dtype=dst_dtype)
        self.assertEqual(output0, output0_npu.cpu(), 1)
        self.assertRtolEqual(output1, output1_npu.cpu())

    @SupportedDevices(['Ascend910B'])
    @unittest.skip("Skipping test_npu_rotate_quant until CANN is updated to support aclnnRotateQuant.")
    def test_npu_rotate_quant_int4(self, device="npu"):
        # Generate input data
        M = 512
        N = 1024
        K = 1024
        dst_dtype = torch.quint4x2
        x, rotation = self.gen_input_data(M, N, K)
        output0, output1 = self.rotate_quant(x.npu(), rotation.npu(), dst_dtype=dst_dtype)
        output0_npu, output1_npu = torch_npu.npu_rotate_quant(x.npu(), rotation.npu(), dst_dtype=dst_dtype)
        self.compare_tensor_nibbles(output0.cpu(), output0_npu.cpu())
        self.assertRtolEqual(output1, output1_npu.cpu())

    def block_rotation(self, x, rotation, block_size, axis):
        blockNum = x.shape[axis] // block_size
        origin_shape = x.shape

        if (len(rotation.shape) == 2):
            x_r = torch.mm(x.reshape(-1, block_size), rotation, out_dtype=torch.float)
        else:
            x = x.reshape(-1, blockNum, block_size).permute(1, 0, 2)
            x_r = torch.bmm(x, rotation, out_dtype=torch.float).permute(1, 0, 2)
        return x_r.reshape(origin_shape).to(torch.bfloat16)

    def npu_rotate_quant_golden(self, x, rotation, alpha, axis, round_mode, scale_alg, dst_type_max, dst_dtype):
        # This benchmark requires torch_npu >= 2.8 (out_dtype parameter needs 2.8+)
        blockSize = rotation.shape[-1]
        x_r = self.block_rotation(x, rotation, blockSize, axis)
        # Accept both torch.dtype and int enum value
        npu_dst_type = dst_dtype if isinstance(dst_dtype, torch.dtype) else {
            23: torch.float8_e5m2,
            24: torch.float8_e4m3fn,
            296: torch_npu.float4_e2m1fn_x2,
        }.get(dst_dtype, torch.float8_e5m2)

        if alpha is not None and alpha.numel() > 0 and alpha[0] > 0 and alpha[0] < 1:
            x_c = x_r.view(-1, 32)
            groupMaxVal = x_c.abs().amax(dim=-1, keepdim=True)
            limit = alpha[0] * groupMaxVal
            x_c = torch.clamp(x_c, min=-limit, max=limit)
            x_r = x_c.view(x_r.shape)

        out, mxscale = torch_npu.npu_dynamic_mx_quant(
            x_r, axis=axis, round_mode=round_mode, dst_type=npu_dst_type,
            block_size=32, scale_alg=scale_alg, dst_type_max=dst_type_max
        )
        return out.to(torch.float), mxscale.to(torch.float)

    @SupportedDevices(['Ascend950'])
    @unittest.skip("Skipping test_npu_rotate_quant until CANN is updated to support aclnnRotateQuant.")
    def test_npu_rotate_quant_fp4(self, device="npu"):
        # Generate input data
        M = 512
        N = 1024
        K = 128
        dst_dtype = torch_npu.float4_e2m1fn_x2  # fp4, enum value 296
        x, rotation = self.gen_input_data(M, N, K)
        # Golden benchmark: mm with out_dtype=torch.float + npu_dynamic_mx_quant
        output0, output1 = self.npu_rotate_quant_golden(
            x.npu(), rotation.npu(), alpha=None, axis=-1,
            round_mode="rint", scale_alg=0, dst_type_max=0.0,
            dst_dtype=dst_dtype
        )
        # Operator under test
        output0_npu, output1_npu = torch_npu.npu_rotate_quant(x.npu(), rotation.npu(), dst_dtype=dst_dtype)
        self.assertEqual(output0, output0_npu.to(torch.float))
        self.assertEqual(output1, output1_npu.view(torch.uint8).to(torch.float))

    @SupportedDevices(['Ascend950'])
    @unittest.skip("Skipping test_npu_rotate_quant until CANN is updated to support aclnnRotateQuant.")
    def test_npu_rotate_quant_fp8_e5m2(self, device="npu"):
        # Generate input data: 3D input to verify A5 platform supports 1~7 dims
        M = 256
        N = 512
        K = 64
        dst_dtype = torch.float8_e5m2  # fp8_e5m2, enum value 23
        x, rotation = self.gen_input_data(M, N, K)
        x = x.view(4, 64, N)  # reshape to 3D, last dim 512 is divisible by K=64
        # Golden benchmark: mm with out_dtype=torch.float + npu_dynamic_mx_quant
        output0, output1 = self.npu_rotate_quant_golden(
            x.npu(), rotation.npu(), alpha=None, axis=-1,
            round_mode="rint", scale_alg=0, dst_type_max=0.0,
            dst_dtype=dst_dtype
        )
        # Operator under test
        output0_npu, output1_npu = torch_npu.npu_rotate_quant(x.npu(), rotation.npu(), dst_dtype=dst_dtype)
        self.assertEqual(output0, output0_npu.to(torch.float))
        self.assertEqual(output1, output1_npu.view(torch.uint8).to(torch.float))

    @SupportedDevices(['Ascend950'])
    @unittest.skip("Skipping test_npu_rotate_quant until CANN is updated to support aclnnRotateQuant.")
    def test_npu_rotate_quant_fp8_e4m3fn(self, device="npu"):
        # Generate input data: K=32, the minimum supported rotation size
        M = 128
        N = 256
        K = 32
        dst_dtype = torch.float8_e4m3fn  # fp8_e4m3fn, enum value 24
        x, rotation = self.gen_input_data(M, N, K)
        # Golden benchmark: mm with out_dtype=torch.float + npu_dynamic_mx_quant
        output0, output1 = self.npu_rotate_quant_golden(
            x.npu(), rotation.npu(), alpha=None, axis=-1,
            round_mode="rint", scale_alg=0, dst_type_max=0.0,
            dst_dtype=dst_dtype
        )
        # Operator under test
        output0_npu, output1_npu = torch_npu.npu_rotate_quant(x.npu(), rotation.npu(), dst_dtype=dst_dtype)
        self.assertEqual(output0, output0_npu.to(torch.float))
        self.assertEqual(output1, output1_npu.view(torch.uint8).to(torch.float))

if __name__ == "__main__":
    run_tests()

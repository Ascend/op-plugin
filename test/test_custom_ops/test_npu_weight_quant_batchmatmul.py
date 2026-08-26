import math
import unittest
import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNPUWeightQuantBatchMatmul(TestCase):

    def supported_op_exec(self, x, weight, antiquant_scale, antiquant_offset=None):
        if antiquant_offset is not None:
            weight = weight + antiquant_offset
        res = torch.matmul(x, weight * antiquant_scale)
        return res

    def custom_op_exec(self, x, weight, antiquant_scale, antiquant_offset, weight_dtype=None):
        return torch_npu.npu_weight_quant_batchmatmul(x, weight, antiquant_scale, antiquant_offset, weight_dtype=weight_dtype)

    @staticmethod
    def pack_int4_along_n(weight_int4):
        # [K, N] int4 值（int8 存储，-8~7）-> uint8 载体 [K, N/2]，低 4 位为偶数列
        w = (weight_int4.numpy() & 0xF).astype(np.uint8)
        return torch.from_numpy((w[:, 1::2] << 4) | w[:, 0::2])

    @staticmethod
    def pack_int4_along_k(weight_int4):
        # [K, N] int4 值 -> uint8 载体物理 [N, K/2]（沿 K 打包，低 4 位为偶数行）
        w = (weight_int4.numpy() & 0xF).astype(np.uint8)
        return torch.from_numpy(np.ascontiguousarray((w[1::2, :].T << 4) | w[0::2, :].T))

    @staticmethod
    def gen_fp16_exact_values(shape, pool):
        # 从 fp16 精确可表示的值池中取样，保证与 CPU 反量化参考比对时基本无舍入差
        return pool[torch.randint(0, len(pool), shape)]

    @SupportedDevices(['Ascend310P'])
    def test_npu_weight_quant_batchmatmul2(self, device="npu"):
        torch.manual_seed(0)
        x = torch.randn((4, 32, 1024, 128), dtype=torch.float16).npu()
        weight = torch.randn((4, 32, 128, 1024), dtype=torch.int8).npu()
        antiquant_scale = torch.randn((1, 1024), dtype=torch.float16).npu()
        antiquant_offset = torch.randn((1, 1024), dtype=torch.float16).npu()

        x_clone = x.clone()
        weight_clone = weight.clone()
        antiquant_scale_clone = antiquant_scale.clone()
        antiquant_offset_clone = antiquant_offset.clone()

        supported_output = self.supported_op_exec(
            x, weight, antiquant_scale, antiquant_offset)
        custom_output = self.custom_op_exec(
            x_clone, weight_clone, antiquant_scale_clone, antiquant_offset_clone)

        self.assertRtolEqual(supported_output, custom_output, 0.001)

    @SupportedDevices(['Ascend950'])
    def test_npu_weight_quant_batchmatmul2_with_hifloat8(self, device="npu"):
        torch.manual_seed(0)
        x = torch.randn((96, 320), dtype=torch.float16).npu()
        weight = torch.randn((320, 256), dtype=torch.float32).npu()
        antiquant_scale = torch.randn((1, 256), dtype=torch.float16).npu()
        weight_hif8 = torch_npu.npu_dtype_cast(weight, torch_npu.hifloat8)

        x_clone = x.clone()
        weight_clone = weight.clone()
        weight_hif8_clone = weight_hif8.clone()
        antiquant_scale_clone = antiquant_scale.clone()

        supported_output = self.supported_op_exec(x, weight, antiquant_scale)
        custom_output = self.custom_op_exec(x_clone, weight_hif8_clone, antiquant_scale_clone, None, torch_npu.hifloat8)

        self.assertRtolEqual(supported_output, custom_output, 0.001)

    @SupportedDevices(['Ascend950'])
    def test_npu_weight_quant_batchmatmul2_with_A16W4_nz_perchannel(self, device="npu"):
        torch.manual_seed(0)
        m = 1
        k = 128
        n = 256
        group_size = 64
        cpu_x = torch.randn((m, k), dtype=torch.float16)
        cpu_weight = torch.randint(low=3, high=4, size=(k, n), dtype=torch.int32)
        cpu_antiquant_scale = torch.randn((1, 256), dtype=torch.float16)

        npu_x = cpu_x.clone().npu()
        npu_weight = cpu_weight.clone().npu()
        npu_weight = torch_npu.npu_format_cast(cpu_weight.npu(), 29, customize_dtype=cpu_x.dtype)
        npu_weight = torch_npu.npu_convert_weight_to_int4pack(npu_weight)
        npu_antiquant_scale = cpu_antiquant_scale.clone().npu()

        supported_output = self.supported_op_exec(cpu_x, cpu_weight, cpu_antiquant_scale)
        custom_output = self.custom_op_exec(npu_x, npu_weight, npu_antiquant_scale, None, None)

        self.assertRtolEqual(supported_output, custom_output, 0.001)

    @SupportedDevices(['Ascend950'])
    def test_npu_weight_quant_batchmatmul2_with_A16W4_nd_uint8_perchannel(self, device="npu"):
        # uint8 载体紧凑打包 int4（沿 N 打包，ND 非转置视图 [K, N/2]，stride(-1)==1），
        # weight_dtype=torch_npu.int4 时 N 维按每字节 2 个 4-bit 还原
        torch.manual_seed(0)
        m, k, n = 16, 256, 128
        x_pool = torch.tensor([0.5, -0.5, 1.0, -1.0], dtype=torch.float16)
        scale_pool = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float16)
        cpu_x = self.gen_fp16_exact_values((m, k), x_pool)
        cpu_weight = torch.randint(low=-8, high=8, size=(k, n), dtype=torch.int8)
        cpu_antiquant_scale = self.gen_fp16_exact_values((n,), scale_pool)

        npu_x = cpu_x.clone().npu()
        npu_weight = self.pack_int4_along_n(cpu_weight).npu()
        npu_antiquant_scale = cpu_antiquant_scale.clone().npu()

        supported_output = torch.matmul(cpu_x.float(), cpu_weight.float() * cpu_antiquant_scale.float())
        custom_output = self.custom_op_exec(
            npu_x, npu_weight, npu_antiquant_scale, None, torch_npu.int4)

        self.assertEqual(custom_output.shape, supported_output.shape)
        self.assertEqual(custom_output.dtype, torch.float16)
        self.assertRtolEqual(supported_output, custom_output.cpu().float(), 0.001)

    @SupportedDevices(['Ascend950'])
    def test_npu_weight_quant_batchmatmul2_with_A16W4_nd_uint8_transposed(self, device="npu"):
        # uint8 载体紧凑打包 int4（沿 K 打包，转置视图 [K/2, N]，stride(-2)==1），
        # weight_dtype=torch_npu.int4 时 K 维按每字节 2 个 4-bit 还原
        torch.manual_seed(0)
        m, k, n = 16, 256, 128
        x_pool = torch.tensor([0.5, -0.5, 1.0, -1.0], dtype=torch.float16)
        scale_pool = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float16)
        cpu_x = self.gen_fp16_exact_values((m, k), x_pool)
        cpu_weight = torch.randint(low=-8, high=8, size=(k, n), dtype=torch.int8)
        cpu_antiquant_scale = self.gen_fp16_exact_values((n,), scale_pool)

        npu_x = cpu_x.clone().npu()
        npu_weight = self.pack_int4_along_k(cpu_weight).npu().transpose(0, 1)
        npu_antiquant_scale = cpu_antiquant_scale.clone().npu()

        supported_output = torch.matmul(cpu_x.float(), cpu_weight.float() * cpu_antiquant_scale.float())
        custom_output = self.custom_op_exec(
            npu_x, npu_weight, npu_antiquant_scale, None, torch_npu.int4)

        self.assertEqual(custom_output.shape, supported_output.shape)
        self.assertEqual(custom_output.dtype, torch.float16)
        self.assertRtolEqual(supported_output, custom_output.cpu().float(), 0.001)


if __name__ == "__main__":
    run_tests()

import math
import unittest
import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNPUConvertWeightToINT4Pack(TestCase):

    def supported_op_exec(self, x, weight, antiquant_scale, antiquant_offset):
        x = x.to(torch.float32)
        if antiquant_offset != None:
            weight = weight.to(torch.float16) + antiquant_offset

        res = torch.matmul(x, (weight * antiquant_scale).to(torch.float32))
        return res

    def custom_op_exec(self, x, weight, antiquant_scale, antiquant_offset, antiquant_group_size=0):
        return torch_npu.npu_weight_quant_batchmatmul(x, weight, antiquant_scale, antiquant_offset,
                                                      antiquant_group_size=antiquant_group_size)

    @SupportedDevices(['Ascend910B', 'Ascend910_95'])
    def test_npu_convert_weight_to_int4pack(self, device="npu"):
        torch.manual_seed(0)
        m = 128
        k = 64
        n = 32
        trans_weight = False

        cpu_x = torch.randn((m, k), dtype=torch.float16)
        if trans_weight:
            cpu_weight = torch.randint(low=-8, high=8, size=(n, k), dtype=torch.int32)
            cpu_antiquantscale = torch.randn((n, 1), dtype=torch.float16)
            cpu_antiquantoffset = torch.randn((n, 1), dtype=torch.float16)
        else:
            cpu_weight = torch.randint(low=-8, high=8, size=(k, n), dtype=torch.int32)
            cpu_antiquantscale = torch.randn((1, n), dtype=torch.float16)
            cpu_antiquantoffset = torch.randn((1, n), dtype=torch.float16)

        weight_int4 = torch_npu.npu_convert_weight_to_int4pack(cpu_weight.npu())

        if trans_weight:
            cpu_weight = cpu_weight.transpose(-1, -2)
            weight_int4 = weight_int4.transpose(-1, -2)
            cpu_antiquantscale = cpu_antiquantscale.transpose(-1, -2)
            cpu_antiquantoffset = cpu_antiquantoffset.transpose(-1, -2)

        supported_output = self.supported_op_exec(
            cpu_x, cpu_weight, cpu_antiquantscale, cpu_antiquantoffset)
        custom_output = self.custom_op_exec(
            cpu_x.npu(), weight_int4.npu(), cpu_antiquantscale.npu(), cpu_antiquantoffset.npu())

        self.assertRtolEqual(supported_output.to(torch.float16), custom_output, 0.001)

    @SupportedDevices(['Ascend910_95'])
    def test_npu_convert_weight_to_int4pack(self, device="npu"):
        torch.manual_seed(0)
        m = 128
        k = 64
        n = 64
        group_size = 32
        trans_weight = False

        cpu_x = torch.randn((m, k), dtype=torch.float16)
        if trans_weight:
            cpu_weight = torch.randint(low=-3, high=3, size=(n, k), dtype=torch.float32)
            cpu_antiquantscale = torch.randint(low=124, high=130, size=(n, k//group_size), dtype=torch.uint8)
        else:
            cpu_weight = torch.randint(low=-3, high=3, size=(k, n), dtype=torch.float32)
            cpu_antiquantscale = torch.randint(low=124, high=130, size=(k//group_size, n), dtype=torch.uint8)

        weight_fp4 = torch_npu.npu_convert_weight_to_int4pack(cpu_weight.npu())

        cpu_antiquantscale_cpu = (2 ** (cpu_antiquantscale.to(torch.float64) - 127))
        cpu_antiquantscale_cpu = torch.repeat_interleave(cpu_antiquantscale_cpu, group_size, dim=0)
        if trans_weight:
            cpu_weight = cpu_weight.transpose(-1, -2)
            weight_fp4 = weight_fp4.transpose(-1, -2)
            cpu_antiquantscale = cpu_antiquantscale.transpose(-1, -2)

        supported_output = self.supported_op_exec(
            cpu_x, cpu_weight, cpu_antiquantscale_cpu.to(torch.float32), None)
        custom_output = self.custom_op_exec(
            cpu_x.npu(), weight_fp4.npu(), cpu_antiquantscale.npu(), None, antiquant_group_size=group_size)

        self.assertRtolEqual(supported_output.to(torch.float16), custom_output, 0.001)

if __name__ == "__main__":
    run_tests()

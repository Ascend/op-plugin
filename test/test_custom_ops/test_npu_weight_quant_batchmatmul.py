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

    @SupportedDevices(['Ascend910_95'])
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

    @SupportedDevices(['Ascend910_95'])
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


if __name__ == "__main__":
    run_tests()

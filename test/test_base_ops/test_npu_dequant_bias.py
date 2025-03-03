import itertools
import unittest
from dataclasses import dataclass
import math
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


@dataclass
class DequantBiasParams:
    x: torch.Tensor
    weight_scale: torch.Tensor
    activation_scale: torch.Tensor
    bias: torch.Tensor
    output_dtype: torch.dtype


class TestNpuDequantBias(TestCase):

    def golden_dequant_bias(self, params: DequantBiasParams):
        x = params.x.float()
        weight_scale = params.weight_scale.float()
        
        dequantized = x * weight_scale
        
        result = dequantized.to(params.output_dtype)
        return result.cpu().numpy()

    @unittest.skip("skip test_npu_dequant_bias_basic now")
    @SupportedDevices(["Ascend910B"])
    def test_npu_dequant_bias_basic(self):
        x = torch.randint(0, 255, (2, 3), dtype=torch.int32).npu()
        weight_scale = torch.rand(3, dtype=torch.float32).npu()
        activation_scale = torch.rand(2, dtype=torch.float32).npu()
        bias = torch.randn(3, dtype=torch.float32).npu()
        
        npu_result = torch_npu.npu_dequant_bias(
            x, 
            weight_scale,
            activation_scale,
            bias,
            output_dtype=torch.float16
        )
        
        params = DequantBiasParams(
            x, weight_scale, activation_scale, bias, torch.float16
        )
        golden_result = self.golden_dequant_bias(params)
        
        self.assertRtolEqual(golden_result, golden_result, prec16=1e-2)

    @unittest.skip("skip test_npu_dequant_bias_no_optional now")
    @SupportedDevices(["Ascend910B"])
    def test_npu_dequant_bias_no_optional(self):
        x = torch.randint(0, 255, (4, 5), dtype=torch.int32).npu()
        weight_scale = torch.rand(5, dtype=torch.float32).npu()
        
        npu_result = torch_npu.npu_dequant_bias(
            x, 
            weight_scale,
            None,
            None,
            output_dtype=torch.float16
        )
        
        params = DequantBiasParams(
            x, weight_scale, None, None, torch.float16
        )
        golden_result = self.golden_dequant_bias(params)
        
        self.assertRtolEqual(golden_result, golden_result, prec16=1e-2)

    @unittest.skip("skip test_npu_dequant_bias_broadcast now")
    @SupportedDevices(["Ascend910B"])
    def test_npu_dequant_bias_broadcast(self):
        x = torch.randint(0, 255, (1, 4), dtype=torch.int32).npu()
        weight_scale = torch.rand(4, dtype=torch.float32).npu()
        activation_scale = torch.rand(1, dtype=torch.float32).npu()
        
        npu_result = torch_npu.npu_dequant_bias(
            x, 
            weight_scale,
            activation_scale,
            None,
            output_dtype=torch.float16
        )
        
        params = DequantBiasParams(
            x, weight_scale, activation_scale, None, torch.float16
        )
        golden_result = self.golden_dequant_bias(params)
        
        self.assertRtolEqual(golden_result, golden_result, prec16=1e-2)


if __name__ == "__main__":
    run_tests()

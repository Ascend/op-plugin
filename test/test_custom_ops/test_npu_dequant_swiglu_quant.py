import os
import shutil
import unittest

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNPUDequantSwigluQuant(TestCase):

    def golden_dequant_swiglu_quant_torch(
        self,
        x,
        weight_scale,
        activation_scale,
        bias,
        quant_scale,
        quant_offset,
        group_num,
        activate_left,
        quant_mode,
    ):
        x = x.to(torch.float32)
        weight_scale = weight_scale.to(torch.float32)
        activation_scale= activation_scale.to(torch.float32)
        quant_scale = quant_scale.to(torch.float32)
        res = torch.mul(x, weight_scale)
        res = torch.mul(res, activation_scale)
        out = torch.chunk(res, 2, dim=-1)
        if activate_left:
            self_tensor = out[0]
            other = out[1]
        else:
            self_tensor = out[1]
            other = out[0]

        output = torch.nn.functional.silu(self_tensor) * other
        output = torch.mul(output, quant_scale)

        scale_out = torch.zeros([x.shape[0]], dtype=torch.float32)
        if quant_mode == "dynamic":
            abs = torch.abs(output)
            max_values = torch.amax(abs, dim=-1)
            scale_out = max_values / 127.0
            max_values = 127.0 / max_values
            output = output * max_values.unsqueeze(1)
        output = torch.clamp(output, -128, 127)
        output = torch.round(output)
        return output.to(torch.int8).cpu().numpy(), scale_out.cpu().numpy()

    @unittest.skip("skip test_npu_dequant_swiglu_quant now")
    @SupportedDevices(["Ascend910B"])
    def test_npu_dequant_swiglu_quant(self, device="npu"):
        x_shape = [4608, 2048]
        x = torch.randint(-10, 10, x_shape, dtype=torch.int32).npu()
        weight_scale = torch.randn(x_shape[1], dtype=torch.float32).npu()
        activatition_scale = torch.randn((x_shape[0], 1), dtype=torch.float32).npu()
        bias = None
        quant_scale = torch.randn((1, x_shape[1] // 2), dtype=torch.float32).npu()
        quant_offset = None
        group_index = None
        for _ in range(10):
            y_npu, scale_npu = torch_npu.npu_dequant_swiglu_quant(
                x,
                weight_scale=weight_scale,
                activation_scale=activatition_scale,
                bias=bias,
                quant_scale=quant_scale,
                quant_offset=quant_offset,
                group_index=group_index,
                activate_left=False,
                quant_mode=1,
            )

        cpu_out = self.golden_dequant_swiglu_quant_torch(
            x.cpu(),
            weight_scale.cpu(),
            activatition_scale.cpu(),
            bias,
            quant_scale.cpu(),
            quant_offset,
            group_index,
            False,
            "dynamic",
        )

        self.assertRtolEqual(cpu_out[0], y_npu.cpu().numpy())
        self.assertRtolEqual(cpu_out[1], scale_npu.cpu().numpy())


if __name__ == "__main__":
    run_tests()

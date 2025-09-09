import os
import shutil
import unittest
import numpy
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNPUDequantSwigluQuant(TestCase):

    def golden_dequant_swiglu_quant_torch(
        self,
        x,
        weight_scale,
        activate_scale,
        bias,
        quant_scale,
        quant_offset,
        group_index,
        activate_left,
        quant_mode,
        swiglu_mode=0,
        clamp_limit=7.0,
        glu_alpha=1.702,
        glu_bias=1.0,
    ):
        x_dtype = x.dtype
        if len(x.shape) > 2:
            x = x.reshape(-1, x.shape[-1])

        if weight_scale is not None and len(weight_scale.shape) == 1:
            weight_scale = weight_scale.reshape(1, -1)

        if activate_scale is not None and len(activate_scale.shape) >= 1:
            activate_scale = activate_scale.reshape(-1, 1)
        if quant_mode == 1:
            if quant_scale is not None and len(quant_scale.shape) == 1:
                quant_scale = quant_scale.reshape(1, -1)

        if group_index is None:
            group_index = torch.randn([x.shape[0]])

        if quant_mode == 1:
            quant_mode = "dynamic"
        elif quant_mode == 0:
            quant_mode = "static"

        res_y = torch.zeros([x.shape[0], x.shape[1] // 2], dtype=torch.float32)
        res_scale = torch.zeros([x.shape[0]], dtype=torch.float32)

        offset = 0
        for g_idx in range(group_index.shape[0]):
            groupIdx = group_index[g_idx].to(torch.int32)
            x_tensor = x[offset: (offset + groupIdx)].to(torch.float32)
            if "int32" in str(x_dtype):
                if bias is not None and bias.dtype is torch.int32:
                    x_tensor = torch.add(x_tensor, bias[g_idx])
                res = torch.mul(x_tensor, weight_scale[g_idx].to(torch.float32))

                if activate_scale is not None:
                    res = torch.mul(res, activate_scale[offset: (offset + groupIdx)].to(torch.float32))

                if bias is not None and bias.dtype is not torch.int32:
                    res = torch.add(res, bias[g_idx].to(torch.float32))
            else:
                res = x_tensor

            if swiglu_mode == 1:
                self_tensor = res[..., ::2]
                other = res[..., 1::2]
            else:
                out = torch.chunk(res, 2, dim=-1)
                if activate_left:
                    self_tensor = out[0]
                    other = out[1]
                else:
                    self_tensor = out[1]
                    other = out[0]

            if swiglu_mode == 1:
                self_tensor = self_tensor.clamp(min=None, max=clamp_limit)
                other = other.clamp(min=-clamp_limit, max=clamp_limit)
                self_tensor = self_tensor * torch.sigmoid(glu_alpha * self_tensor)
                output = self_tensor * (other + glu_bias)
            else:
                output = torch.nn.functional.silu(self_tensor) * other

            if quant_scale is not None:
                if quant_mode == "static":
                    if(len(quant_scale.shape) == 1):
                        quant_scale = quant_scale.unsqueeze(1).expand(-1, output.shape[1])
                    if(len(quant_offset.shape) == 1):
                        quant_offset = quant_offset.unsqueeze(1).expand(-1, output.shape[1])
                    output = torch.div(output, quant_scale[g_idx].to(torch.float32))
                    output = torch.add(output, quant_offset[g_idx].to(torch.float32))

                    scale_out = torch.tensor(0.0)
                else:
                    output = torch.mul(output, quant_scale[g_idx].to(torch.float32))
                    absd = torch.abs(output)
                    max_values = torch.amax(absd, dim=-1)
                    scale_out = max_values / 127
                    max_values = 127 / max_values
                    output = output * max_values.unsqueeze(1)

            output = torch.clamp(output, -128, 127)
            output = torch.round(output)
            res_y[offset: (offset + groupIdx)] = output
            res_scale[offset: (offset + groupIdx)] = scale_out
            offset = offset + groupIdx

        return res_y.to(torch.int8), res_scale

    @unittest.skip("Skip until CANN is updated to 8.3.RC1 to support aclnnDequantSwigluQuantV2")
    @SupportedDevices(["Ascend910B"])
    def test_npu_dequant_swiglu_quant_1(self, device="npu"):
        swiglu_mode = 0
        bias = None
        quant_offset = None
        x_shape = [4608, 2048]
        x = torch.randint(-10, 10, x_shape, dtype=torch.int32)
        weight_scale = torch.randn(x_shape[1], dtype=torch.float32)
        activate_scale = None
        quant_scale = torch.randn((1, x_shape[1] // 2), dtype=torch.float32)
        group_index = torch.tensor([x.shape[0]])
        quant_mode = 1
        if quant_mode == 0:
            quant_offset = torch.randn((1, x_shape[1] // 2), dtype=torch.float32)

        y_cpu, scale_cpu = self.golden_dequant_swiglu_quant_torch(
            x,
            weight_scale,
            activate_scale,
            bias,
            quant_scale,
            quant_offset,
            group_index,
            activate_left=True,
            quant_mode=quant_mode,
            swiglu_mode=swiglu_mode,
            clamp_limit=7.0,
            glu_alpha=1.702,
            glu_bias=1.0,
            )

        group_index_npu = group_index.npu() if group_index is not None else None
        bias_npu = bias.npu() if bias is not None else None
        if quant_offset is not None:
            quant_offset = quant_offset.npu()
        y_npu, scale_npu = torch_npu.npu_dequant_swiglu_quant(
            x.npu(),
            weight_scale=weight_scale.npu(),
            activation_scale=activate_scale,
            bias=bias_npu,
            quant_scale=quant_scale.npu(),
            quant_offset=quant_offset,
            group_index=group_index_npu,
            activate_left=True,
            quant_mode=quant_mode,
            swiglu_mode=swiglu_mode,
            clamp_limit=7.0,
            glu_alpha=1.702,
            glu_bias=1.0,
            )

        self.assertRtolEqual(y_cpu.numpy(), y_npu.cpu().numpy())
        self.assertRtolEqual(scale_cpu.numpy(), scale_npu.cpu().numpy())

    @unittest.skip("Skip until CANN is updated to 8.3.RC1 to support aclnnDequantSwigluQuantV2")
    @SupportedDevices(["Ascend910B"])
    def test_npu_dequant_swiglu_quant_2(self, device="npu"):
        swiglu_mode = 1
        bias = None
        quant_offset = None
        x_shape = [4608, 2048]
        x = torch.randint(-10, 10, x_shape, dtype=torch.int32)
        weight_scale = torch.randn(x_shape[1], dtype=torch.float32)
        activate_scale = torch.randn((x_shape[0], 1), dtype=torch.float32)
        quant_scale = torch.randn((1, x_shape[1] // 2), dtype=torch.float32)
        group_index = torch.tensor([x.shape[0]])
        quant_mode = 1
        if quant_mode == 0:
            quant_offset = torch.randn((1, x_shape[1] // 2), dtype=torch.float32)

        y_cpu, scale_cpu = self.golden_dequant_swiglu_quant_torch(
            x,
            weight_scale,
            activate_scale,
            bias,
            quant_scale,
            quant_offset,
            group_index,
            activate_left=True,
            quant_mode=quant_mode,
            swiglu_mode=swiglu_mode,
            clamp_limit=7.0,
            glu_alpha=1.702,
            glu_bias=1.0,
            )

        group_index_npu = group_index.npu() if group_index is not None else None
        bias_npu = bias.npu() if bias is not None else None
        if quant_offset is not None:
            quant_offset = quant_offset.npu()
        y_npu, scale_npu = torch_npu.npu_dequant_swiglu_quant(
            x.npu(),
            weight_scale=weight_scale.npu(),
            activation_scale=activate_scale.npu(),
            bias=bias_npu,
            quant_scale=quant_scale.npu(),
            quant_offset=quant_offset,
            group_index=group_index_npu,
            activate_left=True,
            quant_mode=quant_mode,
            swiglu_mode=swiglu_mode,
            clamp_limit=7.0,
            glu_alpha=1.702,
            glu_bias=1.0,
            )

        self.assertRtolEqual(y_cpu.numpy(), y_npu.cpu().numpy())
        self.assertRtolEqual(scale_cpu.numpy(), scale_npu.cpu().numpy())

    @unittest.skip("Skip until CANN is updated to 8.3.RC1 to support aclnnDequantSwigluQuantV2")
    @SupportedDevices(["Ascend910B"])
    def test_npu_dequant_swiglu_quant_3(self, device="npu"):
        swiglu_mode = 0
        bias = None
        quant_offset = None
        x_shape = [4608, 2048]
        x = torch.randint(-10, 10, x_shape, dtype=torch.int32)
        weight_scale = torch.randn(x_shape[1], dtype=torch.float32)
        activate_scale = torch.randn((x_shape[0], 1), dtype=torch.float32)
        quant_scale = torch.randn((1, x_shape[1] // 2), dtype=torch.float32)
        group_index = torch.tensor([x.shape[0]])
        quant_mode = 0

        if quant_mode == 0:
            quant_offset = torch.randn((1, x_shape[1] // 2), dtype=torch.float32)
        y_cpu, _ = self.golden_dequant_swiglu_quant_torch(
            x,
            weight_scale,
            activate_scale,
            bias,
            quant_scale,
            quant_offset,
            group_index,
            activate_left=True,
            quant_mode=quant_mode,
            swiglu_mode=swiglu_mode,
            clamp_limit=7.0,
            glu_alpha=1.702,
            glu_bias=1.0,
            )

        group_index_npu = group_index.npu() if group_index is not None else None
        bias_npu = bias.npu() if bias is not None else None
        if quant_offset is not None:
            quant_offset = quant_offset.npu()
        y_npu, scale_npu = torch_npu.npu_dequant_swiglu_quant(
            x.npu(),
            weight_scale=weight_scale.npu(),
            activation_scale=activate_scale.npu(),
            bias=bias_npu,
            quant_scale=quant_scale.npu(),
            quant_offset=quant_offset,
            group_index=group_index_npu,
            activate_left=True,
            quant_mode=quant_mode,
            swiglu_mode=swiglu_mode,
            clamp_limit=7.0,
            glu_alpha=1.702,
            glu_bias=1.0,
            )
        self.assertRtolEqual(y_cpu.numpy(), y_npu.cpu().numpy())

if __name__ == "__main__":
    run_tests()
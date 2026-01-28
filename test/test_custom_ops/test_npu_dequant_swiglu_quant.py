import os
import shutil
import unittest
import numpy
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

MAX_VALUE_WITH_INT8 = 127
MIN_VALUE_WITH_INT8 = -128
MAX_VALUE_WITH_FLOAT8E5M2 = 57344
MAX_VALUE_WITH_FLOAT8E4M3 = 448
MAX_VALUE_WITH_FLOAT4E2M1 = 6
MAX_VALUE_WITH_FLOAT4E1M2 = 1.75


class TestNPUDequantSwigluQuant(TestCase):
    def numpy_float8_e5m2(self):
        try:
            from ml_dtypes import float8_e5m2
            return float8_e5m2
        except ModuleNotFoundError:
            raise RuntimeError("ml_dtypes is needed to support float8_e5m2 dtype!!! "
                               "Please install with `pip3 install ml-dtypes`")

    def numpy_float8_e4m3fn(self):
        try:
            from ml_dtypes import float8_e4m3fn
            return float8_e4m3fn
        except ModuleNotFoundError:
            raise RuntimeError("ml_dtypes is needed to support float8_e4m3fn dtype!!! "
                               "Please install with `pip3 install ml-dtypes`")

    def numpy_float4_e2m1(self):
        try:
            from en_dtypes import float4_e2m1
            return float4_e2m1
        except ModuleNotFoundError:
            raise RuntimeError("en_dtypes is needed to support float4_e2m1 dtype!!! "
                               "Please install with `pip3 install en-dtypes`")

    def numpy_float4_e1m2(self):
        try:
            from en_dtypes import float4_e1m2
            return float4_e1m2
        except ModuleNotFoundError:
            raise RuntimeError("en_dtypes is needed to support float4_e1m2 dtype!!! "
                               "Please install with `pip3 install en-dtypes`")

    def get_max_num(self, dst_type):
        if dst_type == 1:
            return MAX_VALUE_WITH_INT8
        if dst_type == 291:
            return MAX_VALUE_WITH_FLOAT8E5M2
        if dst_type == 292:
            return MAX_VALUE_WITH_FLOAT8E4M3
        if dst_type == 296:
            return MAX_VALUE_WITH_FLOAT4E2M1
        if dst_type == 297:
            return MAX_VALUE_WITH_FLOAT4E1M2

    def transform_output(self, dst_type, round_mode, input):
        dst_type_map = {1: numpy.int8, 291: self.numpy_float8_e5m2(), 292: self.numpy_float8_e4m3fn(),
                        296: self.numpy_float4_e2m1(), 297: self.numpy_float4_e1m2()}

        round_mode_map = {0: numpy.rint, 1: numpy.round, 2: numpy.floor, 3: numpy.ceil, 4: numpy.trunc}

        if dst_type == 1:
            input = round_mode_map[round_mode](input)
            tmp = input.to(torch.int8).numpy()
            return tmp
        elif dst_type == 291 or dst_type == 292:
            tmp = input.numpy().astype(dst_type_map[dst_type])
            return tmp

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
        dst_type=1,
        round_mode=0,
        activate_dim=-1,
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

        if quant_offset is None:
            quant_offset = torch.tensor([0], dtype=torch.float32)

        if group_index is None:
            group_index = numpy.array([x.shape[0]])

        if quant_mode == 1:
            if quant_scale is not None and len(quant_scale.shape) == 1:
                quant_scale = quant_scale.reshape(1, -1)

        if quant_mode == 1:
            quant_mode = "dynamic"
        elif quant_mode == 0:
            quant_mode = "static"

        res_y = torch.zeros([x.shape[0], x.shape[1] // 2], dtype=torch.float32)
        res_scale = torch.zeros([x.shape[0]], dtype=torch.float32)

        offset = 0
        for g_idx in range(group_index.shape[0]):
            groupIdx = group_index[g_idx]
            x_tensor = x[offset: (offset + groupIdx)]
            if "int32" in str(x_dtype):
                if bias is not None and bias.dtype is torch.int32:
                    x_tensor = torch.add(x_tensor, bias[g_idx])

                x_tensor = x_tensor.to(torch.float32)
                res = torch.mul(x_tensor, weight_scale[g_idx].to(torch.float32))

                if activate_scale is not None:
                    res = torch.mul(res, activate_scale[offset: (offset + groupIdx)].to(torch.float32))

                if bias is not None and bias.dtype is not torch.int32:
                    res = torch.add(res, bias.to(torch.float32))
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
            else:
                if quant_mode == "static":
                    output = output + quant_offset
                    scale_out = torch.tensor(0.0)
                else:
                    absd = torch.abs(output)
                    max_values = torch.amax(absd, dim=-1)
                    max_value = self.get_max_num(dst_type)
                    scale_out = max_values / max_value
                    max_values = max_value / max_values
                    output = output * max_values.unsqueeze(1)

            if dst_type == 1:
                output = torch.clamp(output, min=MIN_VALUE_WITH_INT8, max=MAX_VALUE_WITH_INT8)
            else:
                output = torch.clamp(output, min=self.get_max_num(dst_type) * -1, max=self.get_max_num(dst_type))

            res_y[offset: (offset + groupIdx)] = output
            res_scale[offset: (offset + groupIdx)] = scale_out
            offset = offset + groupIdx

        return self.transform_output(dst_type, round_mode, res_y), res_scale

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

    @SupportedDevices(["Ascend910_95", "Ascend950"])
    def test_npu_dequant_swiglu_quant_4(self, device="npu"):
        x_shape = [8, 4]
        x = torch.randint(-10, 10, x_shape, dtype=torch.int32)
        weight_scale = torch.randn([4], dtype=torch.float32)
        activate_scale = None
        bias = torch.randn([4], dtype=torch.float32)
        quant_scale = None
        quant_offset = None
        group_index = None

        activate_left = False
        quant_mode = 1
        dst_type = 291
        round_mode = 0
        activate_dim = -1

        y_cpu, _ = self.golden_dequant_swiglu_quant_torch(
            x,
            weight_scale,
            activate_scale,
            bias,
            quant_scale,
            quant_offset,
            group_index,
            activate_left,
            quant_mode,
            dst_type,
            round_mode,
            activate_dim,
            swiglu_mode=0,
            clamp_limit=7.0,
            glu_alpha=1.702,
            glu_bias=1.0)

        y_npu, _ = torch_npu.npu_dequant_swiglu_quant(
            x.npu(),
            weight_scale=weight_scale.npu(),
            activation_scale=activate_scale,
            bias=bias.npu(),
            quant_scale=quant_scale,
            quant_offset=quant_offset,
            group_index=group_index,
            activate_left=activate_left,
            quant_mode=quant_mode,
            dst_type=dst_type,
            round_mode=round_mode,
            activate_dim=activate_dim,
            swiglu_mode=0,
            clamp_limit=7.0,
            glu_alpha=1.702,
            glu_bias=1.0)
        self.assertRtolEqual(y_cpu.astype(numpy.float32), y_npu.to(torch.float32).cpu().numpy())

if __name__ == "__main__":
    run_tests()

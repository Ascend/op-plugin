import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

torch.npu.config.allow_internal_format = False


class TestNPUGeluQuant(TestCase):
    
    def get_golden(self, input_self_tensor, input_scale_tensor=None, input_offset_tensor=None, approximate="none", quant_mode='static'):
        output = torch.nn.functional.gelu(input_self_tensor, approximate=approximate)
        if quant_mode == 'static':
            result = self.quantize(input_self_tensor, input_scale_tensor, input_offset_tensor)
        else:
            result = self.dynamic_quant(input_self_tensor, input_scale_tensor)
        return result
    
    def quantize(self, x, input_scale_tensor, input_offset_tensor=None):
        input_scale_tensor = input_scale_tensor.to(torch.float32)
        input_offset_tensor = input_offset_tensor.to(torch.float32) if input_offset_tensor is not None else None
        scale_rst = x * input_scale_tensor
        if input_offset_tensor is not None:
            scale_rst = scale_rst + input_offset_tensor
        round_data = torch.round(scale_rst, decimals=0)
        round_data = torch.clip(round_data, -128, 127)
        round_data = round_data.to(torch.int8)
        return round_data, None
    
    def dynamic_quant(self, x, smooth_scale_tensor=None):
        # symmetrical dynamic quant only, quant axis is -1 (per-channel scenario)
        scale_max = 127.0

        if smooth_scale_tensor is not None:
            smooth_scale_tensor = smooth_scale_tensor.to(torch.float32)
        else:
            smooth_scale_tensor = 1
        x = x.to(torch.float32)
        input_mul = x * smooth_scale_tensor
        input_abs = torch.abs(input_mul)
        input_max = torch.max(input_abs, dim=-1, keepdim=True).values
        scale = input_max * (1.0 / scale_max)
        input_scaled = input_mul / scale
        round_data = torch.round(input_scaled, decimals=0)
        round_data = round_data.to(torch.int8)
        return round_data, scale.squeeze(-1)

    @SupportedDevices(['Ascend910_95', 'Ascend950'])
    def test_npu_gelu_quant_static(self):
        # dst_type = int8 only
        shape = [100, 400]

        input_self_tensor = torch.rand(shape, dtype=torch.float16).npu() * 256
        input_scale_tensor = torch.rand(shape[-1:], dtype=torch.float16).npu()
        input_offset_tensor = torch.rand(shape[-1:], dtype=torch.float16).npu()
        torch.npu.synchronize()
        dst_type = torch.int8
        quant_mode = 'static'
        approximate = "tanh"
        round_mode = 'rint'
        output, out_scale = torch.ops.npu.npu_gelu_quant(input_self_tensor, 
                                                         input_scale=input_scale_tensor,
                                                         input_offset=input_offset_tensor,
                                                         approximate=approximate,
                                                         quant_mode=quant_mode,
                                                         dst_type=dst_type,
                                                         round_mode=round_mode)
        torch.npu.synchronize()
        golden, golden_out_scale = self.get_golden(input_self_tensor.cpu(), input_scale_tensor.cpu(), input_offset_tensor.cpu(), approximate)
        self.assertRtolEqual(output.type(torch.int8), golden.type(torch.int8), prec=1.01)
        if out_scale is not None:
            self.assertRtolEqual(out_scale.type(torch.float32), golden_out_scale.type(torch.float32))

    
    @SupportedDevices(['Ascend910_95', 'Ascend950'])
    def test_npu_gelu_quant_dynamic(self):
        # dst_type = int8 only
        shape = [100, 400]

        input_self_tensor = torch.rand(shape, dtype=torch.float16).npu() * 256
        input_scale_tensor = torch.rand(shape[-1:], dtype=torch.float16).npu()
        input_offset_tensor = torch.rand(shape[-1:], dtype=torch.float16).npu()
        torch.npu.synchronize()
        dst_type = torch.int8
        quant_mode = 'dynamic'
        approximate = "tanh"
        round_mode = 'rint'
        output, out_scale = torch.ops.npu.npu_gelu_quant(input_self_tensor, 
                                                         input_scale=input_scale_tensor,
                                                         input_offset=input_offset_tensor,
                                                         approximate=approximate,
                                                         quant_mode=quant_mode,
                                                         dst_type=dst_type,
                                                         round_mode=round_mode)
        torch.npu.synchronize()
        golden, golden_out_scale = self.get_golden(input_self_tensor.cpu(), input_scale_tensor.cpu(), input_offset_tensor.cpu(), approximate, quant_mode=quant_mode)
        self.assertRtolEqual(output.type(torch.int8), golden.type(torch.int8), prec=1.01)
        if out_scale is not None:
            self.assertRtolEqual(out_scale.type(torch.float32), golden_out_scale.type(torch.float32))

if __name__ == "__main__":
    run_tests()

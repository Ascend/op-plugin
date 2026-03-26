import unittest
import math
import numpy as np
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices
from torch.testing import assert_close

class TestSwigluMxQuant(TestCase):
    def npu_op_exec(self, x, group_index=None, activate_dim=-1, activate_left=False, swiglu_mode=0,
                    clamp_limit=7, glu_alpha=1.702, glu_bias=1, group_mode=0, axis=-1,
                    dst_type=torch_npu.float4_e2m1fn_x2, round_mode="rint", scale_alg=0, max_dtype_value=0):
        return torch_npu.npu_swiglu_mx_quant(x, group_index=group_index, activate_dim=activate_dim,
                                             activate_left=activate_left, swiglu_mode=swiglu_mode,
                                             clamp_limit=clamp_limit, glu_alpha=glu_alpha,
                                             glu_bias=glu_bias, group_mode=group_mode,
                                             axis=axis, dst_type=dst_type, round_mode=round_mode,
                                             scale_alg=scale_alg, max_dtype_value=max_dtype_value)

    def golden_op_exec(self, input_tensor):
        if torch.all(torch.eq(input_tensor, 0.0)) and input_tensor.shape == torch.Size([1, 2, 2]):
            device = input_tensor.device
            y = torch.tensor([[119, 119], [119, 119], [119, 119], [119, 119]], dtype=torch_npu.float4_e2m1fn_x2,
                             device=device)
            mxscale = torch.tensor([[[124, 0]], [[124, 0]], [[124, 0]], [[124, 0]]], dtype=torch.uint8,
                                   device=device)

            return y, mxscale

    def generate_input(self, input, value, dtype="float16"):
        if dtype == "float16":
            data_type = torch.float16
        elif dtype == "bfloat16":
            data_type = torch.bfloat16
        input_tensor = torch.full(input, value, dtype=data_type)
        return input_tensor

    @SupportedDevices(['Ascend950'])
    def test_npu_add_rms_norm_quant_float8_e5m2_with_rstd(self, device="npu"):
        x = self.generate_input(input=[4, 8], value=1, dtype="bfloat16")
        x = x.to(device).requires_grad_(True)

        golden_output = self.golden_op_exec(x.clone().detach())
        npu_output = self.npu_op_exec(x)
        y = npu_output[0].view([4, 2]).view(torch.uint8)
        mxscale = npu_output[2].view([4, 1, 2]).to(torch.uint8)

        assert torch.all(y == golden_output[0].view(torch.uint8))
        assert torch.all(mxscale == golden_output[1].view(torch.uint8))
        assert_close(golden_output[1], mxscale, atol=0.01, rtol=0.001)


if __name__ == "__main__":
    run_tests()
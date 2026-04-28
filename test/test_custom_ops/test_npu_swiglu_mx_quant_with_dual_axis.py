import unittest
import math
import numpy as np
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices
from torch.testing import assert_close

class TestSwigluMxQuantWithDualAxis(TestCase):
    def npu_op_exec(self, x, group_index=None, activate_left=True, round_mode="rint",
                    scale_alg=1, dst_type=torch_npu.float8_e4m3fn, dst_type_max=0):
        return torch_npu.npu_swiglu_mx_quant_with_dual_axis(x, group_index=group_index,
                                                             activate_left=activate_left,
                                                             round_mode=round_mode,
                                                             scale_alg=scale_alg,
                                                             dst_type=dst_type,
                                                             dst_type_max=dst_type_max)

    def golden_op_exec(self, input_tensor):
        device = input_tensor.device
        y1 = torch.tensor([[128, 128, 128, 128], [128, 128, 128, 128], [128, 128, 128, 128], [128, 128, 128, 128]],
                          dtype=torch.uint8)
        mxscale1 = torch.tensor([[[118, 0]], [[118, 0]], [[118, 0]], [[118, 0]]], dtype=torch.uint8)
        y2 = torch.tensor([[128, 128, 128, 128], [128, 128, 128, 128], [128, 128, 128, 128], [128, 128, 128, 128]],
                          dtype=torch.uint8)
        mxscale2 = torch.tensor([[[118, 0], [118, 0], [118, 0], [118, 0]], [[118, 0], [118, 0], [118, 0], [118, 0]]],
                                dtype=torch.uint8)
        return y1, mxscale1, y2, mxscale2

    def generate_input(self, input, value, dtype="float16"):
        if dtype == "float16":
            data_type = torch.float16
        elif dtype == "bfloat16":
            data_type = torch.bfloat16
        input_tensor = torch.full(input, value, dtype=data_type)
        return input_tensor

    @SupportedDevices(['Ascend950'])
    def test_npu_swiglu_mx_quant_with_dual_axis_float8_e4m3fn(self, device="npu"):
        x = self.generate_input(input=[4, 8], value=1, dtype="bfloat16")
        x = x.to(device).requires_grad_(True)
        group_index = torch.tensor([2, 4], dtype=torch.int64).npu()

        golden_output = self.golden_op_exec(x.clone().detach())
        npu_output = self.npu_op_exec(x, group_index)

        y1 = npu_output[0].cpu().view([4, 4]).to(torch.uint8)
        mxscale1 = npu_output[1].cpu().view([4, 1, 2]).to(torch.uint8)
        y2 = npu_output[2].cpu().view([4, 4]).to(torch.uint8)
        mxscale2 = npu_output[3].cpu().view([2, 4, 2]).to(torch.uint8)

        assert torch.all(y1 == golden_output[0].view(torch.uint8))
        assert torch.all(mxscale1 == golden_output[1].view(torch.uint8))
        assert torch.all(y2 == golden_output[2].view(torch.uint8))
        assert torch.all(mxscale2 == golden_output[3].view(torch.uint8))


if __name__ == "__main__":
    run_tests()

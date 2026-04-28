import math
import unittest
import copy
import struct
from struct import pack, unpack
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices
from torch.testing import assert_close


class TestGroupedDynamicMxQuant(TestCase):
    def custom_op_exec(self, input_tensor, group_index_tensor, round_mode="rint", dst_type=23, blocksize=32, scale_alg=0):
        return torch_npu.npu_grouped_dynamic_mx_quant(
            input_tensor, group_index_tensor, round_mode=round_mode, dst_type=dst_type, blocksize=32, scale_alg=scale_alg)

    def supported_op_exec(self, input_tensor):
        if torch.all(torch.eq(input_tensor, 0.0)) and input_tensor.shape == torch.Size([1, 2]):
            device = input_tensor.device
            y = torch.tensor([[0, 0]], dtype=torch.float8_e5m2, device=device)
            mxscale = torch.tensor([[[5.87747e-39, 5.87747e-39], [5.87747e-39, 5.87747e-39]]], dtype=torch.uint8, device=device)
            return y, mxscale

    def generate_input(self, input, group_index, dtype="float16"):
        input_data_type = torch.float16 if dtype == "float16" else torch.bfloat16
        input_value = 0.0
        input_tensor = torch.full(input, input_value, dtype=input_data_type)
        group_index_data_type = torch.int32
        group_index_value = 1
        group_index_tensor = torch.full(group_index, group_index_value, dtype=group_index_data_type)
        return input_tensor, group_index_tensor
    
    @SupportedDevices(['Ascend950'])
    def test_npu_grouped_dynamic_mx_quant_(self, device="npu"):
        input_tensor, group_index_tensor = self.generate_input(input=[1, 2], group_index=[1], dtype="float16")
        input_tensor = input_tensor.to(device)
        group_index_tensor = group_index_tensor.to(device)
        supported_output = self.supported_op_exec(input_tensor.clone())
        custom_output = self.custom_op_exec(input_tensor.clone(), group_index_tensor.clone(), "rint", 23, 32, 0)
        y = custom_output[0].view([1, 2]).view(torch.uint8)
        mxscale = custom_output[1].view([1, 2, 2]).view(torch.uint8)

        assert torch.all(y == supported_output[0].view(torch.uint8))
        assert_close(supported_output[1], mxscale, atol=0.01, rtol=0.001)
        
if __name__ == "__main__":
    run_tests()
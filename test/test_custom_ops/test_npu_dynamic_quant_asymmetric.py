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


class TestDynamicQuant(TestCase):
    def supported_op_exec(self, input_tensor, smooth_scales=None, group_index=None, dst_type=torch.int8):
        input_tensor = input_tensor.float()
        if group_index is not None:
            input_shape = input_tensor.shape
            row_num = input_tensor.numel() // input_shape[-1]
            input_tensor = input_tensor.view(row_num, input_shape[-1])
            start = 0
            group_num = group_index.numel()
            for index in range(group_num):
                end = group_index[index]
                if end <= start:
                    start = end
                    continue
                smooth_scales_row = smooth_scales[index]
                input_tensor[start:end] = input_tensor[start:end] * smooth_scales_row
                start = end
            input_tensor = input_tensor.view(input_shape)
        elif smooth_scales is not None:
            input_tensor = input_tensor * smooth_scales
        input_max = input_tensor.max(dim=-1, keepdim=True)[0]
        input_min = input_tensor.min(dim=-1, keepdim=True)[0]
        if dst_type == torch.int8:    
            scale = (input_max - input_min) / 255
            offset = 127 - input_max / scale
        else:
            scale = (input_max - input_min) / 15
            offset = 7 - input_max / scale
        try:
            input_tensor = input_tensor / scale
        except ZeroDivisionError as err:
            raise err
        input_tensor = input_tensor + offset
        output = input_tensor.round()
        return [output.to(torch.int8), scale.squeeze(-1).float(), offset.squeeze(-1).float()]

    def custom_op_exec(self, input_tensor, smooth_scales=None, group_index=None, dst_type=torch.int8):
        return torch_npu.npu_dynamic_quant_asymmetric(input_tensor, smooth_scales=smooth_scales, group_index=group_index, dst_type=dst_type)

    def generate_input(self, input_shape, dtype="float16", use_smooth=False, group_num=1):
        date_type = torch.float16 if dtype == "float16" else torch.bfloat16
        input_tensor = torch.randn(input_shape, dtype=date_type)
        group_index = None
        smooth_scales = None
        if group_num > 1:
            smooth_scales = torch.randn(group_num, input_shape[-1], dtype=date_type)
            row_num = input_tensor.numel() // input_tensor.shape[-1]
            group_index_list = []
            for _ in range(group_num):
                group_index_list.append(np.random.randint(0, row_num))
            group_index_list = sorted(group_index_list)
            group_index_list[-1] = row_num
            group_index = torch.tensor(group_index_list).to(torch.int32)
        else:
            smooth_scales = torch.randn(input_shape[-1], dtype=date_type)
        return input_tensor, smooth_scales, group_index

    def convert_int4_to_int8(self, x):
        x_uint8 = x.view(torch.uint8).view(-1, 1)
        x_uint8_left = ((x_uint8 & 0xF0).view(torch.int8) >> 4)
        x_uint8_right = ((x_uint8 & 0x0F) << 4).view(torch.int8) >> 4
        x_int4 = torch.cat([x_uint8_right, x_uint8_left], dim=-1).contiguous()
        return x_int4
    
    @SupportedDevices(['Ascend910B'])
    def test_npu_dynamic_quant_asymmetric_fp16_input(self, device="npu"):
        input_tensor, _, _ = self.generate_input(input_shape=[2, 32, 256],
                                                                dtype="float16",
                                                                use_smooth=False,
                                                                group_num=1)
        input_tensor = input_tensor.to(device)
        supported_output = self.supported_op_exec(input_tensor.clone())
        custom_output = self.custom_op_exec(input_tensor.clone())
        assert_close(supported_output[0], custom_output[0], atol=1.0, rtol=0.0)
        assert_close(supported_output[1], custom_output[1], atol=0.0, rtol=0.0001)
        assert_close(supported_output[2], custom_output[2], atol=0.0, rtol=0.0001)

    @SupportedDevices(['Ascend910B'])
    def test_npu_dynamic_quant_asymmetric_bf16_input(self, device="npu"):
        input_tensor, _, _ = self.generate_input(input_shape=[2, 32, 256],
                                                                dtype="bfloat16",
                                                                use_smooth=False,
                                                                group_num=1)
        input_tensor = input_tensor.to(device)
        supported_output = self.supported_op_exec(input_tensor.clone())
        custom_output = self.custom_op_exec(input_tensor.clone())
        assert_close(supported_output[0], custom_output[0], atol=1.0, rtol=0.0)
        assert_close(supported_output[1], custom_output[1], atol=0.0, rtol=0.0001)
        assert_close(supported_output[2], custom_output[2], atol=0.0, rtol=0.0001)

    @SupportedDevices(['Ascend910B'])
    def test_npu_dynamic_quant_asymmetric_fp16_input_smooth_group(self, device="npu"):
        input_tensor, smooth_scales, group_index = self.generate_input(input_shape=[2, 32, 256],
                                                                dtype="float16",
                                                                use_smooth=True,
                                                                group_num=64)
        input_tensor, smooth_scales, group_index = input_tensor.to(device), smooth_scales.to(device), group_index.to(device)
        supported_output = self.supported_op_exec(input_tensor.clone(), smooth_scales.clone(), group_index.clone())
        custom_output = self.custom_op_exec(input_tensor.clone(), smooth_scales.clone(), group_index.clone(), torch.int8)
        assert_close(supported_output[0], custom_output[0], atol=1.0, rtol=0.0)
        assert_close(supported_output[1], custom_output[1], atol=0.0, rtol=0.0001)
        assert_close(supported_output[2], custom_output[2], atol=0.0, rtol=0.0001)

    @SupportedDevices(['Ascend910B'])
    def test_npu_dynamic_quant_asymmetric_bfp16_input_smooth_group(self, device="npu"):
        input_tensor, smooth_scales, group_index = self.generate_input(input_shape=[2, 32, 256],
                                                                dtype="bfloat16",
                                                                use_smooth=True,
                                                                group_num=64)
        input_tensor, smooth_scales, group_index = input_tensor.to(device), smooth_scales.to(device), group_index.to(device)
        supported_output = self.supported_op_exec(input_tensor.clone(), smooth_scales.clone(), group_index.clone())
        custom_output = self.custom_op_exec(input_tensor.clone(), smooth_scales.clone(), group_index.clone(), torch.int8)
        assert_close(supported_output[0], custom_output[0], atol=1.0, rtol=0.0)
        assert_close(supported_output[1], custom_output[1], atol=0.0, rtol=0.0001)  
        assert_close(supported_output[2], custom_output[2], atol=0.0, rtol=0.0001) 

    @SupportedDevices(['Ascend910B'])
    def test_npu_dynamic_quant_asymmetric_int4_fp16_input(self, device="npu"):
        input_tensor, _, _ = self.generate_input(input_shape=[2, 32, 256],
                                                                dtype="float16",
                                                                use_smooth=False,
                                                                group_num=1)
                                                                
        input_tensor = input_tensor.to(device)
        supported_output = self.supported_op_exec(input_tensor.clone(), dst_type=torch.quint4x2)
        custom_output = self.custom_op_exec(input_tensor.clone(), dst_type=torch.quint4x2)
        y = self.convert_int4_to_int8(custom_output[0]).view([2, 32, 256])
        assert_close(supported_output[0], y, atol=1.0, rtol=0.0)
        assert_close(supported_output[1], custom_output[1], atol=0.0, rtol=0.0001)
        assert_close(supported_output[2], custom_output[2], atol=0.0, rtol=0.0001)

    @SupportedDevices(['Ascend910B'])
    def test_npu_dynamic_quant_asymmetric_int4_bf16_input(self, device="npu"):
        input_tensor, _, _ = self.generate_input(input_shape=[2, 32, 256],
                                                                dtype="bfloat16",
                                                                use_smooth=False,
                                                                group_num=1)
        input_tensor = input_tensor.to(device)
        supported_output = self.supported_op_exec(input_tensor.clone(), dst_type=torch.quint4x2)
        custom_output = self.custom_op_exec(input_tensor.clone(), dst_type=torch.quint4x2)
        y = self.convert_int4_to_int8(custom_output[0]).view([2, 32, 256])
        assert_close(supported_output[0], y, atol=1.0, rtol=0.0)
        assert_close(supported_output[1], custom_output[1], atol=0.0, rtol=0.0001) 
        assert_close(supported_output[2], custom_output[2], atol=0.0, rtol=0.0001)

    @SupportedDevices(['Ascend910B'])
    def test_npu_dynamic_quant_asymmetric_int4_fp16_input_smooth_group(self, device="npu"):
        input_tensor, smooth_scales, group_index = self.generate_input(input_shape=[2, 32, 256],
                                                                dtype="float16",
                                                                use_smooth=True,
                                                                group_num=64)
        input_tensor, smooth_scales, group_index = input_tensor.to(device), smooth_scales.to(device), group_index.to(device)
        supported_output = self.supported_op_exec(input_tensor.clone(), smooth_scales.clone(), group_index.clone(), dst_type=torch.quint4x2)
        custom_output = self.custom_op_exec(input_tensor.clone(), smooth_scales.clone(), group_index.clone(), torch.quint4x2)
        y = self.convert_int4_to_int8(custom_output[0]).view([2, 32, 256])
        assert_close(supported_output[0], y, atol=1.0, rtol=0.0)
        assert_close(supported_output[1], custom_output[1], atol=0.0, rtol=0.0001)
        assert_close(supported_output[2], custom_output[2], atol=0.0, rtol=0.0001)

    @SupportedDevices(['Ascend910B'])
    def test_npu_dynamic_quant_asymmetric_int4_bf16_input_smooth_group(self, device="npu"):
        input_tensor, smooth_scales, group_index = self.generate_input(input_shape=[2, 32, 256],
                                                                dtype="bfloat16",
                                                                use_smooth=True,
                                                                group_num=64)
        input_tensor, smooth_scales, group_index = input_tensor.to(device), smooth_scales.to(device), group_index.to(device)
        supported_output = self.supported_op_exec(input_tensor.clone(), smooth_scales.clone(), group_index.clone(), dst_type=torch.quint4x2)
        custom_output = self.custom_op_exec(input_tensor.clone(), smooth_scales.clone(), group_index.clone(), torch.quint4x2)
        y = self.convert_int4_to_int8(custom_output[0]).view([2, 32, 256])
        assert_close(supported_output[0], y, atol=1.0, rtol=0.0)
        assert_close(supported_output[1], custom_output[1], atol=0.0, rtol=0.0001)
        assert_close(supported_output[2], custom_output[2], atol=0.0, rtol=0.0001)
        
if __name__ == "__main__":
    run_tests()
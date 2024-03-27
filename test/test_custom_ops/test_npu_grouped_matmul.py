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


class TestGroupedMatmul(TestCase):
    def deq_scale_generate(self, deq_scale_shape):
        fp32_deq_scale = np.random.uniform(low=0.01, high=0.03, size=deq_scale_shape).astype(np.float32)
        temp_quant_tensor = np.random.randint(1, 3, deq_scale_shape).astype(np.float32)
        temp_quant_tensor_api = copy.deepcopy(temp_quant_tensor).astype(np.int64)
        for i, _ in enumerate(temp_quant_tensor_api):
            temp_quant_tensor_api[i] = struct.unpack('!I', struct.pack('!f', temp_quant_tensor_api[i]))[0]
            temp_quant_tensor_api[i] = temp_quant_tensor_api[i] | np.int64(0x400000000000)
        uint64_deq_scale = np.frombuffer(temp_quant_tensor_api, np.int64)
        return fp32_deq_scale, uint64_deq_scale

    def single_matmul(self, x, w, *, b=None, scale=None, offset=None, antiquantScale=None, antiquantOffset=None):
        if antiquantScale is not None:
            w = torch.from_numpy(w).to(torch.float16)
            w = (w + antiquantOffset) * antiquantScale
        y = np.matmul(x, w)
        if b is not None:
            y += b
        if scale is not None:
            y = y * scale
            y = self.float32_to_int9(y)
            y = np.clip(y, -128, 127)
            y = y.astype(np.int8)
        return y

    def supported_op_exec(self, x, weight, *, bias=None, scale=None, offset=None, antiquantScale=None,
                          antiquantOffset=None, group_list=None, split_item=0):
        x_split = []
        if split_item == 0 or split_item == 2:
            x_split = x
        elif split_item == 1 or split_item == 3:
            x_offset = 0
            for item in group_list:
                x_split.append(x[0][x_offset:item, :])
                x_offset = item
        bias = bias or [None] * len(weight)
        output = []
        if antiquantScale is not None:
            output = [self.single_matmul(x_split[i], weight[i], b=bias[i], antiquantScale=antiquantScale[i],
                                         antiquantOffset=antiquantOffset[i]) for i in range(len(weight))]
        elif scale is not None:
            output = [torch.from_numpy(self.single_matmul(x_split[i], weight[i], b=bias[i],
                                       scale=scale[i])) for i in range(len(weight))]
        else:
            output = [self.single_matmul(x_split[i], weight[i], b=bias[i]) for i in range(len(weight))]
        if split_item == 2 or split_item == 3:
            output = [torch.cat(output, 0)]
        return output

    def float32_to_int9(self, fp32):
        int_value = (np.round(fp32.numpy())).astype(int)
        int9_value = np.clip(int_value, -256, 255)
        return int_value

    def custom_op_exec(self, x, weight, *, bias=None, scale=None, offset=None, antiquantScale=None,
                       antiquantOffset=None, group_list=None, split_item=0):
        return torch_npu.npu_grouped_matmul(x, weight, bias=bias, scale=scale, offset=offset,
                                            antiquant_scale=antiquantScale, antiquant_offset=antiquantOffset,
                                            group_list=group_list, split_item=split_item)

    @SupportedDevices(['Ascend910B'])
    def test_npu_grouped_matmul_0(self, device="npu"):
        torch.manual_seed(0)
        x1 = torch.normal(mean=0., std=0.1, size=(256, 256), dtype=torch.float16)
        x2 = torch.normal(mean=0., std=0.1, size=(1024, 256), dtype=torch.float16)
        x3 = torch.normal(mean=0., std=0.1, size=(512, 1024), dtype=torch.float16)
        x = [x1, x2, x3]
        weight1 = torch.normal(mean=0., std=0.1, size=(256, 256), dtype=torch.float16)
        weight2 = torch.normal(mean=0., std=0.1, size=(256, 1024), dtype=torch.float16)
        weight3 = torch.normal(mean=0., std=0.1, size=(1024, 128), dtype=torch.float16)
        weight = [weight1, weight2, weight3]
        bias1 = torch.normal(mean=0., std=0.1, size=(256,), dtype=torch.float16)
        bias2 = torch.normal(mean=0., std=0.1, size=(1024,), dtype=torch.float16)
        bias3 = torch.normal(mean=0., std=0.1, size=(128,), dtype=torch.float16)
        bias = [bias1, bias2, bias3]
        group_list = None
        split_item = 0

        x_clone = []
        weight_clone = []
        bias_clone = []
        for x_i in x:
            x_clone.append(x_i.clone().npu())
        for weight_i in weight:
            weight_clone.append(weight_i.clone().npu())
        for bias_i in bias:
            bias_clone.append(bias_i.clone().npu())

        supported_output = self.supported_op_exec(x, weight, bias=bias, group_list=group_list, split_item=split_item)
        custom_output = self.custom_op_exec(x_clone, weight_clone, bias=bias_clone, group_list=group_list,
                                            split_item=split_item)

        self.assertRtolEqual(x[0], x_clone[0], 0.001)
        self.assertRtolEqual(x[1], x_clone[1], 0.001)
        self.assertRtolEqual(x[2], x_clone[2], 0.001)
        self.assertRtolEqual(supported_output[0], custom_output[0], 0.001)
        self.assertRtolEqual(supported_output[1], custom_output[1], 0.001)
        self.assertRtolEqual(supported_output[2], custom_output[2], 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_npu_grouped_matmul_0_multi_dim(self, device="npu"):
        torch.manual_seed(0)
        x1 = torch.normal(mean=0., std=0.1, size=(7, 1024, 96), dtype=torch.float16)
        x2 = torch.normal(mean=0., std=0.1, size=(7, 1024, 32), dtype=torch.float16)
        x = [x1, x2]
        weight1 = torch.normal(mean=0., std=0.1, size=(96, 5120), dtype=torch.float16)
        weight2 = torch.normal(mean=0., std=0.1, size=(32, 8192), dtype=torch.float16)
        weight = [weight1, weight2]
        bias1 = torch.normal(mean=0., std=0.1, size=(5120,), dtype=torch.float16)
        bias2 = torch.normal(mean=0., std=0.1, size=(8192,), dtype=torch.float16)
        bias = [bias1, bias2]
        group_list = None
        split_item = 0

        x_clone = []
        weight_clone = []
        bias_clone = []
        for x_i in x:
            x_clone.append(x_i.clone().npu())
        for weight_i in weight:
            weight_clone.append(weight_i.clone().npu())
        for bias_i in bias:
            bias_clone.append(bias_i.clone().npu())

        supported_output = self.supported_op_exec(x, weight, bias=bias, group_list=group_list, split_item=split_item)
        custom_output = self.custom_op_exec(x_clone, weight_clone, bias=bias_clone, group_list=group_list,
                                            split_item=split_item)

        self.assertRtolEqual(x[0], x_clone[0], 0.001)
        self.assertRtolEqual(x[1], x_clone[1], 0.001)
        self.assertRtolEqual(supported_output[0], custom_output[0], 0.001)
        self.assertRtolEqual(supported_output[1], custom_output[1], 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_npu_grouped_matmul_1(self, device="npu"):
        torch.manual_seed(0)
        x1 = torch.normal(mean=0., std=0.1, size=(1792, 1024), dtype=torch.float16)
        x = [x1]
        weight1 = torch.normal(mean=0., std=0.1, size=(1024, 256), dtype=torch.float16)
        weight2 = torch.normal(mean=0., std=0.1, size=(1024, 1024), dtype=torch.float16)
        weight3 = torch.normal(mean=0., std=0.1, size=(1024, 128), dtype=torch.float16)
        weight = [weight1, weight2, weight3]
        bias1 = torch.normal(mean=0., std=0.1, size=(256,), dtype=torch.float16)
        bias2 = torch.normal(mean=0., std=0.1, size=(1024,), dtype=torch.float16)
        bias3 = torch.normal(mean=0., std=0.1, size=(128,), dtype=torch.float16)
        bias = [bias1, bias2, bias3]
        group_list = [256, 1280, 1792]
        split_item = 1

        x_clone = []
        weight_clone = []
        bias_clone = []
        for x_i in x:
            x_clone.append(x_i.clone().npu())
        for weight_i in weight:
            weight_clone.append(weight_i.clone().npu())
        for bias_i in bias:
            bias_clone.append(bias_i.clone().npu())

        supported_output = self.supported_op_exec(x, weight, bias=bias, group_list=group_list, split_item=split_item)
        custom_output = self.custom_op_exec(x_clone, weight_clone, bias=bias_clone, group_list=group_list,
                                            split_item=split_item)

        self.assertRtolEqual(x[0], x_clone[0], 0.001)
        self.assertRtolEqual(supported_output[0], custom_output[0], 0.001)
        self.assertRtolEqual(supported_output[1], custom_output[1], 0.001)
        self.assertRtolEqual(supported_output[2], custom_output[2], 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_npu_grouped_matmul_2(self, device="npu"):
        torch.manual_seed(0)
        x1 = torch.normal(mean=0., std=0.1, size=(256, 256), dtype=torch.float16)
        x2 = torch.normal(mean=0., std=0.1, size=(1024, 256), dtype=torch.float16)
        x3 = torch.normal(mean=0., std=0.1, size=(512, 1024), dtype=torch.float16)
        x = [x1, x2, x3]
        weight1 = torch.normal(mean=0., std=0.1, size=(256, 256), dtype=torch.float16)
        weight2 = torch.normal(mean=0., std=0.1, size=(256, 256), dtype=torch.float16)
        weight3 = torch.normal(mean=0., std=0.1, size=(1024, 256), dtype=torch.float16)
        weight = [weight1, weight2, weight3]
        bias1 = torch.normal(mean=0., std=0.1, size=(256,), dtype=torch.float16)
        bias2 = torch.normal(mean=0., std=0.1, size=(256,), dtype=torch.float16)
        bias3 = torch.normal(mean=0., std=0.1, size=(256,), dtype=torch.float16)
        bias = [bias1, bias2, bias3]
        group_list = None
        split_item = 2

        x_clone = []
        weight_clone = []
        bias_clone = []
        for x_i in x:
            x_clone.append(x_i.clone().npu())
        for weight_i in weight:
            weight_clone.append(weight_i.clone().npu())
        for bias_i in bias:
            bias_clone.append(bias_i.clone().npu())

        supported_output = self.supported_op_exec(x, weight, bias=bias, group_list=group_list, split_item=split_item)
        custom_output = self.custom_op_exec(x_clone, weight_clone, bias=bias_clone, group_list=group_list,
                                            split_item=split_item)

        self.assertRtolEqual(x[0], x_clone[0], 0.001)
        self.assertRtolEqual(x[1], x_clone[1], 0.001)
        self.assertRtolEqual(x[2], x_clone[2], 0.001)
        self.assertRtolEqual(supported_output[0], custom_output[0], 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_npu_grouped_matmul_3(self, device="npu"):
        torch.manual_seed(0)
        x1 = torch.normal(mean=0., std=0.1, size=(1792, 1024), dtype=torch.float16)
        x = [x1]
        weight1 = torch.normal(mean=0., std=0.1, size=(1024, 256), dtype=torch.float16)
        weight2 = torch.normal(mean=0., std=0.1, size=(1024, 256), dtype=torch.float16)
        weight3 = torch.normal(mean=0., std=0.1, size=(1024, 256), dtype=torch.float16)
        weight = [weight1, weight2, weight3]
        bias1 = torch.normal(mean=0., std=0.1, size=(256,), dtype=torch.float16)
        bias2 = torch.normal(mean=0., std=0.1, size=(256,), dtype=torch.float16)
        bias3 = torch.normal(mean=0., std=0.1, size=(256,), dtype=torch.float16)
        bias = [bias1, bias2, bias3]
        group_list = [256, 1280, 1792]
        split_item = 3

        x_clone = []
        weight_clone = []
        bias_clone = []
        for x_i in x:
            x_clone.append(x_i.clone().npu())
        for weight_i in weight:
            weight_clone.append(weight_i.clone().npu())
        for bias_i in bias:
            bias_clone.append(bias_i.clone().npu())

        supported_output = self.supported_op_exec(x, weight, bias=bias, group_list=group_list, split_item=split_item)
        custom_output = self.custom_op_exec(x_clone, weight_clone, bias=bias_clone, group_list=group_list,
                                            split_item=split_item)

        self.assertRtolEqual(x[0], x_clone[0], 0.001)
        self.assertRtolEqual(supported_output[0], custom_output[0], 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_npu_grouped_matmul_antiquant_0(self, device="npu"):
        torch.manual_seed(0)
        x1 = torch.normal(mean=0., std=0.05, size=(256, 256), dtype=torch.float16)
        x2 = torch.normal(mean=0., std=0.05, size=(1024, 256), dtype=torch.float16)
        x3 = torch.normal(mean=0., std=0.05, size=(512, 1024), dtype=torch.float16)
        x = [x1, x2, x3]
        weight1 = np.random.randint(-9, 9, size=(256, 256), dtype=np.int8)
        weight2 = np.random.randint(-9, 9, size=(256, 1024), dtype=np.int8)
        weight3 = np.random.randint(-9, 9, size=(1024, 128), dtype=np.int8)
        weight = [weight1, weight2, weight3]
        bias1 = torch.normal(mean=0., std=0.05, size=(256,), dtype=torch.float16)
        bias2 = torch.normal(mean=0., std=0.05, size=(1024,), dtype=torch.float16)
        bias3 = torch.normal(mean=0., std=0.05, size=(128,), dtype=torch.float16)
        bias = [bias1, bias2, bias3]
        antiquantScale1 = torch.normal(mean=0., std=0.05, size=(256,), dtype=torch.float16)
        antiquantScale2 = torch.normal(mean=0., std=0.05, size=(1024,), dtype=torch.float16)
        antiquantScale3 = torch.normal(mean=0., std=0.05, size=(128,), dtype=torch.float16)
        antiquantScale = [antiquantScale1, antiquantScale2, antiquantScale3]
        antiquantOffset1 = torch.normal(mean=0., std=0.05, size=(256,), dtype=torch.float16)
        antiquantOffset2 = torch.normal(mean=0., std=0.05, size=(1024,), dtype=torch.float16)
        antiquantOffset3 = torch.normal(mean=0., std=0.05, size=(128,), dtype=torch.float16)
        antiquantOffset = [antiquantOffset1, antiquantOffset2, antiquantOffset3]
        group_list = None
        split_item = 0

        x_clone = []
        weight_clone = []
        bias_clone = []
        antiquantScale_clone = []
        antiquantOffset_clone = []
        for x_i in x:
            x_clone.append(x_i.clone().npu())
        for weight_i in weight:
            weight_clone.append(torch.from_numpy(weight_i).npu())
        for bias_i in bias:
            bias_clone.append(bias_i.clone().npu())
        for scale_i in antiquantScale:
            antiquantScale_clone.append(scale_i.clone().npu())
        for offset_i in antiquantOffset:
            antiquantOffset_clone.append(offset_i.clone().npu())

        supported_output = self.supported_op_exec(x, weight, bias=bias, antiquantScale=antiquantScale,
                                                  antiquantOffset=antiquantOffset, group_list=group_list,
                                                  split_item=split_item)
        custom_output = self.custom_op_exec(x_clone, weight_clone, bias=bias_clone,
                                            antiquantScale=antiquantScale_clone, antiquantOffset=antiquantOffset_clone,
                                            group_list=group_list, split_item=split_item)

        self.assertRtolEqual(x[0], x_clone[0], 0.001)
        self.assertRtolEqual(x[1], x_clone[1], 0.001)
        self.assertRtolEqual(x[2], x_clone[2], 0.001)
        self.assertRtolEqual(supported_output[0], custom_output[0], 0.001)
        self.assertRtolEqual(supported_output[1], custom_output[1], 0.001)
        self.assertRtolEqual(supported_output[2], custom_output[2], 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_npu_grouped_matmul_quant_3(self, device="npu"):
        torch.manual_seed(0)
        x1 = torch.randint(1, 5, size=(1792, 1024), dtype=torch.int8)
        x = [x1]
        weight1 = torch.randint(1, 5, size=(1024, 256), dtype=torch.int8)
        weight2 = torch.randint(1, 5, size=(1024, 256), dtype=torch.int8)
        weight3 = torch.randint(1, 5, size=(1024, 256), dtype=torch.int8)
        weight = [weight1, weight2, weight3]
        bias1 = torch.randint(1, 5, size=(256,), dtype=torch.int32)
        bias2 = torch.randint(1, 5, size=(256,), dtype=torch.int32)
        bias3 = torch.randint(1, 5, size=(256,), dtype=torch.int32)
        bias = [bias1, bias2, bias3]
        group_list = [256, 1280, 1792]
        split_item = 3

        x_clone = []
        weight_clone = []
        bias_clone = []
        scale_uint64 = []
        scale_fp32 = []

        for x_i in x:
            x_clone.append(x_i.clone().npu())
        for weight_i in weight:
            weight_clone.append(weight_i.clone().npu())
        for bias_i in bias:
            bias_clone.append(bias_i.clone().npu())
        for i in range(3):
            scale, scale_new = self.deq_scale_generate(256)
            scale_fp32.append(scale)
            scale_uint64.append(torch.from_numpy(scale_new).npu())

        supported_output = self.supported_op_exec(x, weight, bias=bias, scale=scale_fp32, group_list=group_list,
                                                  split_item=split_item)
        custom_output = self.custom_op_exec(x_clone, weight_clone, bias=bias_clone, scale=scale_uint64,
                                            group_list=group_list, split_item=split_item)

        self.assertRtolEqual(x[0], x_clone[0], 1)
        self.assertRtolEqual(supported_output[0], custom_output[0], 1)

    @SupportedDevices(['Ascend910B'])
    def test_npu_grouped_matmul_0_empty_x_w(self, device="npu"):
        torch.manual_seed(0)
        x = []
        weight = []
        bias1 = torch.normal(mean=0., std=0.1, size=(256,), dtype=torch.float16)
        bias2 = torch.normal(mean=0., std=0.1, size=(1024,), dtype=torch.float16)
        bias3 = torch.normal(mean=0., std=0.1, size=(128,), dtype=torch.float16)
        bias = [bias1.npu(), bias2.npu(), bias3.npu()]
        group_list = None
        split_item = 0

        with self.assertRaises(RuntimeError):
            custom_output = self.custom_op_exec(x, weight, bias=bias, group_list=group_list, split_item=split_item)

if __name__ == "__main__":
    run_tests()

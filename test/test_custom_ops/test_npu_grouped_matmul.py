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
                          antiquantOffset=None, group_list=None, split_item=0, tuning_config=None):
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
                       antiquantOffset=None, group_list=None, split_item=0, group_type=None,
                       output_dtype=None, tuning_config=None):
        if group_type is not None:
            return torch_npu.npu_grouped_matmul(x, weight, bias=bias, scale=scale, offset=offset,
                                                antiquant_scale=antiquantScale, antiquant_offset=antiquantOffset,
                                                group_list=group_list, split_item=split_item,
                                                group_type=group_type, output_dtype=output_dtype)
        else:
            return torch_npu.npu_grouped_matmul(x, weight, bias=bias, scale=scale, offset=offset,
                                                antiquant_scale=antiquantScale, antiquant_offset=antiquantOffset,
                                                group_list=group_list, split_item=split_item, output_dtype=output_dtype)

    def get_group_list(self, m, g):
        step = (m - 0) // (g - 1)
        data = [i * step for i in range(g)]
        group_list = torch.tensor(data, dtype=torch.int64).npu()
        group_list[-1] = m
        return group_list
    
    @SupportedDevices(['Ascend910B'])
    def test_npu_grouped_matmul_0(self): # 多多多
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
                                            split_item=split_item, group_type=-1)

        self.assertRtolEqual(x[0], x_clone[0], 0.001)
        self.assertRtolEqual(x[1], x_clone[1], 0.001)
        self.assertRtolEqual(x[2], x_clone[2], 0.001)
        self.assertRtolEqual(supported_output[0], custom_output[0], 0.001)
        self.assertRtolEqual(supported_output[1], custom_output[1], 0.001)
        self.assertRtolEqual(supported_output[2], custom_output[2], 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_npu_grouped_matmul_0_multi_dim(self): # 多多多（x多维）
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
                                            split_item=split_item, group_type=-1)

        self.assertRtolEqual(x[0], x_clone[0], 0.001)
        self.assertRtolEqual(x[1], x_clone[1], 0.001)
        self.assertRtolEqual(supported_output[0], custom_output[0], 0.001)
        self.assertRtolEqual(supported_output[1], custom_output[1], 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_npu_grouped_matmul_1(self): # 单单单
        torch.manual_seed(0)
        x1 = torch.normal(mean=0., std=0.1, size=(1792, 1024), dtype=torch.float16)
        x = [x1]
        weight1 = torch.normal(mean=0., std=0.1, size=(2, 1024, 256), dtype=torch.float16)
        weight = [weight1[0], weight1[1]]
        group_list = torch.tensor([256, 1792]).npu()
        split_item = 3
        group_type = 0

        x_clone = []
        weight_clone = []
        bias_clone = []
        for x_i in x:
            x_clone.append(x_i.clone().npu())

        weight_clone.append(weight1.clone().npu())

        supported_output = self.supported_op_exec(x, weight, bias=None, group_list=group_list, split_item=split_item)
        custom_output = self.custom_op_exec(x_clone, weight_clone, bias=None, group_list=group_list,
                                            split_item=split_item, group_type=group_type)

        self.assertRtolEqual(supported_output[0], custom_output[0], 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_npu_grouped_matmul_2(self): # 多多单
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
        group_type = 0

        x_clone = []
        weight_clone = []
        bias_clone = []
        for x_i in x:
            x_clone.append(x_i.clone().npu())
        for weight_i in weight:
            weight_clone.append(weight_i.clone().npu())
        for bias_i in bias:
            bias_clone.append(bias_i.clone().npu())

        supported_output = self.supported_op_exec(x, weight, bias=bias, group_list=group_list, 
                                                  split_item=split_item)
        custom_output = self.custom_op_exec(x_clone, weight_clone, bias=bias_clone, group_list=group_list,
                                            split_item=split_item, group_type=group_type)

        self.assertRtolEqual(x[0], x_clone[0], 0.001)
        self.assertRtolEqual(x[1], x_clone[1], 0.001)
        self.assertRtolEqual(x[2], x_clone[2], 0.001)
        self.assertRtolEqual(supported_output[0], custom_output[0], 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_npu_grouped_matmul_3(self): # 单多单
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
        group_list = torch.tensor([256, 1280, 1792]).npu()
        split_item = 3
        group_type = 0

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
                                            split_item=split_item, group_type=group_type)

        self.assertRtolEqual(x[0], x_clone[0], 0.001)
        self.assertRtolEqual(supported_output[0], custom_output[0], 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_npu_grouped_matmul_4(self): # 单多单
        torch.manual_seed(0)
        x1 = torch.normal(mean=0., std=0.1, size=(1792, 256), dtype=torch.float16)
        x = [x1]
        weight1 = torch.normal(mean=0., std=0.1, size=(3, 256, 128), dtype=torch.float16)
        weight = [weight1[0], weight1[1], weight1[2]]
        bias1 = torch.normal(mean=0., std=0.1, size=(3, 128), dtype=torch.float16)
        bias = [bias1[0], bias1[1], bias1[2]]
        group_list = [256, 1280, 1792]
        split_item = 3

        x_clone = [x1.clone().npu()]
        weight_clone = [weight1.clone().npu()]
        bias_clone = [bias1.clone().npu()]
        group_list_npu = torch.tensor(group_list, dtype=torch.int64).npu()

        supported_output = self.supported_op_exec(x, weight, bias=bias, group_list=group_list, split_item=split_item)
        custom_output = self.custom_op_exec(x_clone, weight_clone, bias=bias_clone, group_list=group_list_npu,
                                            split_item=split_item, group_type=0)

        self.assertRtolEqual(supported_output[0], custom_output[0], 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_npu_grouped_matmul_5(self): # 单多单
        torch.manual_seed(0)
        x1 = torch.normal(mean=0., std=0.1, size=(1792, 256), dtype=torch.float16)
        x = [x1]
        weight1 = torch.normal(mean=0., std=0.1, size=(3, 256, 128), dtype=torch.float16)
        weight = [weight1[0], weight1[1], weight1[2]]
        bias1 = torch.normal(mean=0., std=0.1, size=(3, 128), dtype=torch.float16)
        bias = [bias1[0], bias1[1], bias1[2]]
        group_list = [256, 1280, 1792]
        split_item = 3

        x_clone = [x1.clone().npu()]
        weight_clone = [weight1.clone().npu()]
        bias_clone = [bias1.clone().npu()]
        group_list_npu = torch.tensor(group_list, dtype=torch.int64).npu()

        supported_output = self.supported_op_exec(x, weight, bias=bias, group_list=group_list, split_item=split_item, tuning_config = [1792])
        custom_output = self.custom_op_exec(x_clone, weight_clone, bias=bias_clone, group_list=group_list_npu,
                                            split_item=split_item, group_type=0, tuning_config = [1792])
        self.assertRtolEqual(supported_output[0], custom_output[0], 0.001)

    @unittest.skip("skip test_npu_grouped_matmul_antiquant_0 now")
    @SupportedDevices(['Ascend910B'])
    def test_npu_grouped_matmul_antiquant_0(self): # 伪量化 多多多
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
        group_type = -1

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
                                            group_list=group_list, split_item=split_item, group_type=group_type)

        self.assertRtolEqual(x[0], x_clone[0], 0.001)
        self.assertRtolEqual(x[1], x_clone[1], 0.001)
        self.assertRtolEqual(x[2], x_clone[2], 0.001)
        self.assertRtolEqual(supported_output[0], custom_output[0], 0.005)
        self.assertRtolEqual(supported_output[1], custom_output[1], 0.005)
        self.assertRtolEqual(supported_output[2], custom_output[2], 0.005)

    @SupportedDevices(['Ascend910B'])
    def test_npu_grouped_matmul_quant_3(self): # 量化 单多单
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
        group_list = torch.tensor([256, 1280, 1792]).npu()
        split_item = 3
        group_type = 0

        scale_output_dtype_list = [[torch.int64, None], [torch.float32, torch.float16], [torch.bfloat16, torch.bfloat16]]
        for item in scale_output_dtype_list:
            dtype = item[0]
            output_dtype = item[1]
            x_clone = []
            weight_clone = []
            bias_clone = []
            scale_npu = []
            scale_fp32 = []

            for x_i in x:
                x_clone.append(x_i.clone().npu())
            for weight_i in weight:
                weight_clone.append(weight_i.clone().npu())
            for bias_i in bias:
                bias_clone.append(bias_i.clone().npu())
            for _ in range(3):
                scale, _ = self.deq_scale_generate(256)
                scale_fp32.append(scale)
                scale_npu.append(torch.from_numpy(scale).npu().to(dtype))

            supported_output = self.supported_op_exec(x, weight, bias=bias, scale=scale_fp32, group_list=group_list,
                                                    split_item=split_item)
            custom_output = self.custom_op_exec(x_clone, weight_clone, bias=bias_clone, scale=scale_npu,
                                                group_list=group_list, split_item=split_item, group_type=group_type,
                                                output_dtype=output_dtype)

            self.assertRtolEqual(x[0], x_clone[0], 1)
            self.assertRtolEqual(supported_output[0], custom_output[0].to(torch.int8), 1)

    @unittest.skip("Skipping test_npu_grouped_matmul_quant_3_nz for now")
    @SupportedDevices(['Ascend910B'])
    def test_npu_grouped_matmul_quant_3_nz(self): # 量化 单多单 NZ
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
        group_list = torch.tensor([256, 1280, 1792]).npu()
        split_item = 3
        group_type = 0

        scale_output_dtype_list = [[torch.float32, torch.float16], [torch.bfloat16, torch.bfloat16]]
        for item in scale_output_dtype_list:
            dtype = item[0]
            output_dtype = item[1]
            x_clone = []
            weight_clone = []
            bias_clone = []
            scale_npu = []
            scale_fp32 = []

            for x_i in x:
                x_clone.append(x_i.clone().npu())
            for weight_i in weight:
                weight_nz = torch_npu.npu_format_cast(weight_i.clone().npu(), 29)
                weight_clone.append(weight_nz)
            for bias_i in bias:
                bias_clone.append(bias_i.clone().npu())
            for _ in range(3):
                scale, _ = self.deq_scale_generate(256)
                scale_fp32.append(scale)
                scale_npu.append(torch.from_numpy(scale).npu().to(dtype))

            supported_output = self.supported_op_exec(x, weight, bias=bias, scale=scale_fp32, group_list=group_list,
                                                    split_item=split_item)
            custom_output = self.custom_op_exec(x_clone, weight_clone, bias=bias_clone, scale=scale_npu,
                                                group_list=group_list, split_item=split_item, group_type=group_type,
                                                output_dtype=output_dtype)

            self.assertRtolEqual(x[0], x_clone[0], 1)
            self.assertRtolEqual(supported_output[0], custom_output[0].to(torch.int8), 1)

    @SupportedDevices(['Ascend910B'])
    def test_npu_grouped_matmul_0_empty_x_w(self):
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

    @unittest.skip("Skipping test_npu_grouped_matmul_A16W4 for now")
    @SupportedDevices(['Ascend910B'])
    def test_npu_grouped_matmul_A16W4(self):
        # pylint:disable = huawei-too-many-arguments
        def gmm_a16_w4_golden(x_in, weight_in, bias_in, antiquant_scale_in, antiquant_offset_in, group_list):
            input_dtype = x_in[0].dtype
            x_in = x_in[0]
            weight_in = weight_in[0]
            bias_in = bias_in[0]
            output = []
            x_pre = []
            has_bias = True if bias_in is not None else False
            offset = 0

            antiquant_scale_pre = torch.unbind(antiquant_scale_in[0])
            antiquant_offset_pre = torch.unbind(antiquant_offset_in[0])

            for group in group_list:
                x_pre.append(x_in[offset: group, :])
                offset = group
            weight_pre = torch.unbind(weight_in)

            x = [None] * len(x_pre)
            weight = [None] * len(weight_pre)
            bias = [None] * len(bias_in)
            antiquant_scale = [None] * len(x_pre)
            antiquant_offset = [None] * len(x_pre)
            ds_torch = [None] * len(x_pre)

            for i, x_i in enumerate(x_pre):
                x[i] = x_i.to(torch.float32)
                if input_dtype == torch.bfloat16:
                    x[i] = (x[i].to(torch.bfloat16)).to(torch.float32)
                    antiquant_scale[i] = antiquant_scale_pre[i].to(torch.float32)
                    antiquant_offset[i] = antiquant_offset_pre[i].to(torch.float32)
                    weight[i] = weight_pre[i].to(torch.float32)
                    weight[i] = (weight[i] + antiquant_offset[i]) * antiquant_scale[i]

                    weight[i] = (weight[i].to(torch.bfloat16)).to(torch.float32)
                    ds_torch[i] = torch.matmul(x[i], weight[i])

                    if has_bias:
                        bias[i] = bias_in[i].to(torch.float32)
                        ds_torch[i] += bias[i]
                    output.append(ds_torch[i].to(torch.bfloat16))
                else:
                    x[i] = x_i.to(torch.float32)
                    weight[i] = weight_pre[i].to(torch.float16)

                    weight[i] = (weight[i] + antiquant_offset_pre[i]) * antiquant_scale_pre[i]
                    weight[i] = weight[i].to(torch.float32)

                    ds_torch[i] = torch.matmul(x[i], weight[i])

                    if has_bias:
                        bias[i] = bias_in[i].to(torch.float32)
                        ds_torch[i] += bias[i]
                    output.append(ds_torch[i].to(torch.float16))
            res = torch.cat(output, dim=0)
            return [res]

        E = 8
        M = 2048
        N = 2688
        K = 5120

        def calc_group_list():
            step = M // (E - 1)

            data = [i * step for i in range(E)]

            # 创建整数类型的张量
            group_list = torch.tensor(data, dtype=torch.int64).npu()
            group_list_cpu = torch.tensor(data, dtype=torch.int64)
            group_list[-1] = M
            group_list_cpu[-1] = M
            return group_list, group_list_cpu

        x = torch.randint(-5, 5, (M, K), device="npu").to(torch.bfloat16)
        weight = torch.randint(-5, 5, (E, K, N), dtype=torch.int32, device="npu").to(torch.int32)
        weight_quant = torch_npu.npu_quantize(weight.to(torch.float32), torch.tensor([1.]).npu(),
                                              None, torch.quint4x2, -1, False)
        bias = torch.randn((E, N), dtype=torch.float32, device="npu")
        quantScale = torch.randn((E, N), dtype=torch.bfloat16, device="npu").uniform_()
        quantOffset = torch.randn((E, N), dtype=torch.bfloat16, device="npu").uniform_()
        groupList, group_list_cpu = calc_group_list() # group_list_type=0

        out = torch_npu.npu_grouped_matmul([x], [weight_quant], bias=[bias], scale=None, offset=None,
                                           antiquant_scale=[quantScale], antiquant_offset=[quantOffset],
                                           per_token_scale=None, group_list=groupList, activation_input=None,
                                           activation_quant_scale=None, activation_quant_offset=None, split_item=3,
                                           group_type=0, group_list_type=0, act_type=0, output_dtype=None)
        out_gold = gmm_a16_w4_golden([x.cpu()], [weight.cpu()], [bias.cpu()],
                                     [quantScale.cpu()], [quantOffset.cpu()], group_list_cpu)

        array1 = out[0].cpu().to(torch.float32).numpy().flatten()
        array2 = out_gold[0].cpu().to(torch.float32).numpy().flatten()
        self.assertTrue(len(array1) == len(array2))
        diff = np.isclose(array1, array2, 0.005, 0.005, equal_nan=True)
        compare = np.sum(diff == True) / len(diff)
        self.assertTrue(compare >= 0.995)
    
    @unittest.skip("Skipping test_npu_grouped_matmul_A8W4 for now")
    @SupportedDevices(['Ascend910B'])
    def test_npu_grouped_matmul_A8W4(self):
        # pylint:disable = huawei-too-many-arguments
        def non_quant_golden(x, weight, scale, perTokenScale, groupList, bias):
            groupNum, k, n = weight.shape
            quantGroupNum = scale.shape[1]
            index = np.cumsum(groupList)
            xSplit = np.split(x, index * 2, axis=0)
            perTokenScaleSplit = np.split(perTokenScale, index, axis=0)
            weightGroup = weight.reshape(groupNum, quantGroupNum, k // quantGroupNum, n).astype(np.int32)
            mmOuts = []
            atomic = np.float16
            for i in range(groupNum):
                xi = xSplit[i].reshape(-1, quantGroupNum, k // quantGroupNum).astype(np.int32)
                mmi = np.zeros([xi.shape[0], n], dtype=atomic)
                for j in range(quantGroupNum):
                    mm = np.matmul(xi[:, j, :], weightGroup[i, j, ...])
                    mm = mm.astype(np.float32) * scale[i, j].reshape(1, -1)
                    mmi = (mmi.astype(atomic) + mm.astype(atomic)).astype(atomic)

                mmi = mmi.reshape(-1, 2, n).astype(np.float32)
                mmi = mmi[:, 0, :] * 16 + mmi[:, 1, :] + bias[i].reshape(1, n)
                mmi = mmi * perTokenScaleSplit[i]
                mmOuts.append(mmi)
            golden = np.concatenate(mmOuts, axis=0)
            golden_tensor = torch.from_numpy(golden)
            return golden_tensor.to(torch.float16)

        # pylint:disable = huawei-too-many-arguments
        def gmm_a8w4_golden(x_in, weight_in, bias_in, scale_in, groupList_in, perTokenScale_in):
            weightNz = weight_in.astype(np.int8)
            groupNum = groupList_in.shape[0]
            m = x_in.shape[0]
            k = x_in.shape[1]
            n = scale_in.shape[2]

            weight = weightNz.reshape(groupNum, k, n)
            xC12 = np.concatenate([x_in.reshape(m, 1, k) // 16, (x_in.reshape(m, 1, k) & 0x0F) - 8], axis=1).reshape(m * 2, k)
            scaleUint32 = scale_in.astype(np.uint32)
            scaleUint32.dtype = np.float32
            out = non_quant_golden(xC12, weight, scaleUint32, perTokenScale_in, groupList_in, bias_in)
            return out

        E = 8
        M = 768
        K = 7168
        N = 4096
        quantGroupSize = 256

        x = torch.randint(-5, 5, (M, K), device="npu").to(torch.int8)
        # A8W4 will inplace change the value of x.
        x_copy = x.clone()
        weight = torch.randint(-5, 5, (E, K, N), dtype=torch.int32, device="npu")
        weight_quant = torch_npu.npu_quantize(weight.to(torch.float32), torch.tensor([1.]).npu(), None, torch.quint4x2, -1, False)
        bias = torch.zeros((E, N), dtype=torch.float32, device="npu").uniform_(-5, 5)
        scale_np = np.random.normal(0, 0.01, (E, 1, N)).astype(np.float32)
        perGroupScale = np.ones([E, K//quantGroupSize, N]).astype(np.float32)
        scaleUint32 = (scale_np * perGroupScale).astype(np.float16).astype(np.float32)
        scaleUint32.dtype = np.uint32
        scaleUint64 = np.zeros((E, K//quantGroupSize, N*2), dtype=np.uint32)
        scaleUint64[...,::2] = scaleUint32
        scaleUint64.dtype = np.int64
        scale = torch.from_numpy(scaleUint64).npu()
        groupList = torch.zeros((E,), dtype=torch.int64, device="npu").fill_(1)
        perTokenScale = torch.zeros((M,1), dtype=torch.float32, device="npu").uniform_()

        out = torch_npu.npu_grouped_matmul([x], [weight_quant], bias=[bias], scale=[scale], offset=None, antiquant_scale=None,
                                           antiquant_offset=None, per_token_scale=[perTokenScale], group_list=groupList,
                                           activation_input=None, activation_quant_scale=None, activation_quant_offset=None,
                                           split_item=3, group_type=0, group_list_type=1, act_type=0, output_dtype=torch.float16)

        x_in = x_copy.cpu().numpy()
        weight_in = weight.cpu().numpy()
        bias_in = bias.cpu().numpy()
        scale_in = scaleUint64
        groupList_in = groupList.cpu().numpy()
        perTokenScale_in = perTokenScale.cpu().numpy()

        out_golden = gmm_a8w4_golden(x_in=x_in, weight_in=weight_in, bias_in=bias_in,
                                     scale_in=scale_in, groupList_in=groupList_in, perTokenScale_in=perTokenScale_in)

        out_shape = out[0].shape
        golden_shape = out_golden.shape
        
        out_dim1 = out_shape[1]
        
        golden_dim0 = golden_shape[0]
        golden_dim1 = golden_shape[1]
        
        self.assertEqual(out_dim1, golden_dim1)
        self.assertEqual(out[0][:golden_dim0, :], out_golden.npu())

    @SupportedDevices(['Ascend910B'])
    def test_npu_grouped_matmul_group_list_none(self):
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

        with self.assertRaisesRegex(RuntimeError, "Requires manual passing group_type"):
            _ = self.custom_op_exec(x_clone, weight_clone, bias=bias_clone, group_list=group_list,
                                    split_item=split_item)


if __name__ == "__main__":
    run_tests()

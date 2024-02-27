import math
import unittest
import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestGroupedMatmul(TestCase):
    def single_matmul(self, x, w, b=None):
        y = np.matmul(x, w)
        if b is not None:
            y += b
        return y

    def supported_op_exec(self, x, weight, bias, group_list, split_item):
        x_split = []
        if split_item == 0 or split_item == 2:
            x_split = x
        elif split_item == 1 or split_item == 3:
            offset = 0
            for item in group_list:
                x_split.append(x[0][offset:offset + item, :])
                offset += item
        bias = bias or [None] * len(weight)
        output = [self.single_matmul(x_split[i], weight[i], bias[i]) for i in range(len(weight))]
        if split_item == 2 or split_item == 3:
            output = [torch.cat(output, 0)]
        return output

    def custom_op_exec(self, x, weight, bias, group_list, split_item):
        return torch_npu.npu_grouped_matmul(x, weight, bias=bias, group_list=group_list, split_item=split_item)

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

        supported_output = self.supported_op_exec(x, weight, bias, group_list, split_item)
        custom_output = self.custom_op_exec(x_clone, weight_clone, bias_clone, group_list, split_item)

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

        supported_output = self.supported_op_exec(x, weight, bias, group_list, split_item)
        custom_output = self.custom_op_exec(x_clone, weight_clone, bias_clone, group_list, split_item)

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
        group_list = [256, 1024, 512]
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

        supported_output = self.supported_op_exec(x, weight, bias, group_list, split_item)
        custom_output = self.custom_op_exec(x_clone, weight_clone, bias_clone, group_list, split_item)

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

        supported_output = self.supported_op_exec(x, weight, bias, group_list, split_item)
        custom_output = self.custom_op_exec(x_clone, weight_clone, bias_clone, group_list, split_item)

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
        group_list = [256, 1024, 512]
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

        supported_output = self.supported_op_exec(x, weight, bias, group_list, split_item)
        custom_output = self.custom_op_exec(x_clone, weight_clone, bias_clone, group_list, split_item)

        self.assertRtolEqual(x[0], x_clone[0], 0.001)
        self.assertRtolEqual(supported_output[0], custom_output[0], 0.001)

if __name__ == "__main__":
    run_tests()

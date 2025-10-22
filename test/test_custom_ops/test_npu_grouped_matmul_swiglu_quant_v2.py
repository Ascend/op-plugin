import unittest
import itertools
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices


class TestNpuGroupedMatmulSwigluQuant(TestCase):
    def GMM_Swiglu_quant(self, x: torch.Tensor, weight: torch.Tensor, perChannelScale: torch.Tensor, perTokenScale: torch.Tensor, m: int):
        """
        执行量化的 GMM（通用矩阵乘法）操作，并使用 SwiGLU 激活函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (m, n)。
            weight (torch.Tensor): 权重张量，形状为 (n, k)。
            perChannelScale (torch.Tensor): 每个通道的缩放因子，形状为 (k,)。
            perTokenScale (torch.Tensor): 每个 token 的缩放因子，形状为 (m,)。
            m (int): token 的数量（x 的行数）。

        返回:
            quantOutput (torch.Tensor): 量化后的输出张量，形状为 (m, k // 2)。
            quantScaleOutput (torch.Tensor): 量化缩放因子，形状为 (m,)。
        """
        # 使用 int32 精度执行矩阵乘法
        c_temp1 = torch.matmul(x.to(torch.int32), weight.to(torch.int32))
        c_temp1 = c_temp1.to(torch.float32)  # 转换回 float32 以便进行缩放

        # 应用每个通道和每个 token 的缩放
        c_temp2 = torch.mul(c_temp1, perChannelScale)
        c_temp3 = torch.mul(c_temp2, perTokenScale.reshape(m, 1))

        # 将结果分成两部分以应用 SwiGLU 激活函数
        c_temp4, gate = c_temp3.chunk(2, dim=-1)
        c_temp5 = c_temp4 * torch.sigmoid(c_temp4)  # SwiGLU 激活
        c_temp6 = c_temp5 * gate  # 与门控值进行逐元素相乘

        # 对输出进行量化
        abs_max = torch.max(torch.abs(c_temp6), -1).values  # 找到最大绝对值以计算缩放因子
        quantScaleOutput = 127 / abs_max  # 计算量化缩放因子
        quantOutput = torch.round(c_temp6 * quantScaleOutput.reshape(m, 1)).to(torch.int8)  # 量化为 int8
        quantScaleOutput = 1 / quantScaleOutput  # 反向量化缩放因子以便后续反量化

        return quantOutput, quantScaleOutput


    def process_groups(self, x: torch.Tensor, weight: torch.Tensor, perChannelScale: torch.Tensor, perTokenScale: torch.Tensor, groupList: torch.Tensor):
        """
        按组处理输入数据，并调用 GMM_Swiglu_quant 函数进行量化计算。

        参数:
            x (torch.Tensor): 输入张量，形状为 (M, N)。
            weight (torch.Tensor): 权重张量列表，每个元素的形状为 (n, k)。
            perChannelScale (torch.Tensor): 每个通道的缩放因子列表，每个元素的形状为 (k,)。
            perTokenScale (torch.Tensor): 每个 token 的缩放因子，形状为 (M,)。
            groupList (list): 定义每个组的 token 数量的列表。

        返回:
            quantOutput (torch.Tensor): 量化后的输出张量，形状为 (M, N // 2)。
            quantScaleOutput (torch.Tensor): 量化缩放因子，形状为 (M,)。
        """
        M, N = x.shape[0], weight.shape[2]  # 获取输入张量的形状
        quantOutput = torch.zeros(M, N // 2).to(torch.int8)  # 初始化量化输出张量
        quantScaleOutput = torch.zeros(M).to(torch.float32)  # 初始化量化缩放因子张量

        start_idx = 0  # 起始索引
        preV = 0  # 前一个组的 token 数量
        groupList = groupList.tolist()
        # 遍历 groupList，按组处理数据
        for i, v in enumerate(groupList):
            currV = v
            tempV = currV - preV  # 计算当前组的 token 数量
            preV = currV  # 更新前一个组的 token 数量
            if (tempV > 0):
            # 调用 GMM_Swiglu_quant 处理当前组
                quantOutput[start_idx:start_idx + tempV], quantScaleOutput[start_idx:start_idx + tempV] = \
                    self.GMM_Swiglu_quant(x[start_idx:start_idx + tempV], 
                                    weight[i], 
                                    perChannelScale[i], 
                                    perTokenScale[start_idx:start_idx + tempV], 
                                    tempV)

            start_idx += tempV  # 更新起始索引以处理下一组
        return quantOutput, quantScaleOutput

    def gen_input_data(self, E, M, K, N):
        x = torch.randint(-128, 127, (M, K), dtype=torch.int8)
        weight = torch.randint(-128, 127, (E, K, N), dtype=torch.int8)
        weightScale = torch.randn(E, N)
        xScale = torch.randn(M)
        groupList = torch.tensor([128, 128], dtype=torch.int64)
        return x, weight, weightScale, xScale, groupList

    @unittest.skip("skip case")
    @SupportedDevices(['Ascend910B'])
    def test_npu_grouped_matmul_swiglu_quant(self, device="npu"):
        # 生成数据
        E = 2
        M = 512
        K = 7168
        N = 4096
        x, weight, weightScale, xScale, groupList = self.gen_input_data(E, M, K, N)
        output0, output1 = self.process_groups(x, weight, weightScale, xScale, groupList)
        # 注：有效数据截至到groupList[-1] 即output0[:groupList[-1],:],output0[:groupList[-1]]
        output0_valid = output0[:groupList[-1], :]
        output1_valid = output1[:groupList[-1]]
        weight_npu = torch_npu.npu_format_cast(weight.npu(), 29)
        output0_npu, output1_npu = torch_npu.npu_grouped_matmul_swiglu_quant_v2(x.npu(), [weight_npu], [weightScale.npu()], xScale.npu(), groupList.npu())
        output0_npu_valid = output0_npu[:groupList[-1], :]
        output1_npu_valid = output1_npu[:groupList[-1]]
        self.assertEqual(output0_valid, output0_npu_valid.cpu(), 1)
        self.assertRtolEqual(output1_valid, output1_npu_valid.cpu())

if __name__ == "__main__":
    run_tests()

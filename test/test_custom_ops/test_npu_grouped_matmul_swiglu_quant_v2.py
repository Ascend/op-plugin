import unittest
import itertools
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices
import math


class TestNpuGroupedMatmulSwigluQuant(TestCase):
    def setUp(self):
        self.ori_allow_internal_format = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
        torch.npu.config.allow_internal_format = True

    def tearDown(self):
        torch.npu.config.allow_internal_format = self.ori_allow_internal_format

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

    def GMM_Swiglu_quant_950(self, i, x: torch.Tensor, weight: torch.Tensor, weightScale: torch.Tensor, xScale: torch.Tensor, m: int):
        """
        执行量化的 GMM（通用矩阵乘法）操作，并使用 SwiGLU 激活函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (m, n)。
            weight (torch.Tensor): 权重张量，形状为 (n, k)。
            weightScale (torch.Tensor): 每个通道的缩放因子，形状为 (k,)。
            xScale (torch.Tensor): 每个 token 的缩放因子，形状为 (m,)。
            m (int): token 的数量（x 的行数）。

        返回:
            quantOutput (torch.Tensor): 量化后的输出张量，形状为 (m, k // 2)。
            quantScaleOutput (torch.Tensor): 量化缩放因子，形状为 (m,)。
        """
        pertoken_scale_mx = xScale
        m, k0, k1 = pertoken_scale_mx.shape
        pertoken_scale_mx = pertoken_scale_mx.reshape(m, k0 * k1)
        pertoken_scale_mx_broadcast = np.repeat(pertoken_scale_mx, 32, axis=-1)
        x_dims = len(x.shape)
        x1_pad_len = pertoken_scale_mx_broadcast.shape[-1] - x.shape[-1]
        x_np = x.float().cpu().numpy
        x = np.pad(x, [(0,0)] * (x_dims - 1) + [(0, x1_pad_len)], mode='constant', constant_values=0)
        x = x * pertoken_scale_mx_broadcast
        x = torch.from_numpy(x.astype(np.float32))

        transposed = np.transpose(weightScale, axes=(0,2,1))
        batch_size, height, width = transposed.shape
        weightScale = transposed.reshape(batch_size * height, width)

        deq_scale_mx_broadcast = np.repeat(weightScale, 32, axis=-2)
        x2_dims = len(weight.shape)
        x2_pad_len = deq_scale_mx_broadcast.shape[-2] - weight.shape[-2]
        weight = np.pad(weight, [(0,0)] * (x2_dims - 2) + [(0, x2_pad_len)] + [(0, 0)], mode='constant', constant_values=0)
        weight = weight * deq_scale_mx_broadcast

        if i == 0:
            x1_temp = x[:group_list[i], :]
        else:
            x1_temp = x[group_list[i-1]:group_list[i], :]
        weight = torch.from_numpy(x.astype(weight.float32))
        c_temp3 = torch.matmul(x1_temp, weight)

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

    def process_groups_950(self, x: torch.Tensor, weight: torch.Tensor, weightScale: torch.Tensor, xScale: torch.Tensor, groupList: torch.Tensor):
        """
        按组处理输入数据，并调用 GMM_Swiglu_quant 函数进行量化计算。

        参数:
            x (torch.Tensor): 输入张量，形状为 (M, N)。
            weight (torch.Tensor): 权重张量列表，每个元素的形状为 (n, k)。
            weightScale (torch.Tensor): 每个通道的缩放因子列表，每个元素的形状为 (k,)。
            xScale (torch.Tensor): 每个 token 的缩放因子，形状为 (M,)。
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
                    self.GMM_Swiglu_quant_950(i, x[start_idx:start_idx + tempV],
                                    weight[i],
                                    weightScale[i],
                                    xScale[start_idx:start_idx + tempV],
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

    def gen_input_data_950(self, E, M, K, N, transpose):
        x = torch.randint(-128, 127, (M, K), dtype=torch.int8).to(torch.float8_e4m3fn)
        weight = torch.randint(-128, 127, (E, K, N), dtype=torch.int8).to(torch.float8_e4m3fn)
        weightScale = torch.randint(low=0, high=256, size=(E, math.ceil(K / 64), N, 2), dtype=torch.uint8)
        xScale = torch.randint(low=0, high=256, size=(M, math.ceil(K / 64), 2), dtype=torch.uint8)
        groupList = torch.tensor([M//2, M//2], dtype=torch.int64)
        return x, weight, weightScale, xScale, groupList
    def gen_input_data_950_mxfp4(self, E, M, K, N, transpose):
        x = torch.randint(0, 256, (M, K), dtype=torch.uint8)
        weight = torch.randint(0, 256, (E, K * 2, N), dtype=torch.uint8)
        weightScale = torch.randint(low=0, high=256, size=(E, math.ceil(K / 64), N * 2, 2), dtype=torch.uint8)
        xScale = torch.randint(low=0, high=256, size=(M, math.ceil(K / 64), 2), dtype=torch.uint8)
        groupList = torch.tensor([int(M/2), int(M/2) + 1], dtype=torch.int64)
        return x, weight, weightScale, xScale, groupList

    def gen_input_data_a4w4(self, E, M, K, N):
        x = torch.randint(-8, 7, (M, K), dtype=torch.int8)
        weight = torch.randint(-5, 5, (E, K, N), dtype=torch.int8)
        weightScale = torch.randn(E, N).to(torch.float32) * 0.01
        xScale = torch.randn(M, dtype=torch.float32) * 0.5
        smooth_scale = torch.randn(E, N // 2, dtype=torch.float32)
        groupList = torch.tensor([M], dtype=torch.int64) if E == 1 else torch.tensor([int(M / 2), M], dtype=torch.int64)
        return x, weight, weightScale, xScale, smooth_scale, groupList

    def prepare_a4w4_npu_inputs(self, x, weight, weightScale, xScale, smooth_scale, groupList, E, K, N, weight_format=0):
        if weight_format == 0:
            weight_quant = torch_npu.npu_quantize(
                weight.to(torch.float32).npu(), torch.tensor([1.], device='npu'), None, torch.quint4x2, -1, False)
        else:
            weight_quant = weight.reshape(E, K // 16, 16, N // 64, 64).permute(0, 3, 1, 2, 4).contiguous()
            weight_quant = weight_quant.npu()
            weight_quant = torch_npu.npu_quantize(
                weight_quant.to(torch.float32), torch.tensor([1.], device='npu'), None, torch.quint4x2, -1, False)
        weightScale = weightScale.view(E, N)
        scale_np = weightScale.cpu().numpy()
        scale_uint32 = scale_np.astype(np.float32)
        scale_uint32 = scale_uint32.view(np.uint32)
        scale_uint64 = np.zeros((E, N * 2), dtype=np.uint32)
        scale_uint64[..., ::2] = scale_uint32.reshape(E, N)
        scale_uint64 = scale_uint64.view(np.int64).reshape(E, N)
        scale = torch.from_numpy(scale_uint64.copy())
        x_quant = torch_npu.npu_quantize(
            x.to(torch.float32).npu(), torch.tensor([1.], device='npu'), None, torch.quint4x2, -1, False)
        return x_quant, weight_quant, scale.npu(), xScale.npu(), smooth_scale.npu(), groupList.npu()

    def gen_input_data_a8w8(self, E, M, K, N):
        x = torch.randint(-100, 100, (M, K), dtype=torch.int8)
        weight = torch.randint(-5, 5, (E, K, N), dtype=torch.int8)
        weightScale = torch.randn(E, N, dtype=torch.float32) * 0.01
        xScale = torch.randn(M, dtype=torch.float32) * 0.5
        groupList = torch.tensor([M], dtype=torch.int64) if E == 1 else torch.tensor([M // 2, M], dtype=torch.int64)
        return x, weight, weightScale, xScale, groupList

    def prepare_a8w8_npu_inputs(self, x, weight, weightScale, xScale, groupList, E, K, N, weight_format=1):
        x_npu = x.npu()
        if weight_format == 1:
            K_dim, N_dim = weight.shape[1], weight.shape[2]
            weight_npu = weight.reshape(E, K_dim // 16, 16, N_dim // 32, 32).permute(0, 3, 1, 2, 4).contiguous().npu()
            weightScale_npu = weightScale[:, :N_dim].contiguous().npu()
        else:
            weight_npu = weight.npu()
            weightScale_npu = weightScale.npu()
        return x_npu, weight_npu, weightScale_npu, xScale.npu(), groupList.npu()

    def _assert_gmm_swiglu_quant_shape(self, output, output_scale, output_shape, output_scale_shape):
        self.assertEqual(tuple(output.shape), output_shape)
        self.assertEqual(tuple(output_scale.shape), output_scale_shape)

    def _assert_nz_weight(self, weight):
        if torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT") == b"enable":
            self.assertEqual(torch_npu.get_npu_format(weight), torch_npu.Format.FRACTAL_NZ)
        else:
            self.assertEqual(weight.dim(), 3)

    def _assert_ascend950_mxfp4_shape(self, quant_dtype, expected_output_dtype):
        # Ascend950 MXFP4 shape-only path: CANN sees FP4 through dtype wrappers while tensors use uint8 storage.
        E, M, K, N = 2, 16, 136, 128
        x = torch.empty((M, K // 2), dtype=torch.uint8).npu()
        weight = torch.empty((E, K, N), dtype=torch.uint8).npu()
        # The weight N axis is packed in uint8 storage, so weightScale carries the unpacked logical N.
        weight_scale = torch.empty((E, math.ceil(K / 64), N * 2, 2), dtype=torch.uint8).npu()
        x_scale = torch.empty((M, math.ceil(K / 64), 2), dtype=torch.uint8).npu()
        group_list = torch.tensor([M // 2, M], dtype=torch.int64).npu()

        output, output_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
            x, [weight], [weight_scale], x_scale, group_list,
            dequant_dtype=torch.float32, dequant_mode=2, quant_mode=2,
            quant_dtype=quant_dtype,
            weight_scale_dtype=torch_npu.float8_e8m0fnu,
            x_scale_dtype=torch_npu.float8_e8m0fnu,
            x_dtype=torch_npu.float4_e2m1fn_x2,
            weight_dtype=torch_npu.float4_e2m1fn_x2)

        if quant_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            expected_output_shape = (M, N)
        else:
            expected_output_shape = (M, N // 2)
        self._assert_gmm_swiglu_quant_shape(
            output, output_scale, expected_output_shape, (M, math.ceil(N / 64), 2))
        self.assertTrue(output.dtype == expected_output_dtype)
        self.assertTrue(output_scale.dtype == torch.float8_e8m0fnu)

    @SupportedDevices(['Ascend910B'])
    def test_npu_grouped_matmul_swiglu_quant_v2_nz_a8w4_int32_pack_shape(self, device="npu"):
        # A8W4 INT32 packed NZ: 3D packed view keeps tail N/8, so logical N must come from weightScale.
        E, M, K, N = 2, 2, 64, 512
        x = torch.empty((M, K), dtype=torch.int8).npu()
        weight_pack = torch.empty((E, K, N // 8), dtype=torch.int32).npu()
        weight = torch_npu.npu_format_cast(weight_pack, 29)
        self._assert_nz_weight(weight)
        self.assertEqual(tuple(weight.shape), (E, K, N // 8))
        self.assertNotEqual(weight.size(2), N)
        weight_scale = torch.empty((E, N), dtype=torch.int64).npu()
        weight_assist = torch.empty((E, N), dtype=torch.float32).npu()
        x_scale = torch.empty((M,), dtype=torch.float32).npu()
        group_list = torch.tensor([1, M], dtype=torch.int64).npu()

        output, output_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
            x, [weight], [weight_scale], x_scale, group_list,
            weight_assist_matrix=[weight_assist], dequant_mode=0, dequant_dtype=torch.float32)
        self._assert_gmm_swiglu_quant_shape(output, output_scale, (M, N // 2), (M,))

    @SupportedDevices(['Ascend910B'])
    def test_npu_grouped_matmul_swiglu_quant_v2_nz_a8w8_int8_shape(self, device="npu"):
        # A8W8 NZ: validates weightScale-based N inference does not regress int8 weights.
        E, M, K, N = 2, 2, 64, 128
        x = torch.empty((M, K), dtype=torch.int8).npu()
        weight_nd = torch.empty((E, K, N), dtype=torch.int8).npu()
        weight = torch_npu.npu_format_cast(weight_nd, 29).view(E, N // 32, K // 16, 16, 32)
        self._assert_nz_weight(weight)
        weight_scale = torch.empty((E, N), dtype=torch.float32).npu()
        x_scale = torch.empty((M,), dtype=torch.float32).npu()
        group_list = torch.tensor([1, M], dtype=torch.int64).npu()

        output, output_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
            x, [weight], [weight_scale], x_scale, group_list,
            dequant_mode=0, dequant_dtype=torch.float32)
        self._assert_gmm_swiglu_quant_shape(output, output_scale, (M, N // 2), (M,))

    @SupportedDevices(['Ascend910B'])
    def test_npu_grouped_matmul_swiglu_quant_v2_nz_per_group_shape(self, device="npu"):
        # A8W4 per-group packed NZ: weightScale has shape (E, K_group_num, N), and N is the tail axis.
        E, M, K, N, k_group_num = 2, 2, 64, 512, 2
        x = torch.empty((M, K), dtype=torch.int8).npu()
        weight_pack = torch.empty((E, K, N // 8), dtype=torch.int32).npu()
        weight = torch_npu.npu_format_cast(weight_pack, 29)
        self._assert_nz_weight(weight)
        self.assertEqual(tuple(weight.shape), (E, K, N // 8))
        self.assertNotEqual(weight.size(2), N)
        weight_scale = torch.empty((E, k_group_num, N), dtype=torch.int64).npu()
        weight_assist = torch.empty((E, N), dtype=torch.float32).npu()
        x_scale = torch.empty((M,), dtype=torch.float32).npu()
        group_list = torch.tensor([1, M], dtype=torch.int64).npu()

        output, output_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
            x, [weight], [weight_scale], x_scale, group_list,
            weight_assist_matrix=[weight_assist], dequant_mode=1, dequant_dtype=torch.float32)
        self._assert_gmm_swiglu_quant_shape(output, output_scale, (M, N // 2), (M,))

    @SupportedDevices(['Ascend950'])
    def test_npu_grouped_matmul_swiglu_quant_v2_nz_mx_shape(self, device="npu"):
        # MX FP8 NZ non-transpose: legal Ascend950 WeightNzV2 path with logical 3D weight and NZ storage.
        E, M, K, N = 8, 2048, 7168, 4096
        x = torch.empty((M, K), dtype=torch.float8_e4m3fn).npu()
        weight_nd = torch.empty((E, K, N), dtype=torch.float8_e4m3fn).npu()
        weight = torch_npu.npu_format_cast(weight_nd, 29)
        self._assert_nz_weight(weight)
        self.assertEqual(tuple(weight.shape), (E, K, N))
        weight_scale = torch.empty((E, math.ceil(K / 64), N, 2), dtype=torch.uint8).npu()
        x_scale = torch.empty((M, math.ceil(K / 64), 2), dtype=torch.uint8).npu()
        group_list = torch.tensor([256, 512, 768, 1024, 1280, 1536, 1792, M], dtype=torch.int64).npu()

        output, output_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
            x, [weight], [weight_scale], x_scale, group_list,
            dequant_dtype=torch.float32, dequant_mode=2, quant_mode=2,
            quant_dtype=torch.float8_e4m3fn,
            weight_scale_dtype=torch_npu.float8_e8m0fnu,
            x_scale_dtype=torch_npu.float8_e8m0fnu)
        self._assert_gmm_swiglu_quant_shape(
            output, output_scale, (M, N // 2), (M, math.ceil((N // 2) / 64), 2))

    @SupportedDevices(['Ascend950'])
    def test_npu_grouped_matmul_swiglu_quant_v2_nz_mx_multi_weight_shape(self, device="npu"):
        # MX FP8 WeightNzV2 multi tensor: weight/weight_scale are E independent tensors.
        E, M, K, N = 2, 16, 128, 256
        x = torch.empty((M, K), dtype=torch.float8_e4m3fn).npu()
        weight = [torch_npu.npu_format_cast(torch.empty((K, N), dtype=torch.float8_e4m3fn).npu(), 29)
                  for _ in range(E)]
        weight_scale = [torch.empty((math.ceil(K / 64), N, 2), dtype=torch.uint8).npu() for _ in range(E)]
        x_scale = torch.empty((M, math.ceil(K / 64), 2), dtype=torch.uint8).npu()
        group_list = torch.tensor([M // 2, M], dtype=torch.int64).npu()

        output, output_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
            x, weight, weight_scale, x_scale, group_list,
            dequant_dtype=torch.float32, dequant_mode=2, quant_mode=2,
            quant_dtype=torch.float8_e4m3fn,
            weight_scale_dtype=torch_npu.float8_e8m0fnu,
            x_scale_dtype=torch_npu.float8_e8m0fnu)
        self._assert_gmm_swiglu_quant_shape(
            output, output_scale, (M, N // 2), (M, math.ceil((N // 2) / 64), 2))

    @unittest.skipIf(
        not hasattr(torch_npu, "float4_e2m1fn_x2"),
        "torch_npu float4 dtype wrappers are required for Ascend950 MXFP4 shape cases.")
    @SupportedDevices(['Ascend950'])
    def test_npu_grouped_matmul_swiglu_quant_v2_nz_mxfp4_multi_weight_shape(self, device="npu"):
        # MXFP4 WeightNzV2 multi tensor: 2D viewShape [K, N] is used to distinguish multi tensor.
        E, M, K, N = 2, 16, 128, 256
        x = torch.empty((M, K // 2), dtype=torch.uint8).npu()
        weight = [torch_npu.npu_format_cast(torch.empty((K, N), dtype=torch.uint8).npu(), 29) for _ in range(E)]
        weight_scale = [torch.empty((math.ceil(K / 64), N, 2), dtype=torch.uint8).npu() for _ in range(E)]
        x_scale = torch.empty((M, math.ceil(K / 64), 2), dtype=torch.uint8).npu()
        group_list = torch.tensor([M // 2, M], dtype=torch.int64).npu()

        output, output_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
            x, weight, weight_scale, x_scale, group_list,
            dequant_dtype=torch.float32, dequant_mode=2, quant_mode=2,
            quant_dtype=torch_npu.float4_e2m1fn_x2,
            weight_scale_dtype=torch_npu.float8_e8m0fnu,
            x_scale_dtype=torch_npu.float8_e8m0fnu,
            x_dtype=torch_npu.float4_e2m1fn_x2,
            weight_dtype=torch_npu.float4_e2m1fn_x2)
        self._assert_gmm_swiglu_quant_shape(
            output, output_scale, (M, N // 4), (M, math.ceil((N // 2) / 64), 2))

    @unittest.skipIf(
        not hasattr(torch_npu, "float4_e2m1fn_x2"),
        "torch_npu float4 dtype wrappers are required for Ascend950 MXFP4 shape cases.")
    @SupportedDevices(['Ascend950'])
    def test_npu_grouped_matmul_swiglu_quant_v2_mxfp4_shape_fp8_output(self, device="npu"):
        # MXFP4 input with FP8 output: catches stale FP4-in-uint8 output shape compensation.
        for quant_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            with self.subTest(quant_dtype=quant_dtype):
                self._assert_ascend950_mxfp4_shape(quant_dtype, quant_dtype)

    @unittest.skipIf(
        not hasattr(torch_npu, "float4_e2m1fn_x2"),
        "torch_npu float4 dtype wrappers are required for Ascend950 MXFP4 shape cases.")
    @SupportedDevices(['Ascend950'])
    def test_npu_grouped_matmul_swiglu_quant_v2_mxfp4_shape_fp4_output(self, device="npu"):
        # MXFP4 input with FP4 output: catches stale FP4-in-uint8 outputScale shape compensation.
        expected_output_dtype = torch.float4_e2m1fn_x2 if hasattr(torch, "float4_e2m1fn_x2") else torch.uint8
        self._assert_ascend950_mxfp4_shape(torch_npu.float4_e2m1fn_x2, expected_output_dtype)

    @SupportedDevices(['Ascend910B'])
    def test_npu_grouped_matmul_swiglu_quant_v2_a4w4(self, device="npu"):
        # 生成数据
        E, M, K, N = 1, 7, 64, 64
        x, weight, weightScale, xScale, smooth_scale, groupList = self.gen_input_data_a4w4(E, M, K, N)
        x_quant, weight_quant, weightScale_npu, xScale_npu, smoothScale_npu, groupList_npu = \
            self.prepare_a4w4_npu_inputs(x, weight, weightScale, xScale, smooth_scale, groupList, E, K, N,
                                        weight_format=1)
        output_npu, output_scale_npu = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
            x_quant,
            [weight_quant],
            [weightScale_npu],
            xScale_npu,
            groupList_npu,
            smooth_scale=smoothScale_npu,
            weight_assist_matrix=None,
            dequant_mode=0,
            dequant_dtype=torch.float32,
            group_list_type=0,
        )
        self.assertEqual(output_npu.dim(), 2)
        self.assertEqual(output_npu.shape[0], M)
        self.assertEqual(output_npu.shape[1], N // 2)
        self.assertEqual(output_scale_npu.dim(), 1)
        self.assertEqual(output_scale_npu.shape[0], M)

    @SupportedDevices(['Ascend910B'])
    def test_npu_grouped_matmul_swiglu_quant_v2_a8w8(self, device="npu"):
        # 生成数据
        E, M, K, N = 2, 64, 128, 64
        x, weight, weightScale, xScale, groupList = self.gen_input_data_a8w8(E, M, K, N)
        x_npu, weight_npu, weightScale_npu, xScale_npu, groupList_npu = self.prepare_a8w8_npu_inputs(
            x, weight, weightScale, xScale, groupList, E, K, N, weight_format=1)
        output_npu, output_scale_npu = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
            x_npu,
            [weight_npu],
            [weightScale_npu],
            xScale_npu,
            groupList_npu,
            smooth_scale=None,
            dequant_mode=0,
            dequant_dtype=torch.float32,
            group_list_type=0,
        )
        self.assertEqual(output_npu.dim(), 2)
        self.assertEqual(output_npu.shape[0], M)
        self.assertEqual(output_npu.shape[1], N // 2)
        self.assertEqual(output_scale_npu.dim(), 1)
        self.assertEqual(output_scale_npu.shape[0], M)
        output_cpu, output_scale_cpu = self.process_groups(x, weight, weightScale, xScale, groupList)
        valid_len = groupList[-1].item()
        self.assertRtolEqual(
            output_npu[:valid_len, :].cpu().float(),
            output_cpu[:valid_len, :].float(),
            prec=1e-2,
        )
        self.assertRtolEqual(
            output_scale_npu[:valid_len].cpu(),
            output_scale_cpu[:valid_len],
            prec=1.0,
        )

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
        output0_npu, output1_npu = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
            x.npu(), [weight_npu], [weightScale.npu()], xScale.npu(), groupList.npu(),
            dequant_dtype=torch.float32)
        output0_npu_valid = output0_npu[:groupList[-1], :]
        output1_npu_valid = output1_npu[:groupList[-1]]
        self.assertEqual(output0_valid, output0_npu_valid.cpu(), 1)
        self.assertRtolEqual(output1_valid, output1_npu_valid.cpu())

    @unittest.skip("Skip existing Ascend950 MXFP8 numeric reference: CPU golden does not support float8 MX inputs yet.")
    @SupportedDevices(['Ascend950'])
    def test_npu_grouped_matmul_swiglu_quant_mxfp8(self, device="npu"):
        # 生成数据
        E = 2
        M = 512
        K = 7168
        N = 4096
        transpose = False
        x, weight, weightScale, xScale, groupList = self.gen_input_data_950(E, M, K, N, transpose)
        output0, output1 = self.process_groups_950(x, weight, weightScale, xScale, groupList)
        # 注：有效数据截至到groupList[-1] 即output0[:groupList[-1],:],output0[:groupList[-1]]
        output0_valid = output0[:groupList[-1], :]
        output1_valid = output1[:groupList[-1]]
        output0_npu, output1_npu = torch_npu.npu_grouped_matmul_swiglu_quant_v2(x.npu(), [weight.npu()], [weightScale.npu()], xScale.npu(), groupList.npu(), dequant_dtype=torch.float32, quant_mode=2,
                                    quant_dtype=torch.float8_e5m2, weight_scale_dtype=torch_npu.float8_e8m0fnu, x_scale_dtype=torch_npu.float8_e8m0fnu)
        output0_npu_valid = output0_npu[:groupList[-1], :]
        output1_npu_valid = output1_npu[:groupList[-1]]
        self.assertEqual(output0_valid, output0_npu_valid.cpu(), 1)
        self.assertRtolEqual(output1_valid, output1_npu_valid.cpu())

    @unittest.skip("Skip existing Ascend950 MXFP4 numeric reference: CPU golden does not support packed float4 MX inputs yet.")
    @SupportedDevices(['Ascend950'])
    def test_npu_grouped_matmul_swiglu_quant_mxfp4(self, device="npu"):
        # 生成数据
        E = 2
        M = 2255
        K = 9
        N = 896
        transpose = False
        x, weight, weightScale, xScale, groupList = self.gen_input_data_950_mxfp4(E, M, K, N, transpose)
        output0, output1 = self.process_groups_950(x, weight, weightScale, xScale, groupList)
        # 注：有效数据截至到groupList[-1] 即output0[:groupList[-1],:],output0[:groupList[-1]]
        output0_valid = output0[:groupList[-1], :]
        output1_valid = output1[:groupList[-1]]
        output0_npu, output1_npu = torch_npu.npu_grouped_matmul_swiglu_quant_v2(x.npu(), [weight.npu()], [weightScale.npu()], xScale.npu(), groupList.npu(), dequant_dtype=torch.float32,
                                   quant_mode=2, quant_dtype=torch.float8_e5m2, weight_scale_dtype=torch_npu.float8_e8m0fnu, x_scale_dtype=torch_npu.float8_e8m0fnu,
                                   x_dtype=torch_npu.float4_e2m1fn_x2, weight_dtype=torch_npu.float4_e2m1fn_x2)
        output0_npu_valid = output0_npu[:groupList[-1], :]
        output1_npu_valid = output1_npu[:groupList[-1]]
        self.assertEqual(output0_valid, output0_npu_valid.cpu(), 1)
        self.assertRtolEqual(output1_valid, output1_npu_valid.cpu())

    @SupportedDevices(['Ascend950'])
    def test_npu_grouped_matmul_swiglu_quant_v2_dequant_dtype_valid_mx(self, device="npu"):
        """Ascend950 上 dequant_dtype 合法值 (float32) 校验"""
        E, M, K, N = 2, 2, 16, 128
        dequant_dtype = torch.float32
        with self.subTest(dequant_dtype=dequant_dtype):
            x, weight, weightScale, xScale, groupList = self.gen_input_data_950(E, M, K, N, False)
            output_npu, output_scale_npu = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
                x.npu(), [weight.npu()], [weightScale.npu()], xScale.npu(), groupList.npu(),
                dequant_dtype=dequant_dtype, quant_mode=2, dequant_mode=2,
                quant_dtype=torch.float8_e5m2,
                weight_scale_dtype=torch_npu.float8_e8m0fnu,
                x_scale_dtype=torch_npu.float8_e8m0fnu)
            self.assertEqual(output_npu.dim(), 2)

    @SupportedDevices(['Ascend950'])
    def test_npu_grouped_matmul_swiglu_quant_v2_dequant_dtype_valid_pertoken(self, device="npu"):
        """Ascend950 上 dequant_dtype 合法值 (float16/bfloat16) 校验"""
        E, M, K, N = 2, 128, 128, 64
        x = torch.randint(0, 256, (M, K), dtype=torch.uint8).to(torch.float8_e4m3fn)
        weight = torch.randint(0, 256, (E, K, N), dtype=torch.uint8).to(torch.float8_e4m3fn)
        weightScale = torch.randint(0, 256, (E, N), dtype=torch.float)
        xScale = torch.randint(0, 256, (M,), dtype=torch.float)
        groupList = torch.tensor([int(M//2), int(M//2) + 1], dtype=torch.int64)
        for dequant_dtype in [torch.float16, torch.bfloat16]:
            with self.subTest(dequant_dtype=dequant_dtype):
                output_npu, output_scale_npu = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
                    x.npu(), [weight.npu()], [weightScale.npu()], xScale.npu(), groupList.npu(),
                    dequant_dtype=dequant_dtype, quant_mode=0, dequant_mode=0,
                    quant_dtype=torch.float8_e5m2
                    )
                self.assertEqual(output_npu.dim(), 2)

    @SupportedDevices(['Ascend950'])
    def test_npu_grouped_matmul_swiglu_quant_v2_dequant_dtype_invalid(self, device="npu"):
        """Ascend950 上 dequant_dtype 非法值应抛出 RuntimeError"""
        E, M, K, N = 2, 128, 128, 64
        x, weight, weightScale, xScale, groupList = self.gen_input_data_950(E, M, K, N, False)
        with self.assertRaises(RuntimeError):
            torch_npu.npu_grouped_matmul_swiglu_quant_v2(
                x.npu(), [weight.npu()], [weightScale.npu()], xScale.npu(), groupList.npu(),
                dequant_dtype=torch.int32, quant_mode=2, dequant_mode=2,
                quant_dtype=torch.float8_e5m2,
                weight_scale_dtype=torch_npu.float8_e8m0fnu,
                x_scale_dtype=torch_npu.float8_e8m0fnu)

if __name__ == "__main__":
    run_tests()

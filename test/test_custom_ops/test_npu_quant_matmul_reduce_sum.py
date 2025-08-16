import unittest
import torch
import torch_npu
import numpy as np
import torch.nn.functional as F

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

MIN_ERR = 1e-7
EB_THRESHOLD = 2**(-8)
ERR_THRESHOLD = 2**(-8)


def get_mare(golden: torch.Tensor, actual: torch.Tensor):
    # 最大相对误差
    golden = golden.to(torch.float32)
    abs_error = torch.abs(actual.to(torch.float32) - golden) / (torch.abs(golden) + MIN_ERR)
    mare = torch.max(abs_error.flatten())
    return mare


def get_mere(golden: torch.Tensor, actual: torch.Tensor):
    # 平均相对误差
    golden = golden.to(torch.float32)
    abs_error = torch.abs(actual.to(torch.float32) - golden) / (torch.abs(golden) + MIN_ERR)
    mere = torch.mean(abs_error)
    return mere


def get_rmse(golden: torch.Tensor, actual: torch.Tensor):
    # 均方根误差
    golden = golden.to(torch.float32)
    sqr_err = torch.pow((actual.to(torch.float32) - golden), 2)
    rmse = torch.sqrt(torch.mean(sqr_err))
    return rmse


def compare_cv(golden: torch.Tensor, golden_high_type: torch.Tensor, actual: torch.Tensor):
    mare_npu = get_mare(golden, actual)
    mare_high_type = get_mare(golden, golden_high_type)

    mere_npu = get_mere(golden, actual)
    mere_high_type = get_mere(golden, golden_high_type)

    rmse_npu = get_rmse(golden, actual)
    rmse_high_type = get_rmse(golden, golden_high_type)

    mare_rate = mare_npu / max(mare_high_type, ERR_THRESHOLD)
    mere_rate = mere_npu / max(mere_high_type, ERR_THRESHOLD)
    rmse_rate = rmse_npu / max(rmse_high_type, ERR_THRESHOLD)

    result = (mare_rate < 10) and (mere_rate < 2) and (rmse_rate < 2)
    return result


class TestGroupedMatmulAdd(TestCase):

    @unittest.skip('Skip until cann package after 20240813 is used.')
    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_reduce_sum_01(self):
        B, M, K, N = 8, 2048, 1024, 7168
        # --- init on CPU---
        x_nd = torch.randint(-10, 10, (B, M, K), dtype=torch.int8)
        w_nd = torch.randint(-10, 10, (B, K, N), dtype=torch.int8)
        scale = torch.rand((N,), dtype=torch.bfloat16)
        pertoken_scale = torch.rand((B, M), dtype=torch.float32)
        y_dtype = torch.bfloat16

        # --- Cal Golden_fp32 ---
        golden = torch.bmm(x_nd.float(), w_nd.float()).to(torch.float32)
        golden = scale[None, None, :] * golden
        golden = pertoken_scale[:, :, None] * golden
        golden_fp32 = torch.sum(golden, dim=0).to(torch.bfloat16)

        # --- Cal golden_bf16 ---
        torch.use_deterministic_algorithms(True)
        golden_bf16 = torch.zeros(M, N, dtype=torch.bfloat16)
        for i in range(B):
            golden_bf16 += golden[i, ...].to(torch.bfloat16)

        # --- Move to NPU ---
        device = torch.device("npu:0")
        x_nd_npu = x_nd.to(device)
        w_nd_npu = w_nd.to(device)
        w_nz_npu = torch_npu.npu_format_cast(w_nd_npu.contiguous(), 29)
        x2_scale = scale.to(device)
        x1_scale = pertoken_scale.to(device)

        # --- Calculation ---
        custom_out = torch_npu.npu_quant_matmul_reduce_sum(x_nd_npu, w_nz_npu, x1_scale=x1_scale, x2_scale=x2_scale)

        self.assertTrue(compare_cv(golden_fp32, golden_bf16, custom_out.cpu()))

    @unittest.skip('Skip until cann package after 20240813 is used.')
    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_reduce_sum_02(self):
        B, M, K, N = 16, 1024, 1024, 7168
        # --- init on CPU---
        x_nd = torch.randint(-10, 10, (B, M, K), dtype=torch.int8)
        w_nd = torch.randint(-10, 10, (B, K, N), dtype=torch.int8)
        scale = torch.rand((N,), dtype=torch.bfloat16)
        pertoken_scale = torch.rand((B, M), dtype=torch.float32)

        # --- Cal Golden_fp32 ---
        golden = torch.bmm(x_nd.float(), w_nd.float()).to(torch.float32)
        golden = scale[None, None, :] * golden
        golden = pertoken_scale[:, :, None] * golden
        golden_fp32 = torch.sum(golden, dim=0).to(torch.bfloat16)

        # --- Cal golden_bf16 ---
        torch.use_deterministic_algorithms(True)
        golden_bf16 = torch.zeros(M, N, dtype=torch.bfloat16)
        for i in range(B):
            golden_bf16 += golden[i, ...].to(torch.bfloat16)

        # --- Move to NPU ---
        device = torch.device("npu:0")
        x_nd_npu = x_nd.to(device)
        w_nd_npu = w_nd.to(device)
        w_nz_npu = torch_npu.npu_format_cast(w_nd_npu.contiguous(), 29)
        x2_scale = scale.to(device)
        x1_scale = pertoken_scale.to(device)

        # --- Calculation ---
        custom_out = torch_npu.npu_quant_matmul_reduce_sum(x_nd_npu, w_nz_npu, x1_scale=x1_scale, x2_scale=x2_scale)

        self.assertTrue(compare_cv(golden_fp32, golden_bf16, custom_out.cpu()))


if __name__ == "__main__":
    run_tests()

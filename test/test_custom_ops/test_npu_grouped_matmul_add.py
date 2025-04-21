import unittest

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

MIN_ERR = 1e-7


class TestGroupedMatmulAdd(TestCase):

    def get_eb(self, golden:torch.Tensor, actual:torch.Tensor):
        golden = golden.to(torch.float32)
        golden_nmax = torch.clamp(torch.abs(golden), min = 1)
        actual_error = actual.to(torch.float32) - golden
        error_balance = torch.mean(actual_error / golden_nmax)
        return error_balance

    def get_mare(self, golden:torch.Tensor, actual:torch.Tensor):
        golden = golden.to(torch.float32)
        abs_error = torch.abs(actual.to(torch.float32) - golden) / (torch.abs(golden) + MIN_ERR)
        mare = torch.max(abs_error.flatten())
        return mare

    def get_mere(self, golden:torch.Tensor, actual:torch.Tensor):
        golden = golden.to(torch.float32)
        abs_error = torch.abs(actual.to(torch.float32) - golden) / (torch.abs(golden) + MIN_ERR)
        mere = torch.mean(abs_error)
        return mere

    def get_rmse(self, golden:torch.Tensor, actual:torch.Tensor):
        golden = golden.to(torch.float32)
        sqr_err = torch.pow((actual.to(torch.float32) - golden), 2)
        rmse = torch.sqrt(torch.mean(sqr_err))
        return rmse

    def compare_cv(self, golden:torch.Tensor, golden_high_type:torch.Tensor, actual:torch.Tensor):
        eb_threshold = 2**(-14)
        err_threshold = 2**(-14)
        mare_npu = self.get_mare(golden, actual)
        mare_high_type = self.get_mare(golden, golden_high_type)

        mere_npu = self.get_mere(golden, actual)
        mere_high_type = self.get_mere(golden, golden_high_type)

        rmse_npu = self.get_rmse(golden, actual)
        rmse_high_type = self.get_rmse(golden, golden_high_type)

        mare_rate = mare_npu / max(mare_high_type, err_threshold)
        mere_rate = mere_npu / max(mere_high_type, err_threshold)
        rmse_rate = rmse_npu / max(rmse_high_type, err_threshold)

        EB = self.get_eb(golden_high_type, actual)
        result = (mare_rate < 10) and (mere_rate < 2) and (rmse_rate < 2) and (EB < eb_threshold)
        return result

    def cpu_golden_fp32(self, y, x, weight, group_list, transpose_x, transpose_weight, group_type):
        result = []
        last = 0
        for i in group_list:
            x_tensor = x[last:i, :].cpu().to(torch.float)
            weight_tensor = weight[last:i, :].cpu().to(torch.float)
            result.append(torch.matmul(x_tensor.t(), weight_tensor))
            last = i
        
        result = torch.stack(result).reshape(y.shape) + y.cpu()
        
        return result.npu()

    def cpu_golden_fp64(self, y, x, weight, group_list, transpose_x, transpose_weight, group_type):
        result = []
        last = 0
        for i in group_list:
            x_tensor = x[last:i, :].cpu().to(torch.float64)
            weight_tensor = weight[last:i, :].cpu().to(torch.float64)
            result.append(torch.matmul(x_tensor.t(), weight_tensor))
            last = i
        
        result = torch.stack(result).reshape(y.shape) + y.cpu().to(torch.float64)
        
        return result.npu()

    @unittest.skip("Skipping test_npu_grouped_matmul_add_ for now")
    @SupportedDevices(['Ascend910B'])
    def test_npu_grouped_matmul_add_(self):
        x = torch.randn(512, 256, dtype=torch.float16, device="npu")
        weight = torch.randn(512, 256, dtype=torch.float16, device="npu")
        y = torch.randn(512, 256, dtype=torch.float, device="npu")
        group_list = torch.tensor([256, 512]).to(torch.int64).npu()
        transpose_x = True
        transpose_weight = False
        group_type = 2

        mx = x.clone()
        mweight = weight.clone()
        my = y.clone()
        mgroup_list = group_list.clone()
        torch_npu.npu_grouped_matmul_add_(y, x, weight, group_list, transpose_x=transpose_x, transpose_weight=transpose_weight, group_type=group_type)

        golden_fp32 = self.cpu_golden_fp32(my, mx, mweight, mgroup_list, transpose_x, transpose_weight, group_type)
        golden_fp64 = self.cpu_golden_fp64(my, mx, mweight, mgroup_list, transpose_x, transpose_weight, group_type)

        self.assertTrue(self.compare_cv(golden_fp32, golden_fp64, y))


if __name__ == "__main__":
    run_tests()

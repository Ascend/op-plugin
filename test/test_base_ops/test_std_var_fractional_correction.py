# Copyright (c) 2023 Huawei Technologies Co., Ltd
# Licensed under the BSD 3-Clause License.
#
# Guards PTA op-plugin fallback when correction (ddof) is not representable as int64 for aclnn
# std / std_mean / var / var_mean.

import unittest

import torch

from torch_npu.testing.testcase import TestCase, run_tests

RUN_NPU = torch.npu.is_available()


@unittest.skipIf(not RUN_NPU, "requires NPU")
class TestStdVarFractionalCorrection(TestCase):
    """Fractional Scalar correction must match CPU; integer ddof only in aclnn would truncate via toLong()."""

    def setUp(self):
        super().setUp()
        torch.manual_seed(0)
        # (5, 5) along dim=1 -> n=5 per row; correction=1.3 is valid and differs from truncated 1
        self.x_cpu = torch.randn(5, 5, dtype=torch.float32)
        self.x_npu = self.x_cpu.npu()
        self.dim = 1
        self.correction = 1.3

    def test_std_fractional_correction_matches_cpu(self):
        expected = torch.std(self.x_cpu, dim=self.dim, correction=self.correction)
        actual = torch.std(self.x_npu, dim=self.dim, correction=self.correction).cpu()
        self.assertRtolEqual(expected.numpy(), actual.numpy())

    def test_var_fractional_correction_matches_cpu(self):
        expected = torch.var(self.x_cpu, dim=self.dim, correction=self.correction)
        actual = torch.var(self.x_npu, dim=self.dim, correction=self.correction).cpu()
        self.assertRtolEqual(expected.numpy(), actual.numpy())

    def test_std_mean_fractional_correction_matches_cpu(self):
        std_e, mean_e = torch.std_mean(self.x_cpu, dim=self.dim, correction=self.correction)
        std_a, mean_a = torch.std_mean(self.x_npu, dim=self.dim, correction=self.correction)
        self.assertRtolEqual(std_e.numpy(), std_a.cpu().numpy())
        self.assertRtolEqual(mean_e.numpy(), mean_a.cpu().numpy())

    def test_var_mean_fractional_correction_matches_cpu(self):
        var_e, mean_e = torch.var_mean(self.x_cpu, dim=self.dim, correction=self.correction)
        var_a, mean_a = torch.var_mean(self.x_npu, dim=self.dim, correction=self.correction)
        self.assertRtolEqual(var_e.numpy(), var_a.cpu().numpy())
        self.assertRtolEqual(mean_e.numpy(), mean_a.cpu().numpy())

    def test_std_out_fractional_correction_matches_cpu(self):
        expected = torch.empty(5, dtype=torch.float32)
        actual = torch.empty(5, dtype=torch.float32).npu()
        torch.std(self.x_cpu, dim=self.dim, correction=self.correction, out=expected)
        torch.std(self.x_npu, dim=self.dim, correction=self.correction, out=actual)
        self.assertRtolEqual(expected.numpy(), actual.cpu().numpy())


if __name__ == "__main__":
    run_tests()

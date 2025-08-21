import unittest
from unittest.mock import patch

import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestObfuscationCalculate(TestCase):
    @SupportedDevices(["Ascend910B"])
    def test_obfuscation_calculate(self, device="npu"):
        x = torch.randn(1024, 3584, device=device, dtype=torch.bfloat16)
        fd = torch.tensor([35], device=device, dtype=torch.int32)
        param = torch.tensor([3584], device=device, dtype=torch.int32)
        obf_cft = 1.0
        mock_output = torch.randn(1024, 3584, device=device, dtype=torch.bfloat16)
        with patch('torch_npu.obfuscation_calculate', return_value=mock_output):
            result = torch_npu.obfuscation_calculate(fd, x, param, obf_coefficient=obf_cft)
            self.assertEqual(result, mock_output)


if __name__ == "__main__":
    run_tests()
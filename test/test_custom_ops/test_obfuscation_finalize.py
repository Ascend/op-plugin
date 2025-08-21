import unittest
from unittest.mock import patch

import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestObfuscationFinalize(TestCase):
    @SupportedDevices(["Ascend910B"])
    def test_obfuscation_finalize(self, device="npu"):
        fd_to_close = torch.tensor([35], device=device, dtype=torch.int32)
        mock_output = torch.tensor([35], device=device, dtype=torch.int32)
        with patch('torch_npu.obfuscation_finalize', return_value=mock_output):
            result = torch_npu.obfuscation_finalize(fd_to_close)
            self.assertEqual(result, mock_output)


if __name__ == "__main__":
    run_tests()

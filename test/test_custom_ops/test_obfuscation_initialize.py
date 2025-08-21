import unittest
from unittest.mock import patch

import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestObfuscationInitialize(TestCase):
    @SupportedDevices(["Ascend910B"])
    def test_obfuscation_initialize(self, device="npu"):
        hidden_size = 3584
        tp_rank = 4
        cmd = 1
        data_type = torch.bfloat16
        thread_num = 4
        obf_cft = 1.0
        mock_output = torch.tensor([35], device=device, dtype=torch.int32)
        with patch('torch_npu.obfuscation_initialize', return_value=mock_output):
            result = torch_npu.obfuscation_initialize(hidden_size, tp_rank, cmd, data_type=data_type, thread_num=thread_num, obf_coefficient=obf_cft)
            self.assertEqual(result, mock_output)


if __name__ == "__main__":
    run_tests()
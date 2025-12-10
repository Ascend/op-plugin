import unittest
import torch
import torch_npu
import hypothesis

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestSTFT(TestCase):

    @SupportedDevices(['Ascend910B'])
    def test_stft_complex64(self):
        input_tensor = torch.randn(8, 30000, dtype=torch.complex64)
        res = torch.stft(input_tensor, 400, 160, 400, center=False, normalized=False,
                         onesided=False, return_complex=True)
        input_tensor_npu = input_tensor.npu()
        res_npu = torch.stft(input_tensor_npu, 400, 160, 400, center=False, normalized=False,
                             onesided=False, return_complex=True)

        cpu_output = torch.view_as_real(res)
        npu_output = torch.view_as_real(res_npu)

        self.assertRtolEqual(cpu_output, npu_output)

if __name__ == "__main__":
    run_tests()

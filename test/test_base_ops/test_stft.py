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

    @SupportedDevices(['Ascend910B'])
    def test_stft_empty_input(self):
        input_tensor = torch.zeros(0, 19, dtype=torch.float32)
        window = torch.ones(14, dtype=torch.float32)
        res = torch.stft(input_tensor, 18, 8, 14, window=window, center=False,
                         normalized=True, onesided=False, return_complex=False)

        input_tensor_npu = input_tensor.npu()
        window_npu = window.npu()
        res_npu = torch.stft(input_tensor_npu, 18, 8, 14, window=window_npu, center=False,
                             normalized=True, onesided=False, return_complex=False)

        self.assertEqual(res.shape, res_npu.shape)
        self.assertEqual(res.shape, torch.Size([0, 18, 1, 2]))
        self.assertEqual(res.numel(), 0)
        self.assertEqual(res_npu.numel(), 0)

if __name__ == "__main__":
    run_tests()

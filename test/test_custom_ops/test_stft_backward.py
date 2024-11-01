import unittest
import torch
import torch_npu
import hypothesis

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestSTFTBackward(TestCase):

    @unittest.skipIf("1.11.0" in torch.__version__,
                     "OP `stft_backward` is not supported on torch v1.11.0, skip this ut for this torch version")
    @SupportedDevices(['Ascend910B'])
    def test_stft_backward_float32(self):
        input_tensor = torch.randn(10)
        input_tensor.requires_grad = True
        window_tensor = torch.randn(8)

        res = torch.stft(input_tensor, 8, win_length=8, window=window_tensor,
                        onesided=False, center=False, return_complex=False).sum()
        res.backward()

        input_tensor_npu = input_tensor.npu().detach()
        input_tensor_npu.requires_grad = True
        window_tensor_npu = window_tensor.npu().detach()

        res_npu = torch.stft(input_tensor_npu, 8, win_length=8, window=window_tensor_npu,
                        onesided=False, center=False, return_complex=False).sum()
        res_npu.backward()

        grad = input_tensor.grad
        grad_npu = input_tensor_npu.grad

        self.assertRtolEqual(grad, grad_npu)

    @unittest.skipIf("1.11.0" in torch.__version__,
                     "OP `stft_backward` is not supported on torch v1.11.0, skip this ut for this torch version")
    @SupportedDevices(['Ascend910B'])
    def test_stft_backward_complex64(self):
        input_tensor = torch.randn(10, dtype=torch.complex64)
        input_tensor.requires_grad = True
        window_tensor = torch.randn(8, dtype=torch.complex64)

        res = torch.stft(input_tensor, 8, win_length=8, window=window_tensor,
                        onesided=False, center=False, return_complex=False).sum()
        res.backward()

        input_tensor_npu = input_tensor.npu().detach()
        input_tensor_npu.requires_grad = True
        window_tensor_npu = window_tensor.npu().detach()

        res_npu = torch.stft(input_tensor_npu, 8, win_length=8, window=window_tensor_npu,
                        onesided=False, center=False, return_complex=False).sum()
        res_npu.backward()

        grad = torch.view_as_real(input_tensor.grad)
        grad_npu = torch.view_as_real(input_tensor_npu.grad)

        self.assertRtolEqual(grad, grad_npu)

if __name__ == "__main__":
    run_tests()

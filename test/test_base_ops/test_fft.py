import unittest
import torch
import torch_npu
import hypothesis

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestFFT1d(TestCase):

    @SupportedDevices(['Ascend910B'])
    def test_fft_1d(self):
        real_part = torch.tensor([1.0, 2.0, 3.0, 4.0])
        image_part = torch.tensor([5.0, 6.0, 7.0, 8.0])
        complex_tensor = torch.complex(real_part, image_part)
        complex_tensor_npu = complex_tensor.npu()

        cpu_output = torch.fft.fft(complex_tensor)
        npu_output = torch.fft.fft(complex_tensor_npu)

        cpu_output_real = torch.view_as_real(cpu_output)
        npu_output_real = torch.view_as_real(npu_output)

        self.assertRtolEqual(cpu_output_real, npu_output_real)

    @SupportedDevices(['Ascend910B'])
    def test_fft_1d_float32(self):
        input_tensor = torch.rand(5, 128, 128)

        cpu_output = torch.fft.fft(input_tensor)
        npu_output = torch.fft.fft(input_tensor.npu())

        cpu_output_real = torch.view_as_real(cpu_output)
        npu_output_real = torch.view_as_real(npu_output)

        self.assertRtolEqual(cpu_output_real, npu_output_real.cpu())

    @SupportedDevices(['Ascend910B'])
    def test_rfft_1d_dtype(self):
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0]).bool()
        tensor_npu = tensor.npu()

        cpu_output = torch.fft.rfft(tensor)
        npu_output = torch.fft.rfft(tensor_npu)

        cpu_output_real = torch.view_as_real(cpu_output)
        npu_output_real = torch.view_as_real(npu_output)

        self.assertRtolEqual(cpu_output_real, npu_output_real)

    @SupportedDevices(['Ascend910B'])
    def test_fft_1d_nfft_equal_to_1(self):
        real_part = torch.tensor([1.0])
        image_part = torch.tensor([5.0])
        complex_tensor = torch.complex(real_part, image_part)
        complex_tensor_npu = complex_tensor.npu()

        cpu_output = torch.fft.fft(complex_tensor)
        npu_output = torch.fft.fft(complex_tensor_npu)

        cpu_output_real = torch.view_as_real(cpu_output)
        npu_output_real = torch.view_as_real(npu_output)

        self.assertRtolEqual(cpu_output_real, npu_output_real)
    
    def test_fft_2d_float(self):
        real = torch.randn(32, 1, dtype=torch.float32)
        imag = torch.randn(32, 1, dtype=torch.float32)
        complex_tensor = torch.complex(real, imag)
        complex_tensor_npu = complex_tensor.npu()

        cpu_output = torch.fft.fft(complex_tensor, dim=0)
        npu_output = torch.fft.fft(complex_tensor_npu, dim=0)

        self.assertTrue(torch.allclose(cpu_output, npu_output.cpu(), rtol=1e-3, atol=1e-3))


if __name__ == "__main__":
    run_tests()

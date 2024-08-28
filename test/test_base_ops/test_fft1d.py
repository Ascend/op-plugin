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


if __name__ == "__main__":
    run_tests()

import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices


class TestRFFT(TestCase):
    @SupportedDevices(['Ascend910B'])
    def supported_op_exec(self, x, length, dim, norm):
        return torch.fft.rfft(input=x, n=length, dim=dim, norm=norm)

    def custom_op_exec(self, x, length, dim, norm):
        return torch.fft.rfft(input=x.npu(), n=length, dim=dim, norm=norm)
        
    def test_npu_rfft_meta(self):
        shape = [64, 64, 1024]
        length = shape[-1]
        dim = -1
        norm = "backward"
        x = torch.randn(shape, dtype=torch.float32)

        supported_output = self.supported_op_exec(x, length, dim, norm)
        custom_output = self.custom_op_exec(x, length, dim, norm)
        self.assertRtolEqual(supported_output.real, custom_output.real)
        self.assertRtolEqual(supported_output.imag, custom_output.imag)
            
if __name__ == "__main__":
    run_tests()
import unittest
import random
import torch
import torch_npu
import hypothesis
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestIrfftBackward(TestCase):

    rtol = 0.001
    atol = 0.001
    error_extent = 0.001

    def assert_equal(self, cpu_out, npu_out):
        cpu_out = torch.view_as_real(cpu_out)
        npu_out = torch.view_as_real(npu_out.cpu())

        if (cpu_out.shape != npu_out.shape):
            self.fail("shape error")
        if (cpu_out.dtype != npu_out.dtype):
            self.fail("dtype error!")

        error_count = torch.sum(~torch.isclose(cpu_out, npu_out, TestIrfftBackward.rtol, TestIrfftBackward.atol)).item()
        error_percent = error_count / cpu_out.numel()
        if error_percent > TestIrfftBackward.error_extent:
            self.fail("value error!")

        return True
    
    def create_input_tensor(self, shape, dtype):
        return torch.randn(shape, dtype=dtype)


    @SupportedDevices(['Ascend910B'])
    @unittest.skipIf("1.11.0" in torch.__version__,
                "OP `irfft_backward` is not supported on torch v1.11.0, skip this ut for this torch version")
    def test_rfft_backward_complex64(self):
        shapes = [[256, 66], [128, 129]]
        for shape in shapes:
            cpu_in = self.create_input_tensor(shape, torch.complex64).requires_grad_(True)
            npu_in = cpu_in.detach().npu().requires_grad_(True)

            cpu_graph_val = torch.fft.irfft(cpu_in)
            npu_graph_val = torch.fft.irfft(npu_in)

            cpu_in.grad = None
            npu_in.grad = None
            cpu_graph_val = cpu_graph_val.backward(cpu_graph_val)
            npu_graph_val = npu_graph_val.backward(npu_graph_val)

            cpu_out = cpu_in.grad
            npu_out = npu_in.grad

            self.assert_equal(cpu_out, npu_out)


if __name__ == "__main__":
    run_tests()

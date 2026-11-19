import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestSlogdet(TestCase):
    def cpu_op_exec(self, input1):
        sign, logabsdet = torch.slogdet(input1)
        sign = sign.numpy()
        logabsdet = logabsdet.numpy()
        return sign, logabsdet

    def npu_op_exec(self, input1):
        sign, logabsdet = torch.slogdet(input1)
        sign = sign.cpu()
        logabsdet = logabsdet.cpu()
        sign = sign.numpy()
        logabsdet = logabsdet.numpy()
        return sign, logabsdet

    def test_slogdet_shape_format(self):
        shape_format = [
            [np.float32, -1, (3, 3)],
            [np.float32, -1, (4, 3, 3)],
            [np.float32, -1, (5, 5, 5, 5)],
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -100, 100)
            cpu_output, cpu_indices = self.cpu_op_exec(cpu_input)
            npu_output, npu_indices = self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_indices, npu_indices)

    def test_slogdet_complex(self):
        """torch.slogdet must return a real-valued logabsdet for complex input.
        aclnnSlogdet/LogMatrixDeterminant keep the input dtype for logOut, so
        the adapter converts the complex temporary to the corresponding real
        dtype, while sign keeps the complex dtype."""
        cases = [(torch.complex64, torch.float32), (torch.complex128, torch.float64)]
        shapes = [(3, 3), (4, 3, 3), (2, 5, 5), (0, 3, 3)]
        for in_dtype, expected_dtype in cases:
            tol = dict(rtol=1e-4, atol=1e-5) if in_dtype == torch.complex128 else dict(rtol=1e-3, atol=1e-3)
            for shape in shapes:
                real = torch.randn(shape)
                imag = torch.randn(shape)
                cpu_input = torch.complex(real, imag).to(in_dtype)
                npu_input = cpu_input.npu()

                cpu_sign, cpu_logabsdet = torch.slogdet(cpu_input)
                npu_sign, npu_logabsdet = torch.slogdet(npu_input)
                npu_sign, npu_logabsdet = npu_sign.cpu(), npu_logabsdet.cpu()

                self.assertEqual(cpu_logabsdet.dtype, expected_dtype)
                self.assertEqual(npu_logabsdet.dtype, expected_dtype)
                self.assertEqual(cpu_sign.dtype, in_dtype)
                self.assertEqual(npu_sign.dtype, in_dtype)
                torch.testing.assert_close(cpu_sign, npu_sign, **tol)
                torch.testing.assert_close(cpu_logabsdet, npu_logabsdet, **tol)


if __name__ == "__main__":
    run_tests()

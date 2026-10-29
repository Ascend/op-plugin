import torch
import numpy as np
from torch import linalg as LA

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestLinalgSvd(TestCase):

    def cpu_out_exec(self, data, full_matrices):
        cpu_U, cpu_S, cpu_Vh = LA.svd(data, full_matrices=full_matrices)
        return cpu_U, cpu_S, cpu_Vh

    def npu_out_exec(self, data, full_matrices):
        npu_U, npu_S, npu_Vh = LA.svd(data, full_matrices=full_matrices)
        return npu_U.cpu(), npu_S.cpu(), npu_Vh.cpu()

    def construct_S_diag_filled(self, full_matrices, U, S, Vh):
        *batch_dims, r = S.shape
        *batch_dims, _, m = U.shape
        *batch_dims, n, _ = Vh.shape
        S_diag_filled = torch.zeros(*batch_dims, m, n).to(U.dtype)
        diag_blocks = torch.diag_embed(S)
        S_diag_filled[..., :r, :r] = diag_blocks
        return S_diag_filled

    def exec_linalg_svd(self, dtype_list):
        format_list = [0]
        shape_list = [[2, 3], [1, 1, 1, 1, 2, 2, 2, 3], [2, 2, 3, 2]]
        dtype_shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]
        full_matrices_list = [True, False]
        # assertRtolEqual requires an exact numpy dtype match and rejects complex
        # dtypes, so use torch.testing.assert_close for complex inputs.
        is_complex = any(np.dtype(d).kind == "c" for d in dtype_list)
        for item in dtype_shape_format:
            for full_matrices in full_matrices_list:
                cpu_input_1, npu_input_1 = create_common_tensor(item, -100, 100)
                cpu_U, cpu_S, cpu_Vh = self.cpu_out_exec(cpu_input_1, full_matrices)
                npu_U, npu_S, npu_Vh = self.npu_out_exec(npu_input_1, full_matrices)
                if is_complex:
                    torch.testing.assert_close(cpu_U.abs(), npu_U.abs(), rtol=1e-4, atol=1e-5)
                    torch.testing.assert_close(cpu_S.abs(), npu_S.abs(), rtol=1e-4, atol=1e-5)
                    torch.testing.assert_close(cpu_Vh.abs(), npu_Vh.abs(), rtol=1e-4, atol=1e-5)
                else:
                    self.assertRtolEqual(cpu_U.abs().numpy(), npu_U.abs().numpy())
                    self.assertRtolEqual(cpu_S.abs().numpy(), npu_S.abs().numpy())
                    self.assertRtolEqual(cpu_Vh.abs().numpy(), npu_Vh.abs().numpy())

                cpu_S_diag_filled = self.construct_S_diag_filled(full_matrices, cpu_U, cpu_S, cpu_Vh)
                cpu_A_reconstructed = cpu_U @ cpu_S_diag_filled @ cpu_Vh
                npu_S_diag_filled = self.construct_S_diag_filled(full_matrices, npu_U, npu_S, npu_Vh)
                npu_A_reconstructed = npu_U @ npu_S_diag_filled @ npu_Vh

                if cpu_A_reconstructed.is_complex():
                    torch.testing.assert_close(cpu_A_reconstructed, npu_A_reconstructed, rtol=1e-3, atol=1e-3)
                else:
                    self.assertRtolEqual(cpu_A_reconstructed.numpy(), npu_A_reconstructed.numpy())

    def test_linalg_svd(self):
        dtype_list = [np.float32]
        self.exec_linalg_svd(dtype_list)

    def test_linalg_svd_fp64(self):
        dtype_list = [np.float64]
        self.exec_linalg_svd(dtype_list)

    def test_linalg_svd_complex(self):
        dtype_list = [np.complex64, np.complex128]
        self.exec_linalg_svd(dtype_list)

    def test_linalg_svd_complex_s_dtype(self):
        """torch.linalg.svd must return real S even for complex input, and the
        decomposition must actually reconstruct a truly complex input.
        aclnnSvd shares the input dtype for sigma, so the adapter must convert
        the complex temporary S to the corresponding real dtype."""
        cases = [(torch.complex64, torch.float32), (torch.complex128, torch.float64)]
        for in_dtype, expected_s_dtype in cases:
            for full_matrices in [True, False]:
                ctx = f"in_dtype={in_dtype} full_matrices={full_matrices}"
                # Build a genuinely complex input: pick a non-zero imaginary part
                # (create_common_tensor / astype would leave the imaginary part zero).
                real = torch.randn(3, 4, 2)
                imag = torch.randn(3, 4, 2)
                cpu_input = (real + 1j * imag).to(in_dtype)
                npu_input = cpu_input.npu()

                cpu_U, cpu_S, cpu_Vh = LA.svd(cpu_input, full_matrices=full_matrices)
                npu_U, npu_S, npu_Vh = LA.svd(npu_input, full_matrices=full_matrices)
                npu_U, npu_S, npu_Vh = npu_U.cpu(), npu_S.cpu(), npu_Vh.cpu()

                # dtype contract: S is real, U/Vh retain the complex dtype
                self.assertEqual(npu_S.dtype, expected_s_dtype)
                self.assertEqual(npu_U.dtype, in_dtype)
                self.assertEqual(npu_Vh.dtype, in_dtype)
                self.assertEqual(cpu_S.dtype, expected_s_dtype)

                # numeric check against the CPU reference (compare real, non-zero
                # singular values; avoids assertRtolEqual which rejects complex)
                torch.testing.assert_close(cpu_S, npu_S, rtol=1e-4, atol=1e-5)

                # reconstruction: U @ S_diag @ Vh must recover the complex input.
                # Convert the (real) diag_embed into the complex dtype so matmul is legal.
                k = min(2, 4)
                cpu_reconstructed = (cpu_U[..., :k] @ cpu_S.diag_embed().to(cpu_U.dtype)[..., :k, :k]) @ cpu_Vh[..., :k, :]
                npu_reconstructed = (npu_U[..., :k] @ npu_S.diag_embed().to(npu_U.dtype)[..., :k, :k]) @ npu_Vh[..., :k, :]
                torch.testing.assert_close(cpu_reconstructed, cpu_input, rtol=1e-2, atol=1e-3)
                torch.testing.assert_close(npu_reconstructed, npu_input.cpu(), rtol=1e-2, atol=1e-3, msg=f"NPU reconstruction failed ({ctx})")
                torch.testing.assert_close(cpu_reconstructed, npu_reconstructed, rtol=1e-2, atol=1e-3, msg=f"NPU vs CPU reconstruction mismatch ({ctx})")

                # empty input: S must still come back as the real dtype
                real_e = torch.randn(3, 0, 2)
                imag_e = torch.randn(3, 0, 2)
                cpu_empty = (real_e + 1j * imag_e).to(in_dtype)
                npu_empty = cpu_empty.npu()
                npu_eS = LA.svd(npu_empty, full_matrices=full_matrices)[1]
                self.assertEqual(npu_eS.dtype, expected_s_dtype)

    def test_linalg_svd_empty_and_zero_dim(self):
        """Test SVD with empty tensors and zero dimensions (aligned with PyTorch community test)"""
        # Reference: pytorch/test/test_linalg.py::test_svd
        # ns = (12, 4, 2, 0), batches = ((), (0,), (1,), (2,), (2, 1), (0, 2))
        dtype = torch.float32

        # Test various combinations of dimensions including 0
        ns = (4, 2, 0)
        batches = ((), (0,), (2,), (2, 1))

        for batch in batches:
            for m in ns:
                for n in ns:
                    shape = batch + (m, n)
                    k = min(m, n)

                    cpu_input = torch.randn(shape, dtype=dtype)
                    npu_input = cpu_input.npu()

                    for full_matrices in [True, False]:
                        cpu_U, cpu_S, cpu_Vh = LA.svd(cpu_input, full_matrices=full_matrices)
                        npu_U, npu_S, npu_Vh = LA.svd(npu_input, full_matrices=full_matrices)
                        npu_U_cpu = npu_U.cpu()
                        npu_S_cpu = npu_S.cpu()
                        npu_Vh_cpu = npu_Vh.cpu()

                        # Check shapes match
                        self.assertEqual(cpu_U.shape, npu_U_cpu.shape)
                        self.assertEqual(cpu_S.shape, npu_S_cpu.shape)
                        self.assertEqual(cpu_Vh.shape, npu_Vh_cpu.shape)

                        # For non-empty tensors, verify reconstruction
                        if cpu_input.numel() > 0:
                            cpu_reconstructed = (cpu_U[..., :k] @ cpu_S.diag_embed()) @ cpu_Vh[..., :k, :]
                            npu_reconstructed = (npu_U_cpu[..., :k] @ npu_S_cpu.diag_embed()) @ npu_Vh_cpu[..., :k, :]
                            self.assertRtolEqual(cpu_reconstructed.numpy(), npu_reconstructed.numpy())

                        # For empty tensors with full_matrices=True, verify identity matrix property
                        if cpu_input.numel() == 0 and full_matrices:
                            if cpu_U.numel() > 0:
                                self.assertRtolEqual(cpu_U.numpy(), npu_U_cpu.numpy())
                            if cpu_Vh.numel() > 0:
                                self.assertRtolEqual(cpu_Vh.numpy(), npu_Vh_cpu.numpy())


if __name__ == "__main__":
    run_tests()

import torch
import numpy as np
from torch import linalg as LA
import unittest

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices


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
        for item in dtype_shape_format:
            for full_matrices in full_matrices_list:
                cpu_input_1, npu_input_1 = create_common_tensor(item, -100, 100)
                cpu_U, cpu_S, cpu_Vh = self.cpu_out_exec(cpu_input_1, full_matrices)
                npu_U, npu_S, npu_Vh = self.npu_out_exec(npu_input_1, full_matrices)
                self.assertRtolEqual(cpu_U.abs().numpy(), npu_U.abs().numpy())
                self.assertRtolEqual(cpu_S.abs().numpy(), npu_S.abs().numpy())
                self.assertRtolEqual(cpu_Vh.abs().numpy(), npu_Vh.abs().numpy())

                cpu_S_diag_filled = self.construct_S_diag_filled(full_matrices, cpu_U, cpu_S, cpu_Vh)
                cpu_A_reconstructed = cpu_U @ cpu_S_diag_filled @ cpu_Vh
                npu_S_diag_filled = self.construct_S_diag_filled(full_matrices, npu_U, npu_S, npu_Vh)
                npu_A_reconstructed = npu_U @ npu_S_diag_filled @ npu_Vh

                self.assertRtolEqual(cpu_A_reconstructed.numpy(), npu_A_reconstructed.numpy())

    def test_linalg_svd(self):
        dtype_list = [np.float32]
        self.exec_linalg_svd(dtype_list)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_linalg_svd_fp64(self):
        dtype_list = [np.float64]
        self.exec_linalg_svd(dtype_list)

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

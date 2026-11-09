import unittest

import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


def random_pd_matrix(n, batch, dtype=torch.float32, device="cpu", ridge=0.5):
    """Build a strictly positive-definite symmetric matrix: B @ B^T + ridge * I (0/1/2-dim batch)."""
    b = torch.randn(*batch, n, n, dtype=torch.float32)
    eye = torch.eye(n, dtype=torch.float32)
    return (b @ b.transpose(-2, -1) + ridge * eye).to(dtype).to(device)


# torch.linalg.matrix_sqrth was introduced in torch 2.14; this repo also maintains the
# v2.1~v2.16 branches where the API does not exist, so skip at class level via hasattr
# (more robust than comparing version numbers).
@unittest.skipIf(not hasattr(torch.linalg, "matrix_sqrth"), "requires torch>=2.14")
class TestLinalgMatrixSqrth(TestCase):
    def _assert_symmetric(self, tensor):
        self.assertTrue(torch.equal(tensor.cpu(), tensor.transpose(-2, -1).cpu()))

    def test_linalg_matrix_sqrth(self):
        for n in (1, 5):
            for batch in ((), (3,), (2, 2)):
                a_cpu = random_pd_matrix(n, batch)
                a_npu = a_cpu.npu()
                x_cpu = torch.linalg.matrix_sqrth(a_cpu)
                x_npu = torch.linalg.matrix_sqrth(a_npu)
                # Exactly symmetric output (the kernel symmetrizes at the end, bitwise equal)
                self._assert_symmetric(x_npu)
                # Compare with the CPU reference implementation (eigh path)
                self.assertRtolEqual(x_cpu.numpy(), x_npu.cpu().numpy(), prec=2e-4)
                # Defining property: X @ X == A
                self.assertRtolEqual(
                    (x_cpu @ x_cpu).numpy(), (x_npu @ x_npu).cpu().numpy(), prec=2e-4)

    def test_linalg_matrix_sqrth_psd_boundary(self):
        # Aligned with upstream semantics: singular PSD input is accepted and zero/negative
        # eigenvalues are clamped to 0.
        # The diag(-1e-6, 1, 4) case pins down "restore the sign of the SVD negative eigenvalue
        # before clamping": taking sqrt of the non-negative sigma directly would produce a -1e-3
        # component, off by 1e-3 from CPU, which this case catches.
        torch.manual_seed(0)
        q, _ = torch.linalg.qr(torch.randn(4, 4))
        repeated = (q * torch.tensor([2.0, 2.0, 5.0, 5.0])) @ q.transpose(-2, -1)
        # Indefinite, full rank and with distinct |lambda|: only summing along dim=-2
        # (diag(U^T V)) recovers the signs, while the diagonal sample below cannot tell the two
        # axes apart (row and column sums match). Keeping it full rank avoids the null-space noise
        # of a rank-deficient case, whose sqrt(eps) contribution is comparable to the 2e-4
        # cross-backend tolerance.
        indefinite = (q * torch.tensor([3.0, -1.0, 0.5, 2.0])) @ q.transpose(-2, -1)
        indefinite = 0.5 * (indefinite + indefinite.transpose(-2, -1))
        cases = (
            torch.diag(torch.tensor([0.0, 4.0, 9.0])),
            0.5 * (repeated + repeated.transpose(-2, -1)),
            torch.diag(torch.tensor([-1e-6, 1.0, 4.0])),
            indefinite,
        )
        for item in cases:
            a_npu = item.npu()
            x_npu = torch.linalg.matrix_sqrth(a_npu)
            self.assertTrue(torch.isfinite(x_npu).all().item())
            self._assert_symmetric(x_npu)
            self.assertRtolEqual(
                torch.linalg.matrix_sqrth(item).numpy(), x_npu.cpu().numpy(), prec=2e-4)

    def test_linalg_matrix_sqrth_empty(self):
        for shape in ((2, 0, 0), (0, 3, 3)):
            a_npu = torch.empty(shape, device="npu", dtype=torch.float32)
            x_npu = torch.linalg.matrix_sqrth(a_npu)
            self.assertEqual(x_npu.shape, a_npu.shape)
            self.assertEqual(x_npu.numel(), 0)

    def test_linalg_matrix_sqrth_out(self):
        # Upstream uses an autogen out (native_functions.yaml: `autogen: linalg_matrix_sqrth.out`),
        # so no out= argument is generated on the Python side and the .out variant has to be
        # called at the aten level.
        a_cpu = random_pd_matrix(4, (2,))
        a_npu = a_cpu.npu()
        out_npu = torch.empty_like(a_npu)
        torch.ops.aten.linalg_matrix_sqrth.out(a_npu, out=out_npu)
        self.assertRtolEqual(
            torch.linalg.matrix_sqrth(a_cpu).numpy(), out_npu.cpu().numpy(), prec=2e-4)

    def test_linalg_matrix_sqrth_dtype(self):
        # Build the low/high precision tensors on CPU and move them to NPU to avoid dtype
        # conversions on NPU (some SoC/CANN versions do not support fp64 tensors, which would make
        # the case fail during construction with a different error).
        # Half precision: same as CPU/CUDA, identical error message.
        with self.assertRaisesRegex(RuntimeError, "Low precision dtypes not supported. Got Half"):
            torch.linalg.matrix_sqrth(torch.randn(3, 3).half().npu())
        # float64: supported upstream, but the NPU reconstruction goes through matmul and
        # aclnnMatmul only supports fp16/bf16/fp32, so fail explicitly instead of silently falling
        # back to the CPU.
        with self.assertRaisesRegex(RuntimeError, "not supported on the NPU backend yet"):
            torch.linalg.matrix_sqrth(torch.randn(3, 3).double().npu())

    def test_linalg_matrix_sqrth_not_square(self):
        with self.assertRaisesRegex(RuntimeError, "must be batches of square matrices"):
            torch.linalg.matrix_sqrth(torch.randn(2, 3, device="npu"))

    def test_linalg_matrix_sqrth_too_many_dims(self):
        # aclnnSvd accepts only 2-8 dims; report it here instead of letting aclnn fail later.
        # The 9-dim input is built with a view: creating it directly on NPU (torch.randn) goes
        # through aclnnInplaceNormal, which itself rejects tensors with more than 8 dims.
        a_npu = torch.randn(4, 4, device="npu").view(1, 1, 1, 1, 1, 1, 1, 4, 4)
        with self.assertRaisesRegex(RuntimeError, "at most 8 dimensions"):
            torch.linalg.matrix_sqrth(a_npu)

    def test_linalg_matrix_sqrth_backward(self):
        # The backward is bound by op-plugin's derivatives.yaml to the custom op
        # npu_linalg_matrix_sqrth_backward (SVD + Loewner formula, no eigh dependency).
        # Note: an asymmetric upstream gradient is required here to cover the
        # grad_sym = 0.5 * (grad + grad^T) step in the formula (ones would bypass it).
        torch.manual_seed(0)
        for batch in ((), (2,)):
            a_cpu = random_pd_matrix(4, batch).requires_grad_(True)
            a_npu = a_cpu.detach().clone().npu().requires_grad_(True)
            g = torch.randn(*batch, 4, 4)
            torch.linalg.matrix_sqrth(a_cpu).backward(g)
            torch.linalg.matrix_sqrth(a_npu).backward(g.npu())
            self.assertTrue(torch.isfinite(a_npu.grad).all().item())
            self.assertRtolEqual(a_cpu.grad.numpy(), a_npu.grad.cpu().numpy(), prec=1e-3)

    def test_linalg_matrix_sqrth_jvp(self):
        # derivatives.yaml also binds a `result:` formula and upstream OpInfo declares
        # supports_forward_ad=True, so forward-mode AD has to work as well. The Loewner operator is
        # self-adjoint, hence one kernel serves both VJP and JVP; a broken forward-grad redispatch
        # would only surface here. The tangent is asymmetric on purpose to cover grad_sym.
        torch.manual_seed(0)
        for batch in ((), (2,)):
            a_cpu = random_pd_matrix(4, batch)
            t_cpu = torch.randn(*batch, 4, 4)
            primal_cpu, tangent_cpu = torch.func.jvp(
                torch.linalg.matrix_sqrth, (a_cpu,), (t_cpu,))
            primal_npu, tangent_npu = torch.func.jvp(
                torch.linalg.matrix_sqrth, (a_cpu.npu(),), (t_cpu.npu(),))
            self.assertTrue(torch.isfinite(tangent_npu).all().item())
            self.assertRtolEqual(primal_cpu.numpy(), primal_npu.cpu().numpy(), prec=2e-4)
            self.assertRtolEqual(tangent_cpu.numpy(), tangent_npu.cpu().numpy(), prec=1e-3)


if __name__ == "__main__":
    run_tests()

import os
os.environ["TORCH_NPU_USE_COMPATIBLE_IMPL"] = "1"

import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests

torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False


class TestBmmCompatible(TestCase):
    """Test bmm when input tensors are broadcast-expanded (stride(0)==0).

    In compatible mode, _matmul_impl expands tensors for batch broadcasting
    before calling bmm, making them non-contiguous. The BmmKernelNpuOpApi
    restores them to pre-broadcast form to avoid redundant CANN ops.
    """

    def test_bmm_self_broadcast_expanded(self):
        """self is broadcast-expanded: [1,m,k] -> [b,m,k], stride(0)==0"""
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            b, m, k, n = 4, 8, 16, 12
            cpu_self_base = torch.randn(1, m, k, dtype=dtype)
            cpu_mat2 = torch.randn(b, k, n, dtype=dtype)

            cpu_self = cpu_self_base.expand(b, m, k)
            npu_self = cpu_self_base.npu().expand(b, m, k)
            npu_mat2 = cpu_mat2.npu()

            self.assertEqual(npu_self.stride(0), 0)

            if dtype in (torch.float16, torch.bfloat16):
                cpu_compute_self = cpu_self.float()
                cpu_compute_mat2 = cpu_mat2.float()
            else:
                cpu_compute_self = cpu_self
                cpu_compute_mat2 = cpu_mat2
            cpu_out = torch.bmm(cpu_compute_self, cpu_compute_mat2)
            npu_out = torch.bmm(npu_self, npu_mat2).cpu()
            cpu_out = cpu_out.to(npu_out.dtype)

            if dtype == torch.bfloat16:
                self.assertRtolEqual(cpu_out.float().numpy(), npu_out.float().numpy(), prec=0.001)
            else:
                self.assertRtolEqual(cpu_out.numpy(), npu_out.numpy())

    def test_bmm_mat2_broadcast_expanded(self):
        """mat2 is broadcast-expanded: [1,k,n] -> [b,k,n], stride(0)==0"""
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            b, m, k, n = 4, 8, 16, 12
            cpu_self = torch.randn(b, m, k, dtype=dtype)
            cpu_mat2_base = torch.randn(1, k, n, dtype=dtype)

            cpu_mat2 = cpu_mat2_base.expand(b, k, n)
            npu_self = cpu_self.npu()
            npu_mat2 = cpu_mat2_base.npu().expand(b, k, n)

            self.assertEqual(npu_mat2.stride(0), 0)

            if dtype in (torch.float16, torch.bfloat16):
                cpu_compute_self = cpu_self.float()
                cpu_compute_mat2 = cpu_mat2.float()
            else:
                cpu_compute_self = cpu_self
                cpu_compute_mat2 = cpu_mat2
            cpu_out = torch.bmm(cpu_compute_self, cpu_compute_mat2)
            npu_out = torch.bmm(npu_self, npu_mat2).cpu()
            cpu_out = cpu_out.to(npu_out.dtype)

            if dtype == torch.bfloat16:
                self.assertRtolEqual(cpu_out.float().numpy(), npu_out.float().numpy(), prec=0.001)
            else:
                self.assertRtolEqual(cpu_out.numpy(), npu_out.numpy())

    def test_bmm_both_broadcast_expanded(self):
        """Both self and mat2 are broadcast-expanded from different batch sizes"""
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            b, m, k, n = 8, 6, 10, 14
            cpu_self_base = torch.randn(1, m, k, dtype=dtype)
            cpu_mat2_base = torch.randn(1, k, n, dtype=dtype)

            cpu_self = cpu_self_base.expand(b, m, k)
            cpu_mat2 = cpu_mat2_base.expand(b, k, n)
            npu_self = cpu_self_base.npu().expand(b, m, k)
            npu_mat2 = cpu_mat2_base.npu().expand(b, k, n)

            self.assertEqual(npu_self.stride(0), 0)
            self.assertEqual(npu_mat2.stride(0), 0)

            if dtype in (torch.float16, torch.bfloat16):
                cpu_compute_self = cpu_self.float()
                cpu_compute_mat2 = cpu_mat2.float()
            else:
                cpu_compute_self = cpu_self
                cpu_compute_mat2 = cpu_mat2
            cpu_out = torch.bmm(cpu_compute_self, cpu_compute_mat2)
            npu_out = torch.bmm(npu_self, npu_mat2).cpu()
            cpu_out = cpu_out.to(npu_out.dtype)

            if dtype == torch.bfloat16:
                self.assertRtolEqual(cpu_out.float().numpy(), npu_out.float().numpy(), prec=0.001)
            else:
                self.assertRtolEqual(cpu_out.numpy(), npu_out.numpy())

    def test_bmm_zero_batch(self):
        """Both self and mat2 have batch=0: [0,m,k] @ [0,k,n]"""
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            m, k, n = 8, 16, 12
            cpu_self = torch.randn(0, m, k, dtype=dtype)
            cpu_mat2 = torch.randn(0, k, n, dtype=dtype)

            npu_self = cpu_self.npu()
            npu_mat2 = cpu_mat2.npu()

            if dtype in (torch.float16, torch.bfloat16):
                cpu_compute_self = cpu_self.float()
                cpu_compute_mat2 = cpu_mat2.float()
            else:
                cpu_compute_self = cpu_self
                cpu_compute_mat2 = cpu_mat2
            cpu_out = torch.bmm(cpu_compute_self, cpu_compute_mat2)
            npu_out = torch.bmm(npu_self, npu_mat2).cpu()
            cpu_out = cpu_out.to(npu_out.dtype)

            if dtype == torch.bfloat16:
                self.assertRtolEqual(cpu_out.float().numpy(), npu_out.float().numpy(), prec=0.001)
            else:
                self.assertRtolEqual(cpu_out.numpy(), npu_out.numpy())

    def test_bmm_broadcast_backward(self):
        """Broadcast-expanded bmm with backward pass"""
        for dtype in [torch.float32, torch.float16]:
            b, m, k, n = 4, 8, 16, 12
            cpu_self_base = torch.randn(1, m, k, dtype=dtype)
            cpu_mat2_base = torch.randn(b, k, n, dtype=dtype)

            if dtype == torch.float32:
                cpu_self = cpu_self_base.clone().requires_grad_(True)
                cpu_mat2 = cpu_mat2_base.clone().requires_grad_(True)
            else:
                cpu_self = cpu_self_base.float().requires_grad_(True)
                cpu_mat2 = cpu_mat2_base.float().requires_grad_(True)
            cpu_self_expanded = cpu_self.expand(b, m, k)

            npu_self = cpu_self_base.npu().requires_grad_(True)
            npu_mat2 = cpu_mat2_base.npu().requires_grad_(True)
            npu_self_expanded = npu_self.expand(b, m, k)

            self.assertEqual(npu_self_expanded.stride(0), 0)

            cpu_out = torch.bmm(cpu_self_expanded, cpu_mat2)
            npu_out = torch.bmm(npu_self_expanded, npu_mat2)

            cpu_out.backward(torch.ones_like(cpu_out))
            npu_out.backward(torch.ones_like(npu_out))

            cpu_out = cpu_out.detach().to(npu_out.dtype)
            cpu_self_grad = cpu_self.grad.to(npu_self.grad.dtype)
            cpu_mat2_grad = cpu_mat2.grad.to(npu_mat2.grad.dtype)

            self.assertRtolEqual(cpu_out.numpy(), npu_out.detach().cpu().numpy())
            self.assertRtolEqual(cpu_self_grad.numpy(), npu_self.grad.cpu().numpy(), prec16=0.005)
            self.assertRtolEqual(cpu_mat2_grad.numpy(), npu_mat2.grad.cpu().numpy(), prec16=0.005)

    def test_bmm_batch1_non_standard_stride(self):
        """batch=1 with non-standard stride(0) is normalized to avoid Transpose.

        _matmul_impl's reshape may set stride(0) to stride(1)*size(1) instead of
        size(1)*size(2) for batch=1 tensors, which makes CANN insert a redundant
        Transpose. This simulates that via as_strided and verifies correctness.
        """
        for dtype in [torch.float32, torch.float16]:
            m, k, n = 16, 16, 64
            # Transpose to get matrix-dim transposed layout, then override only
            # stride(0) to stride(1)*size(1) (simulating _matmul_impl's reshape).
            # stride(1)/stride(2) must be preserved from the transposed tensor.
            a_std = torch.randn(1, k, m, dtype=dtype).transpose(1, 2)  # [1,m,k] stride [m*k,1,m]
            a = a_std.as_strided([1, m, k],
                                 [a_std.stride(1) * m, a_std.stride(1), a_std.stride(2)])
            b_std = torch.randn(1, n, k, dtype=dtype).transpose(1, 2)  # [1,k,n] stride [k*n,1,k]
            b = b_std.as_strided([1, k, n],
                                 [b_std.stride(1) * k, b_std.stride(1), b_std.stride(2)])

            npu_a = a.npu()
            npu_b = b.npu()

            if dtype == torch.float16:
                cpu_out = torch.bmm(a.float(), b.float())
            else:
                cpu_out = torch.bmm(a, b)
            npu_out = torch.bmm(npu_a, npu_b).cpu()
            cpu_out = cpu_out.to(npu_out.dtype)

            self.assertRtolEqual(cpu_out.numpy(), npu_out.numpy(), prec=0.01)


if __name__ == "__main__":
    np.random.seed(1234)
    run_tests()

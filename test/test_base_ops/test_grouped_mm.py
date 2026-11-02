import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

# _grouped_mm is available since PyTorch 2.8, skip on older versions
SKIP_TESTS = not hasattr(torch, "_grouped_mm")

@unittest.skipIf(SKIP_TESTS, "_grouped_mm not available in this PyTorch version")
class TestGroupedMm(TestCase):
    # =====================================================
    # CPU reference implementations (mm-loop fallback)
    # =====================================================

    def cpu_ref_2d_x_3d(self, mat_a, mat_b, offs):
        """2D x 3D: (M, K) x (G, K, N), offs(G,) -> (M, N)"""
        outs = []
        start = 0
        for g in range(offs.size(0)):
            end = offs[g].item()
            outs.append(torch.mm(mat_a[start:end, :], mat_b[g]))
            start = end
        return torch.cat(outs, dim=0)

    def cpu_ref_3d_x_2d(self, mat_a, mat_b, offs):
        """3D x 2D: (G, M, K) x (K, N), offs(G,) -> (M, N)"""
        outs = []
        start = 0
        for g in range(offs.size(0)):
            end = offs[g].item()
            outs.append(torch.mm(mat_a[g], mat_b[:, start:end]))
            start = end
        return torch.cat(outs, dim=1)

    def cpu_ref_2d_x_2d(self, mat_a, mat_b, offs):
        """2D x 2D: (M, K) x (K, N), offs(G,) -> (G, M, N)"""
        outs = []
        start = 0
        for g in range(offs.size(0)):
            end = offs[g].item()
            outs.append(torch.mm(mat_a[:, start:end], mat_b[start:end, :]).unsqueeze(0))
            start = end
        return torch.cat(outs, dim=0)

    def cpu_ref_3d_x_3d(self, mat_a, mat_b):
        """3D x 3D: (G, M, K) x (G, K, N) -> (G, M, N)"""
        return torch.bmm(mat_a, mat_b)

    def generate_group_list(self, total, num_groups):
        """Generate cumulative group boundaries (int32, last value = total)."""
        step = total // num_groups
        data = [(i + 1) * step for i in range(num_groups)]
        data[-1] = total
        return torch.tensor(data, dtype=torch.int32)

    # =====================================================
    # Scenario 1: 2D x 3D (MoE forward)
    # =====================================================

    def test_2d_x_3d_fp16(self, device="npu"):
        """2D x 3D, fp16, 3 experts"""
        torch.manual_seed(0)
        M, K, N, G = 768, 256, 128, 3
        mat_a = torch.normal(mean=0., std=0.1, size=(M, K), dtype=torch.float16)
        mat_b = torch.normal(mean=0., std=0.1, size=(G, K, N), dtype=torch.float16)
        offs = self.generate_group_list(M, G)

        cpu_out = self.cpu_ref_2d_x_3d(mat_a, mat_b, offs)
        npu_out = torch._grouped_mm(mat_a.npu(), mat_b.npu(), offs=offs.npu())
        self.assertRtolEqual(cpu_out, npu_out.cpu(), 0.001)

    def test_2d_x_3d_bf16(self, device="npu"):
        """2D x 3D, bf16, 4 experts. K and rtol aligned with native
        test_grouped_gemm_cublaslt (jagged_size=32, bf16 default rtol=0.016)."""
        torch.manual_seed(0)
        M, K, N, G = 1024, 32, 256, 4
        mat_a = torch.normal(mean=0., std=0.1, size=(M, K), dtype=torch.bfloat16)
        mat_b = torch.normal(mean=0., std=0.1, size=(G, K, N), dtype=torch.bfloat16)
        offs = self.generate_group_list(M, G)

        cpu_out = self.cpu_ref_2d_x_3d(mat_a, mat_b, offs)
        npu_out = torch._grouped_mm(mat_a.npu(), mat_b.npu(), offs=offs.npu())
        self.assertRtolEqual(cpu_out, npu_out.cpu(), 0.016)

    def test_2d_x_3d_single_group(self, device="npu"):
        """2D x 3D, fp16, single group (G=1)"""
        torch.manual_seed(0)
        M, K, N, G = 256, 128, 64, 1
        mat_a = torch.normal(mean=0., std=0.1, size=(M, K), dtype=torch.float16)
        mat_b = torch.normal(mean=0., std=0.1, size=(G, K, N), dtype=torch.float16)
        offs = self.generate_group_list(M, G)

        cpu_out = self.cpu_ref_2d_x_3d(mat_a, mat_b, offs)
        npu_out = torch._grouped_mm(mat_a.npu(), mat_b.npu(), offs=offs.npu())
        self.assertRtolEqual(cpu_out, npu_out.cpu(), 0.001)

    def test_2d_x_3d_backward(self, device="npu"):
        """2D x 3D, fp16, backward gradients"""
        torch.manual_seed(0)
        M, K, N, G = 512, 256, 128, 3
        mat_a = torch.normal(mean=0., std=0.1, size=(M, K), dtype=torch.float16, requires_grad=True)
        mat_b = torch.normal(mean=0., std=0.1, size=(G, K, N), dtype=torch.float16, requires_grad=True)
        offs = self.generate_group_list(M, G)

        # NPU forward + backward
        npu_out = torch._grouped_mm(mat_a.npu(), mat_b.npu(), offs=offs.npu())
        npu_loss = npu_out.sum()
        npu_loss.backward()
        npu_grad_a = mat_a.grad.cpu()
        npu_grad_b = mat_b.grad.cpu()

        # CPU reference forward + backward
        mat_a_cpu = mat_a.detach().clone().requires_grad_(True)
        mat_b_cpu = mat_b.detach().clone().requires_grad_(True)
        cpu_out = self.cpu_ref_2d_x_3d(mat_a_cpu, mat_b_cpu, offs)
        cpu_out.sum().backward()

        self.assertRtolEqual(cpu_out, npu_out.cpu(), 0.001)
        self.assertRtolEqual(mat_a_cpu.grad, npu_grad_a, 0.001)
        self.assertRtolEqual(mat_b_cpu.grad, npu_grad_b, 0.001)

    # =====================================================
    # Scenario 2: 3D x 2D (backward mat2 gradient)
    # =====================================================

    def test_3d_x_2d_fp16(self, device="npu"):
        """3D x 2D, fp16, 3 groups"""
        torch.manual_seed(0)
        G, M, K, N = 3, 256, 128, 768
        mat_a = torch.normal(mean=0., std=0.1, size=(G, M, K), dtype=torch.float16)
        mat_b = torch.normal(mean=0., std=0.1, size=(K, N), dtype=torch.float16)
        offs = self.generate_group_list(N, G)

        cpu_out = self.cpu_ref_3d_x_2d(mat_a, mat_b, offs)
        npu_out = torch._grouped_mm(mat_a.npu(), mat_b.npu(), offs=offs.npu())
        self.assertRtolEqual(cpu_out, npu_out.cpu(), 0.001)

    def test_3d_x_2d_bf16(self, device="npu"):
        """3D x 2D, bf16, 4 groups. K and rtol aligned with native."""
        torch.manual_seed(0)
        G, M, K, N = 4, 128, 32, 1024
        mat_a = torch.normal(mean=0., std=0.1, size=(G, M, K), dtype=torch.bfloat16)
        mat_b = torch.normal(mean=0., std=0.1, size=(K, N), dtype=torch.bfloat16)
        offs = self.generate_group_list(N, G)

        cpu_out = self.cpu_ref_3d_x_2d(mat_a, mat_b, offs)
        npu_out = torch._grouped_mm(mat_a.npu(), mat_b.npu(), offs=offs.npu())
        self.assertRtolEqual(cpu_out, npu_out.cpu(), 0.016)

    def test_3d_x_2d_single_group(self, device="npu"):
        """3D x 2D, fp16, single group (G=1)"""
        torch.manual_seed(0)
        G, M, K, N = 1, 128, 64, 256
        mat_a = torch.normal(mean=0., std=0.1, size=(G, M, K), dtype=torch.float16)
        mat_b = torch.normal(mean=0., std=0.1, size=(K, N), dtype=torch.float16)
        offs = self.generate_group_list(N, G)

        cpu_out = self.cpu_ref_3d_x_2d(mat_a, mat_b, offs)
        npu_out = torch._grouped_mm(mat_a.npu(), mat_b.npu(), offs=offs.npu())
        self.assertRtolEqual(cpu_out, npu_out.cpu(), 0.001)

    def test_3d_x_2d_backward(self, device="npu"):
        """3D x 2D, fp16, backward gradients"""
        torch.manual_seed(0)
        G, M, K, N = 3, 128, 128, 512
        mat_a = torch.normal(mean=0., std=0.1, size=(G, M, K), dtype=torch.float16, requires_grad=True)
        mat_b = torch.normal(mean=0., std=0.1, size=(K, N), dtype=torch.float16, requires_grad=True)
        offs = self.generate_group_list(N, G)

        npu_out = torch._grouped_mm(mat_a.npu(), mat_b.npu(), offs=offs.npu())
        npu_loss = npu_out.sum()
        npu_loss.backward()
        npu_grad_a = mat_a.grad.cpu()
        npu_grad_b = mat_b.grad.cpu()

        mat_a_cpu = mat_a.detach().clone().requires_grad_(True)
        mat_b_cpu = mat_b.detach().clone().requires_grad_(True)
        cpu_out = self.cpu_ref_3d_x_2d(mat_a_cpu, mat_b_cpu, offs)
        cpu_out.sum().backward()

        self.assertRtolEqual(cpu_out, npu_out.cpu(), 0.001)
        self.assertRtolEqual(mat_a_cpu.grad, npu_grad_a, 0.001)
        self.assertRtolEqual(mat_b_cpu.grad, npu_grad_b, 0.001)

    # =====================================================
    # Scenario 3: 2D x 2D (backward mat1 gradient)
    # =====================================================

    def test_2d_x_2d_fp16(self, device="npu"):
        """2D x 2D, fp16, 3 groups"""
        torch.manual_seed(0)
        M, K, N, G = 256, 768, 128, 3
        mat_a = torch.normal(mean=0., std=0.1, size=(M, K), dtype=torch.float16)
        mat_b = torch.normal(mean=0., std=0.1, size=(K, N), dtype=torch.float16)
        offs = self.generate_group_list(K, G)

        cpu_out = self.cpu_ref_2d_x_2d(mat_a, mat_b, offs)
        npu_out = torch._grouped_mm(mat_a.npu(), mat_b.npu(), offs=offs.npu())
        self.assertRtolEqual(cpu_out, npu_out.cpu(), 0.001)

    def test_2d_x_2d_bf16(self, device="npu"):
        """2D x 2D, bf16, 4 groups. K and rtol aligned with native."""
        torch.manual_seed(0)
        M, K, N, G = 128, 32, 256, 4
        mat_a = torch.normal(mean=0., std=0.1, size=(M, K), dtype=torch.bfloat16)
        mat_b = torch.normal(mean=0., std=0.1, size=(K, N), dtype=torch.bfloat16)
        offs = self.generate_group_list(K, G)

        cpu_out = self.cpu_ref_2d_x_2d(mat_a, mat_b, offs)
        npu_out = torch._grouped_mm(mat_a.npu(), mat_b.npu(), offs=offs.npu())
        self.assertRtolEqual(cpu_out, npu_out.cpu(), 0.016)

    def test_2d_x_2d_single_group(self, device="npu"):
        """2D x 2D, fp16, single group (G=1)"""
        torch.manual_seed(0)
        M, K, N, G = 128, 256, 64, 1
        mat_a = torch.normal(mean=0., std=0.1, size=(M, K), dtype=torch.float16)
        mat_b = torch.normal(mean=0., std=0.1, size=(K, N), dtype=torch.float16)
        offs = self.generate_group_list(K, G)

        cpu_out = self.cpu_ref_2d_x_2d(mat_a, mat_b, offs)
        npu_out = torch._grouped_mm(mat_a.npu(), mat_b.npu(), offs=offs.npu())
        self.assertRtolEqual(cpu_out, npu_out.cpu(), 0.001)

    def test_2d_x_2d_backward(self, device="npu"):
        """2D x 2D, fp16, backward gradients"""
        torch.manual_seed(0)
        M, K, N, G = 128, 512, 128, 3
        mat_a = torch.normal(mean=0., std=0.1, size=(M, K), dtype=torch.float16, requires_grad=True)
        mat_b = torch.normal(mean=0., std=0.1, size=(K, N), dtype=torch.float16, requires_grad=True)
        offs = self.generate_group_list(K, G)

        npu_out = torch._grouped_mm(mat_a.npu(), mat_b.npu(), offs=offs.npu())
        npu_loss = npu_out.sum()
        npu_loss.backward()
        npu_grad_a = mat_a.grad.cpu()
        npu_grad_b = mat_b.grad.cpu()

        mat_a_cpu = mat_a.detach().clone().requires_grad_(True)
        mat_b_cpu = mat_b.detach().clone().requires_grad_(True)
        cpu_out = self.cpu_ref_2d_x_2d(mat_a_cpu, mat_b_cpu, offs)
        cpu_out.sum().backward()

        self.assertRtolEqual(cpu_out, npu_out.cpu(), 0.001)
        self.assertRtolEqual(mat_a_cpu.grad, npu_grad_a, 0.001)
        self.assertRtolEqual(mat_b_cpu.grad, npu_grad_b, 0.001)

    # =====================================================
    # Scenario 4: 3D x 3D (standard BMM)
    # =====================================================

    def test_3d_x_3d_fp16(self, device="npu"):
        """3D x 3D, fp16, 4 groups"""
        torch.manual_seed(0)
        G, M, K, N = 4, 128, 256, 64
        mat_a = torch.normal(mean=0., std=0.1, size=(G, M, K), dtype=torch.float16)
        mat_b = torch.normal(mean=0., std=0.1, size=(G, K, N), dtype=torch.float16)

        cpu_out = self.cpu_ref_3d_x_3d(mat_a, mat_b)
        npu_out = torch._grouped_mm(mat_a.npu(), mat_b.npu())
        self.assertRtolEqual(cpu_out, npu_out.cpu(), 0.001)

    def test_3d_x_3d_bf16(self, device="npu"):
        """3D x 3D, bf16, 3 groups. rtol aligned with native bf16 default (0.016)."""
        torch.manual_seed(0)
        G, M, K, N = 3, 64, 128, 256
        mat_a = torch.normal(mean=0., std=0.1, size=(G, M, K), dtype=torch.bfloat16)
        mat_b = torch.normal(mean=0., std=0.1, size=(G, K, N), dtype=torch.bfloat16)

        cpu_out = self.cpu_ref_3d_x_3d(mat_a, mat_b)
        npu_out = torch._grouped_mm(mat_a.npu(), mat_b.npu())
        self.assertRtolEqual(cpu_out, npu_out.cpu(), 0.016)

    def test_3d_x_3d_single_group(self, device="npu"):
        """3D x 3D, fp16, single group (G=1)"""
        torch.manual_seed(0)
        G, M, K, N = 1, 64, 128, 64
        mat_a = torch.normal(mean=0., std=0.1, size=(G, M, K), dtype=torch.float16)
        mat_b = torch.normal(mean=0., std=0.1, size=(G, K, N), dtype=torch.float16)

        cpu_out = self.cpu_ref_3d_x_3d(mat_a, mat_b)
        npu_out = torch._grouped_mm(mat_a.npu(), mat_b.npu())
        self.assertRtolEqual(cpu_out, npu_out.cpu(), 0.001)

    def test_3d_x_3d_backward(self, device="npu"):
        """3D x 3D, fp16, backward gradients"""
        torch.manual_seed(0)
        G, M, K, N = 3, 64, 128, 64
        mat_a = torch.normal(mean=0., std=0.1, size=(G, M, K), dtype=torch.float16, requires_grad=True)
        mat_b = torch.normal(mean=0., std=0.1, size=(G, K, N), dtype=torch.float16, requires_grad=True)

        npu_out = torch._grouped_mm(mat_a.npu(), mat_b.npu())
        npu_loss = npu_out.sum()
        npu_loss.backward()
        npu_grad_a = mat_a.grad.cpu()
        npu_grad_b = mat_b.grad.cpu()

        mat_a_cpu = mat_a.detach().clone().requires_grad_(True)
        mat_b_cpu = mat_b.detach().clone().requires_grad_(True)
        cpu_out = self.cpu_ref_3d_x_3d(mat_a_cpu, mat_b_cpu)
        cpu_out.sum().backward()

        self.assertRtolEqual(cpu_out, npu_out.cpu(), 0.001)
        self.assertRtolEqual(mat_a_cpu.grad, npu_grad_a, 0.001)
        self.assertRtolEqual(mat_b_cpu.grad, npu_grad_b, 0.001)

    # =====================================================
    # Output shape verification
    # =====================================================

    def test_output_shape(self, device="npu"):
        """Verify output shapes for all 4 scenarios"""
        torch.manual_seed(0)

        # Scenario 1: 2D x 3D -> (M, N)
        M, K, N, G = 512, 256, 128, 3
        mat_a = torch.randn(M, K, dtype=torch.float16).npu()
        mat_b = torch.randn(G, K, N, dtype=torch.float16).npu()
        offs = self.generate_group_list(M, G).npu()
        out = torch._grouped_mm(mat_a, mat_b, offs=offs)
        self.assertEqual(out.shape, torch.Size([M, N]))

        # Scenario 2: 3D x 2D -> (M, N)
        G, M, K, N = 3, 256, 128, 768
        mat_a = torch.randn(G, M, K, dtype=torch.float16).npu()
        mat_b = torch.randn(K, N, dtype=torch.float16).npu()
        offs = self.generate_group_list(N, G).npu()
        out = torch._grouped_mm(mat_a, mat_b, offs=offs)
        self.assertEqual(out.shape, torch.Size([M, N]))

        # Scenario 3: 2D x 2D -> (G, M, N)
        M, K, N, G = 256, 768, 128, 3
        mat_a = torch.randn(M, K, dtype=torch.float16).npu()
        mat_b = torch.randn(K, N, dtype=torch.float16).npu()
        offs = self.generate_group_list(K, G).npu()
        out = torch._grouped_mm(mat_a, mat_b, offs=offs)
        self.assertEqual(out.shape, torch.Size([G, M, N]))

        # Scenario 4: 3D x 3D -> (G, M, N)
        G, M, K, N = 4, 128, 256, 64
        mat_a = torch.randn(G, M, K, dtype=torch.float16).npu()
        mat_b = torch.randn(G, K, N, dtype=torch.float16).npu()
        out = torch._grouped_mm(mat_a, mat_b)
        self.assertEqual(out.shape, torch.Size([G, M, N]))

    # =====================================================
    # Negative tests
    # =====================================================

    def test_neg_dtype_mismatch(self, device="npu"):
        """Negative: mat_a and mat_b have different dtypes"""
        mat_a = torch.randn(64, 32, dtype=torch.float16).npu()
        mat_b = torch.randn(2, 32, 16, dtype=torch.bfloat16).npu()
        offs = torch.tensor([32, 64], dtype=torch.int32).npu()
        with self.assertRaisesRegex(RuntimeError, r"same dtype"):
            torch._grouped_mm(mat_a, mat_b, offs=offs)

    def test_neg_invalid_dim(self, device="npu"):
        """Negative: mat_a is 1D"""
        mat_a = torch.randn(64, dtype=torch.float16).npu()
        mat_b = torch.randn(2, 64, 32, dtype=torch.float16).npu()
        offs = torch.tensor([32, 64], dtype=torch.int32).npu()
        with self.assertRaisesRegex(RuntimeError, r"2 or 3d"):
            torch._grouped_mm(mat_a, mat_b, offs=offs)

    def test_neg_missing_offs(self, device="npu"):
        """Negative: offs not provided for 2D input"""
        mat_a = torch.randn(64, 32, dtype=torch.float16).npu()
        mat_b = torch.randn(2, 32, 16, dtype=torch.float16).npu()
        with self.assertRaisesRegex(RuntimeError, r"offset"):
            torch._grouped_mm(mat_a, mat_b)

    def test_neg_offs_with_3d_3d(self, device="npu"):
        """Negative: offs provided for 3D x 3D"""
        mat_a = torch.randn(2, 64, 32, dtype=torch.float16).npu()
        mat_b = torch.randn(2, 32, 16, dtype=torch.float16).npu()
        offs = torch.tensor([32, 64], dtype=torch.int32).npu()
        with self.assertRaisesRegex(RuntimeError, r"offset"):
            torch._grouped_mm(mat_a, mat_b, offs=offs)

    def test_neg_bias_not_supported(self, device="npu"):
        """Negative: bias not supported"""
        mat_a = torch.randn(64, 32, dtype=torch.float16).npu()
        mat_b = torch.randn(2, 32, 16, dtype=torch.float16).npu()
        offs = torch.tensor([32, 64], dtype=torch.int32).npu()
        bias = torch.randn(16, dtype=torch.float16).npu()
        with self.assertRaisesRegex(RuntimeError, r"[Bb]ias"):
            torch._grouped_mm(mat_a, mat_b, offs=offs, bias=bias)


if __name__ == "__main__":
    run_tests()

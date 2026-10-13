import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestThnnFusedGruCell(TestCase):
    def make_thnn_inputs(self, B=8, H=32, dtype=torch.float32):
        device = "npu"
        gi = torch.randn(B, 3 * H, dtype=dtype, device=device)
        gh = torch.randn(B, 3 * H, dtype=dtype, device=device).clamp(-1.5, 1.5)
        hx = torch.randn(B, H, dtype=dtype, device=device).clamp(-1.5, 1.5)
        return gi, gh, hx

    def thnn_fused_gru_cell_reference(self, gi, gh, hx):
        gi_r, gi_z, gi_n = gi.float().chunk(3, dim=1)
        gh_r, gh_z, gh_n = gh.float().chunk(3, dim=1)
        r = torch.sigmoid(gi_r + gh_r)
        z = torch.sigmoid(gi_z + gh_z)
        n = torch.tanh(gi_n + r * gh_n)
        return ((1 - z) * n + z * hx.float()).to(gi.dtype)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_thnn_fused_gru_cell_float32(self):
        gi, gh, hx = self.make_thnn_inputs()
        hy = torch.ops.aten._thnn_fused_gru_cell(gi, gh, hx)
        ref = self.thnn_fused_gru_cell_reference(gi.cpu(), gh.cpu(), hx.cpu())
        self.assertRtolEqual(hy[0].cpu().float().numpy(), ref.float().numpy(), prec=1.e-3)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_thnn_fused_gru_cell_float16(self):
        gi, gh, hx = self.make_thnn_inputs(dtype=torch.float16)
        hy = torch.ops.aten._thnn_fused_gru_cell(gi, gh, hx)
        ref = self.thnn_fused_gru_cell_reference(gi.cpu(), gh.cpu(), hx.cpu())
        self.assertRtolEqual(hy[0].cpu().float().numpy(), ref.float().numpy(), prec=1.e-2)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_thnn_fused_gru_cell_bfloat16(self):
        gi, gh, hx = self.make_thnn_inputs(dtype=torch.bfloat16)
        hy = torch.ops.aten._thnn_fused_gru_cell(gi, gh, hx)
        ref = self.thnn_fused_gru_cell_reference(gi.cpu(), gh.cpu(), hx.cpu())
        self.assertRtolEqual(hy[0].cpu().float().numpy(), ref.float().numpy(), prec=2.e-2)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_thnn_fused_gru_cell_no_bias(self):
        gi, gh, hx = self.make_thnn_inputs(B=4, H=64)
        hy = torch.ops.aten._thnn_fused_gru_cell(gi, gh, hx)
        ref = self.thnn_fused_gru_cell_reference(gi.cpu(), gh.cpu(), hx.cpu())
        self.assertRtolEqual(hy[0].cpu().float().numpy(), ref.float().numpy(), prec=1.e-3)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_thnn_fused_gru_cell_with_bias(self):
        """Direct call with both biases."""
        B, H = 8, 32
        gi, gh, hx = self.make_thnn_inputs(B=B, H=H)
        b_ih = torch.randn(3 * H, dtype=torch.float32, device="npu")
        b_hh = torch.randn(3 * H, dtype=torch.float32, device="npu")
        hy = torch.ops.aten._thnn_fused_gru_cell(gi, gh, hx, b_ih, b_hh)
        ref = self.thnn_fused_gru_cell_reference(
            (gi.cpu() + b_ih.cpu()), (gh.cpu() + b_hh.cpu()), hx.cpu())
        self.assertRtolEqual(hy[0].cpu().float().numpy(), ref.float().numpy(), prec=1.e-3)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_thnn_fused_gru_cell_3d_input_rejected(self):
        """3D inputs are rejected (aten expects 2D gates)."""
        B, T, H = 2, 3, 8
        gi = torch.randn(B, T, 3 * H, dtype=torch.float32, device="npu")
        gh = torch.randn(B, T, 3 * H, dtype=torch.float32, device="npu")
        hx = torch.randn(B, T, H, dtype=torch.float32, device="npu")
        with self.assertRaises(RuntimeError):
            torch.ops.aten._thnn_fused_gru_cell(gi, gh, hx)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_thnn_fused_gru_cell_double_dtype_rejected(self):
        """float64 is not supported and is rejected."""
        B, H = 4, 8
        gi = torch.randn(B, 3 * H, dtype=torch.float64, device="npu")
        gh = torch.randn(B, 3 * H, dtype=torch.float64, device="npu").clamp(-1.5, 1.5)
        hx = torch.randn(B, H, dtype=torch.float64, device="npu").clamp(-1.5, 1.5)
        with self.assertRaises(RuntimeError):
            torch.ops.aten._thnn_fused_gru_cell(gi, gh, hx)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_thnn_fused_gru_cell_large_batch(self):
        gi, gh, hx = self.make_thnn_inputs(B=512, H=128)
        hy = torch.ops.aten._thnn_fused_gru_cell(gi, gh, hx)
        ref = self.thnn_fused_gru_cell_reference(gi.cpu(), gh.cpu(), hx.cpu())
        self.assertRtolEqual(hy[0].cpu().float().numpy(), ref.float().numpy(), prec=1.e-3)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_thnn_fused_gru_cell_non_contiguous(self):
        B, H = 8, 32
        gi_full = torch.randn(B, 6 * H, dtype=torch.float32, device="npu")
        gh_full = torch.randn(B, 6 * H, dtype=torch.float32, device="npu")
        gi = gi_full[:, :3 * H]
        gh = gh_full[:, :3 * H]
        hx = torch.randn(B, H, dtype=torch.float32, device="npu")
        self.assertFalse(gi.is_contiguous() and gh.is_contiguous())
        hy = torch.ops.aten._thnn_fused_gru_cell(gi, gh, hx)
        ref = self.thnn_fused_gru_cell_reference(gi.contiguous().cpu(), gh.contiguous().cpu(), hx.cpu())
        self.assertRtolEqual(hy[0].cpu().float().numpy(), ref.float().numpy(), prec=1.e-3)

    @SupportedDevices(["Ascend950"])
    def test_thnn_fused_gru_cell_fused_workspace(self):
        """Fused path (Ascend950 only): workspace is (B, 5H); the hx plane
        must match the input."""
        B, H = 4, 32
        gi, gh, hx = self.make_thnn_inputs(B=B, H=H)
        hy, workspace = torch.ops.aten._thnn_fused_gru_cell(gi, gh, hx)
        self.assertEqual(tuple(workspace.shape), (B, 5 * H))
        hx_w = workspace[:, 3 * H:4 * H]
        self.assertRtolEqual(hx_w.cpu().numpy(), hx.cpu().numpy(), prec=1.e-5)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_thnn_fused_gru_cell_empty_batch(self):
        H = 64
        gi = torch.randn(0, 3 * H, dtype=torch.float32, device="npu")
        gh = torch.randn(0, 3 * H, dtype=torch.float32, device="npu")
        hx = torch.randn(0, H, dtype=torch.float32, device="npu")
        hy, ws = torch.ops.aten._thnn_fused_gru_cell(gi, gh, hx)
        self.assertEqual(tuple(hy.shape), (0, H))

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_thnn_fused_gru_cell_empty_hidden(self):
        B = 4
        gi = torch.randn(B, 0, dtype=torch.float32, device="npu")
        gh = torch.randn(B, 0, dtype=torch.float32, device="npu")
        hx = torch.randn(B, 0, dtype=torch.float32, device="npu")
        hy, ws = torch.ops.aten._thnn_fused_gru_cell(gi, gh, hx)
        self.assertEqual(tuple(hy.shape), (B, 0))

    @SupportedDevices(["Ascend910B"])
    def test_thnn_fused_gru_cell_lazy_workspace(self):
        """The composite fallback (Ascend910B) builds the workspace only for
        calls that will be differentiated; inference and no_grad stay free of
        the extra allocation."""
        B, H = 8, 32
        gi, gh, hx = self.make_thnn_inputs(B=B, H=H)
        hy, ws = torch.ops.aten._thnn_fused_gru_cell(gi, gh, hx)
        self.assertIsNone(ws)  # pure inference: no requires_grad input

        rg = [t.detach().requires_grad_(True) for t in (gi, gh, hx)]
        with torch.no_grad():
            hy2, ws2 = torch.ops.aten._thnn_fused_gru_cell(*rg)
        self.assertIsNone(ws2)  # eval loop: requires_grad inputs but grad mode off

        hy3, ws3 = torch.ops.aten._thnn_fused_gru_cell(*rg)
        self.assertIsNotNone(ws3)  # training: node will be recorded
        self.assertEqual(tuple(ws3.shape), (B, 5 * H))

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_thnn_fused_gru_cell_no_grad(self):
        """GradMode: fused-cell forward under no_grad matches the default mode."""
        gi, gh, hx = self.make_thnn_inputs()
        hy_d = torch.ops.aten._thnn_fused_gru_cell(gi, gh, hx)
        with torch.no_grad():
            hy_n = torch.ops.aten._thnn_fused_gru_cell(gi, gh, hx)
        self.assertRtolEqual(hy_d[0].cpu().numpy(), hy_n[0].cpu().numpy(), prec=1.e-5)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_thnn_fused_gru_cell_inference_mode(self):
        """GradMode: fused-cell forward under inference_mode."""
        gi, gh, hx = self.make_thnn_inputs()
        with torch.inference_mode():
            hy = torch.ops.aten._thnn_fused_gru_cell(gi, gh, hx)
        self.assertFalse(hy[0].requires_grad)


if __name__ == "__main__":
    run_tests()

import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestThnnFusedGruCellBackward(TestCase):
    """Direct calls of _thnn_fused_gru_cell_backward. Runs on Ascend910B
    (decomposed fallback) and Ascend950 (fused kernel, requires the
    ThnnFusedGruCellGrad vendor package).
    """

    def make_backward_inputs(self, B=8, H=32, dtype=torch.float32):
        device = "npu"
        grad_hy = torch.randn(B, H, dtype=dtype, device=device)
        workspace = torch.randn(B, 5 * H, dtype=dtype, device=device).clamp(-1.5, 1.5)
        return grad_hy, workspace

    def ref_backward(self, grad_hy, workspace, has_bias):
        B, H = grad_hy.shape
        r = workspace[:, 0 * H:1 * H].float()
        z = workspace[:, 1 * H:2 * H].float()
        n = workspace[:, 2 * H:3 * H].float()
        hx = workspace[:, 3 * H:4 * H].float()
        gh_n = workspace[:, 4 * H:5 * H].float()
        go = grad_hy.float()
        gin = go * (1 - z) * (1 - n * n)
        gig = go * (hx - n) * (1 - z) * z
        grg = gin * gh_n * (1 - r) * r
        ghn = gin * r
        ghx = go * z
        in_gates = torch.cat((grg, gig, gin), dim=1)
        hid_gates = torch.cat((grg, gig, ghn), dim=1)
        if has_bias:
            return (in_gates, hid_gates, ghx, in_gates.sum(0), hid_gates.sum(0))
        return (in_gates, hid_gates, ghx, None, None)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_thnn_fused_gru_cell_backward_float32(self):
        """Per-output comparison with float32."""
        gh, ws = self.make_backward_inputs()
        grads = torch.ops.aten._thnn_fused_gru_cell_backward(gh, ws, True)
        ref = self.ref_backward(gh.cpu(), ws.cpu(), True)
        for i in range(5):
            if ref[i] is None:
                continue
            self.assertRtolEqual(grads[i].cpu().numpy(), ref[i].numpy(), prec=1.e-2)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_thnn_fused_gru_cell_backward_float16(self):
        gh, ws = self.make_backward_inputs(dtype=torch.float16)
        grads = torch.ops.aten._thnn_fused_gru_cell_backward(gh, ws, True)
        ref = self.ref_backward(gh.cpu(), ws.cpu(), True)
        for i in range(3):
            self.assertRtolEqual(grads[i].cpu().float().numpy(), ref[i].numpy(), prec=5.e-2)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_thnn_fused_gru_cell_backward_bfloat16(self):
        gh, ws = self.make_backward_inputs(dtype=torch.bfloat16)
        grads = torch.ops.aten._thnn_fused_gru_cell_backward(gh, ws, True)
        ref = self.ref_backward(gh.cpu(), ws.cpu(), True)
        for i in range(3):
            self.assertRtolEqual(grads[i].cpu().float().numpy(), ref[i].numpy(), prec=1.e-1)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_thnn_fused_gru_cell_backward_no_bias(self):
        """has_bias=False: bias outputs are undefined."""
        gh, ws = self.make_backward_inputs(B=4, H=64)
        grads = torch.ops.aten._thnn_fused_gru_cell_backward(gh, ws, False)
        ref = self.ref_backward(gh.cpu(), ws.cpu(), False)
        self.assertIsNone(grads[3])
        self.assertIsNone(grads[4])
        for i in range(3):
            self.assertRtolEqual(grads[i].cpu().numpy(), ref[i].numpy(), prec=1.e-2)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_thnn_fused_gru_cell_backward_empty_batch(self):
        """B=0 with bias: bias grads are all-zero."""
        H = 64
        gh = torch.randn(0, H, dtype=torch.float32, device="npu")
        ws = torch.randn(0, 5 * H, dtype=torch.float32, device="npu")
        grads = torch.ops.aten._thnn_fused_gru_cell_backward(gh, ws, True)
        self.assertTrue((grads[3].cpu() == 0).all())

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_thnn_fused_gru_cell_backward_non_contiguous(self):
        """Non-contiguous grad_hy/workspace inputs."""
        B, H = 8, 32
        gh_full = torch.randn(B, 2 * H, dtype=torch.float32, device="npu")
        ws_full = torch.randn(B, 6 * H, dtype=torch.float32, device="npu").clamp(-1.5, 1.5)
        gh = gh_full[:, :H]
        ws = ws_full[:, :5 * H]
        self.assertFalse(gh.is_contiguous() and ws.is_contiguous())
        grads = torch.ops.aten._thnn_fused_gru_cell_backward(gh, ws, True)
        ref = self.ref_backward(gh.contiguous().cpu(), ws.contiguous().cpu(), True)
        for i in range(3):
            self.assertRtolEqual(grads[i].cpu().numpy(), ref[i].numpy(), prec=1.e-2)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_thnn_fused_gru_cell_backward_undefined_workspace(self):
        """An undefined workspace is rejected (contract check)."""
        gh, _ = self.make_backward_inputs()
        with self.assertRaises(RuntimeError):
            torch.ops.aten._thnn_fused_gru_cell_backward(gh, None, True)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_thnn_fused_gru_cell_backward_wrong_shape(self):
        """A workspace with wrong width (B, 4H) is rejected (contract check)."""
        B, H = 8, 32
        gh = torch.randn(B, H, dtype=torch.float32, device="npu")
        ws = torch.randn(B, 4 * H, dtype=torch.float32, device="npu")
        with self.assertRaises(RuntimeError):
            torch.ops.aten._thnn_fused_gru_cell_backward(gh, ws, True)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_thnn_fused_gru_cell_backward_wrong_rank(self):
        """A non-2D grad_hy is rejected (contract check, clean error)."""
        B, H = 8, 32
        gh = torch.randn(B * H, dtype=torch.float32, device="npu")
        ws = torch.randn(B, 5 * H, dtype=torch.float32, device="npu")
        with self.assertRaises(RuntimeError):
            torch.ops.aten._thnn_fused_gru_cell_backward(gh, ws, True)


if __name__ == "__main__":
    run_tests()

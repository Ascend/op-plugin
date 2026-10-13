import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestGruCell(TestCase):
    def gru_cell_reference(self, input, hx, w_ih, w_hh, b_ih=None, b_hh=None):
        """Reference implementation of gru_cell for CPU comparison.

        GRU cell equations:
        gi = input @ w_ih.T (+ b_ih)
        gh = hx @ w_hh.T (+ b_hh)
        r = sigmoid(gi_r + gh_r)
        z = sigmoid(gi_z + gh_z)
        n = tanh(gi_n + r * gh_n)
        hy = (1 - z) * n + z * hx
        """
        gi = input.float() @ w_ih.float().t()
        gh = hx.float() @ w_hh.float().t()
        if b_ih is not None:
            gi = gi + b_ih.float()
        if b_hh is not None:
            gh = gh + b_hh.float()

        gi_r, gi_z, gi_n = gi.chunk(3, dim=1)
        gh_r, gh_z, gh_n = gh.chunk(3, dim=1)

        r = torch.sigmoid(gi_r + gh_r)
        z = torch.sigmoid(gi_z + gh_z)
        n = torch.tanh(gi_n + r * gh_n)
        hy = (1 - z) * n + z * hx.float()

        dtype = input.dtype
        return hy.to(dtype)

    def make_inputs(self, B=8, H=32, I=64):
        device = "npu"
        return (torch.randn(B, I, dtype=torch.float32, device=device),
                torch.randn(B, H, dtype=torch.float32, device=device),
                torch.randn(3 * H, I, dtype=torch.float32, device=device),
                torch.randn(3 * H, H, dtype=torch.float32, device=device),
                torch.randn(3 * H, dtype=torch.float32, device=device),
                torch.randn(3 * H, dtype=torch.float32, device=device))

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_gru_cell_float32(self):
        """Test gru_cell with float32 dtype."""
        batch_size = 32
        hidden_size = 16

        input = torch.randn(batch_size, 64, dtype=torch.float32, device="npu")
        hx = torch.randn(batch_size, hidden_size, dtype=torch.float32, device="npu")
        w_ih = torch.randn(3 * hidden_size, 64, dtype=torch.float32, device="npu")
        w_hh = torch.randn(3 * hidden_size, hidden_size, dtype=torch.float32, device="npu")
        b_ih = torch.randn(3 * hidden_size, dtype=torch.float32, device="npu")
        b_hh = torch.randn(3 * hidden_size, dtype=torch.float32, device="npu")

        npu_hy = torch.ops.aten.gru_cell(input, hx, w_ih, w_hh, b_ih, b_hh)
        cpu_hy = self.gru_cell_reference(input.cpu(), hx.cpu(), w_ih.cpu(), w_hh.cpu(),
                                         b_ih.cpu(), b_hh.cpu())

        self.assertRtolEqual(npu_hy.cpu().numpy(), cpu_hy.numpy(), prec=1.e-3)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_gru_cell_float16(self):
        """Test gru_cell with float16 dtype."""
        batch_size = 16
        hidden_size = 32

        input = torch.randn(batch_size, 64, dtype=torch.float16, device="npu")
        hx = torch.randn(batch_size, hidden_size, dtype=torch.float16, device="npu")
        w_ih = torch.randn(3 * hidden_size, 64, dtype=torch.float16, device="npu")
        w_hh = torch.randn(3 * hidden_size, hidden_size, dtype=torch.float16, device="npu")

        npu_hy = torch.ops.aten.gru_cell(input, hx, w_ih, w_hh)
        cpu_hy = self.gru_cell_reference(input.cpu(), hx.cpu(), w_ih.cpu(), w_hh.cpu())

        self.assertRtolEqual(npu_hy.cpu().float().numpy(), cpu_hy.float().numpy(), prec=1.e-2)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_gru_cell_bfloat16(self):
        """Test gru_cell with bfloat16 dtype."""
        batch_size = 8
        hidden_size = 32

        input = torch.randn(batch_size, 64, dtype=torch.bfloat16, device="npu")
        hx = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device="npu")
        w_ih = torch.randn(3 * hidden_size, 64, dtype=torch.bfloat16, device="npu")
        w_hh = torch.randn(3 * hidden_size, hidden_size, dtype=torch.bfloat16, device="npu")

        npu_hy = torch.ops.aten.gru_cell(input, hx, w_ih, w_hh)
        cpu_hy = self.gru_cell_reference(input.cpu(), hx.cpu(), w_ih.cpu(), w_hh.cpu())

        self.assertRtolEqual(npu_hy.cpu().float().numpy(), cpu_hy.float().numpy(), prec=2.e-2)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_gru_cell_no_bias(self):
        """Test gru_cell without bias."""
        batch_size = 4
        hidden_size = 128

        input = torch.randn(batch_size, 256, dtype=torch.float32, device="npu")
        hx = torch.randn(batch_size, hidden_size, dtype=torch.float32, device="npu")
        w_ih = torch.randn(3 * hidden_size, 256, dtype=torch.float32, device="npu")
        w_hh = torch.randn(3 * hidden_size, hidden_size, dtype=torch.float32, device="npu")

        npu_hy = torch.ops.aten.gru_cell(input, hx, w_ih, w_hh)
        cpu_hy = self.gru_cell_reference(input.cpu(), hx.cpu(), w_ih.cpu(), w_hh.cpu())

        self.assertRtolEqual(npu_hy.cpu().numpy(), cpu_hy.numpy(), prec=1.e-3)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_gru_cell_no_grad_consistent(self):
        """GradMode: no_grad output matches the default-mode output."""
        inp = self.make_inputs()
        hy_d = torch.ops.aten.gru_cell(*inp)
        with torch.no_grad():
            hy_n = torch.ops.aten.gru_cell(*inp)
        self.assertFalse(hy_d.requires_grad or hy_n.requires_grad)
        self.assertRtolEqual(hy_d.cpu().numpy(), hy_n.cpu().numpy(), prec=1.e-5)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_gru_cell_inference_mode(self):
        """GradMode: inference_mode forward works."""
        inp = self.make_inputs()
        with torch.inference_mode():
            hy = torch.ops.aten.gru_cell(*inp)
        self.assertFalse(hy.requires_grad)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_gru_cell_inference_then_training(self):
        """GradMode: after an inference-mode round, training still records gradients."""
        inp = self.make_inputs()
        with torch.inference_mode():
            torch.ops.aten.gru_cell(*inp)
        ts = [t.detach().clone().requires_grad_(True) for t in inp]
        hy = torch.ops.aten.gru_cell(*ts)
        hy.sum().backward()
        for t in ts:
            self.assertFalse(t.grad is None)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_gru_cell_enable_grad_in_no_grad(self):
        """GradMode: enable_grad nested inside no_grad re-enables recording."""
        inp = self.make_inputs()
        with torch.no_grad():
            with torch.enable_grad():
                ts = [t.detach().clone().requires_grad_(True) for t in inp[:4]]
                hy = torch.ops.aten.gru_cell(*ts)
                self.assertTrue(hy.requires_grad)
                hy.sum().backward()
                for t in ts:
                    self.assertFalse(t.grad is None)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_gru_cell_grad_mode_switch(self):
        """GradMode: default -> no_grad -> default produces identical outputs."""
        inp = self.make_inputs()
        hy1 = torch.ops.aten.gru_cell(*inp)
        with torch.no_grad():
            hy2 = torch.ops.aten.gru_cell(*inp)
        hy3 = torch.ops.aten.gru_cell(*inp)
        self.assertRtolEqual(hy1.cpu().numpy(), hy2.cpu().numpy(), prec=1.e-5)
        self.assertRtolEqual(hy1.cpu().numpy(), hy3.cpu().numpy(), prec=1.e-5)

    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_gru_cell_no_grad_output_inplace(self):
        """GradMode: no_grad outputs are plain (non-inference) tensors."""
        inp = self.make_inputs()
        with torch.no_grad():
            hy = torch.ops.aten.gru_cell(*inp)
        hy.add_(1.0)
        self.assertFalse(hy.requires_grad)


if __name__ == "__main__":
    run_tests()

import copy
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices, SkipIfNotGteCANNVersion


class TestLstmCellBackward(TestCase):
    def lstm_cell_backward_reference(self, grad_hy, grad_c, cx, cy, workspace, has_bias=True):
        if grad_hy is None and grad_c is None:
            return None, None, None
        i, f, g, o = workspace.chunk(4, dim=1)
        ghy = grad_hy.float()
        gcy = grad_c.float()
        _cx = cx.float()
        _cy = cy.float()
        _i, _f, _g, _o = i.float(), f.float(), g.float(), o.float()
        do = ghy * torch.tanh(_cy)
        dcy_total = gcy + ghy * _o * (1 - torch.tanh(_cy) ** 2)

        di = dcy_total * _g * (_i * (1 - _i))
        df = dcy_total * _cx * (_f * (1 - _f))
        dg = dcy_total * _i * (1 - _g ** 2)
        do = do * (_o * (1 - _o))

        grad_gates_f32 = torch.cat([di, df, dg, do], dim=1)
        grad_gates = grad_gates_f32.to(workspace.dtype)
        grad_cx_f32 = dcy_total * _f
        grad_cx = grad_cx_f32.to(cx.dtype)
        grad_bias = grad_gates_f32.sum(0).to(workspace.dtype)
        return grad_gates, grad_cx, grad_bias


    @SkipIfNotGteCANNVersion("9.0.0")
    @SupportedDevices(["Ascend910B", "Ascend910C"])
    def test_thnn_fused_lstm_cell_backward(self):
        torch.set_grad_enabled(False)
        grad_hy, grad_c, cx, cy = [torch.randn(40, 8, dtype=torch.float32, device="npu") for _ in range(4)]
        gates = torch.randn(40, 32, dtype=torch.float32, device="npu")
        npu_out = torch.ops.aten._thnn_fused_lstm_cell_backward_impl(grad_hy, grad_c, cx, cy, gates, True)
        cpu_out = self.lstm_cell_backward_reference(grad_hy.cpu(), grad_c.cpu(), cx.cpu(), cy.cpu(), gates.cpu(), True)
        self.assertRtolEqual(npu_out[0].cpu().to(torch.float).numpy(), cpu_out[0].numpy(), prec=1.e-3)
        self.assertRtolEqual(npu_out[1].cpu().to(torch.float).numpy(), cpu_out[1].numpy(), prec=1.e-3)
        self.assertRtolEqual(npu_out[2].cpu().to(torch.float).numpy(), cpu_out[2].numpy(), prec=1.e-3)


if __name__ == "__main__":
    run_tests()

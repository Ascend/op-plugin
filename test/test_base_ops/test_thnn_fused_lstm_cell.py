import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices, SkipIfNotGteCANNVersion


class TestThnnFusedLstmCell(TestCase):
    def thnn_fused_lstm_cell_reference(self, input_gates, hidden_gates, c, input_bias=None, hidden_bias=None):
        """Reference implementation of _thnn_fused_lstm_cell for CPU comparison.
        
        LSTM cell equations:
        i = sigmoid(W_ii * x + b_ii + W_hi * h + b_hi)
        f = sigmoid(W_if * x + b_if + W_hf * h + b_hf)
        g = tanh(W_ig * x + b_ig + W_hg * h + b_hg)
        o = sigmoid(W_io * x + b_io + W_ho * h + b_ho)
        c' = f * c + i * g
        h' = o * tanh(c')
        
        Args:
            input_gates: Tensor of shape (batch_size, 4*hidden_size) containing input gates
            hidden_gates: Tensor of shape (batch_size, 4*hidden_size) containing hidden gates
            c: Tensor of shape (batch_size, hidden_size) - cell state
            input_bias: Optional bias for input gates, shape (4*hidden_size,)
            hidden_bias: Optional bias for hidden gates, shape (4*hidden_size,)
        
        Returns:
            hy: Tensor of shape (batch_size, hidden_size) - new hidden state
            cy: Tensor of shape (batch_size, hidden_size) - new cell state
            workspace: Tensor of shape (batch_size, 4*hidden_size) - intermediate gates
        """
        # Add biases if provided
        gates = input_gates.float() + hidden_gates.float()
        if input_bias is not None:
            gates = gates + input_bias.float()
        if hidden_bias is not None:
            gates = gates + hidden_bias.float()

        # Split gates into i, f, g, o components
        # The order is: input, forget, cell, output gates
        i, f, g, o = gates.chunk(4, dim=1)

        # Apply activations
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        # Compute new cell state and hidden state
        c_f32 = c.float()
        cy = f * c_f32 + i * g
        hy = o * torch.tanh(cy)

        # Workspace contains the computed gates (i, f, g, o)
        workspace = torch.cat([i, f, g, o], dim=1)

        # Convert back to original dtype
        dtype = input_gates.dtype
        return hy.to(dtype), cy.to(dtype), workspace.to(dtype)

    @SkipIfNotGteCANNVersion("9.0.0")
    @SupportedDevices(["Ascend910B", "Ascend910C"])
    def test_thnn_fused_lstm_cell_float32(self):
        """Test _thnn_fused_lstm_cell with float32 dtype."""
        batch_size = 32
        hidden_size = 16

        # Create input tensors on NPU
        input_gates = torch.randn(batch_size, 4 * hidden_size, dtype=torch.float32, device="npu")
        hidden_gates = torch.randn(batch_size, 4 * hidden_size, dtype=torch.float32, device="npu")
        c = torch.randn(batch_size, hidden_size, dtype=torch.float32, device="npu")
        input_bias = torch.randn(4 * hidden_size, dtype=torch.float32, device="npu")
        hidden_bias = torch.randn(4 * hidden_size, dtype=torch.float32, device="npu")

        # NPU forward
        npu_hy, npu_cy, npu_workspace = torch.ops.aten._thnn_fused_lstm_cell(
            input_gates, hidden_gates, c, input_bias, hidden_bias
        )

        # CPU reference
        cpu_hy, cpu_cy, cpu_workspace = self.thnn_fused_lstm_cell_reference(
            input_gates.cpu(), hidden_gates.cpu(), c.cpu(), input_bias.cpu(), hidden_bias.cpu()
        )

        # Compare results
        self.assertRtolEqual(npu_hy.cpu().numpy(), cpu_hy.numpy(), prec=1.e-3)
        self.assertRtolEqual(npu_cy.cpu().numpy(), cpu_cy.numpy(), prec=1.e-3)
        self.assertRtolEqual(npu_workspace.cpu().numpy(), cpu_workspace.numpy(), prec=1.e-3)

    @SkipIfNotGteCANNVersion("9.0.0")
    @SupportedDevices(["Ascend910B", "Ascend910C"])
    def test_thnn_fused_lstm_cell_float16(self):
        """Test _thnn_fused_lstm_cell with float16 dtype."""
        batch_size = 64
        hidden_size = 32

        # Create input tensors on NPU
        input_gates = torch.randn(batch_size, 4 * hidden_size, dtype=torch.float16, device="npu")
        hidden_gates = torch.randn(batch_size, 4 * hidden_size, dtype=torch.float16, device="npu")
        c = torch.randn(batch_size, hidden_size, dtype=torch.float16, device="npu")
        input_bias = torch.randn(4 * hidden_size, dtype=torch.float16, device="npu")
        hidden_bias = torch.randn(4 * hidden_size, dtype=torch.float16, device="npu")

        # NPU forward
        npu_hy, npu_cy, npu_workspace = torch.ops.aten._thnn_fused_lstm_cell(
            input_gates, hidden_gates, c, input_bias, hidden_bias
        )

        # CPU reference (convert to float32 for computation)
        cpu_hy, cpu_cy, cpu_workspace = self.thnn_fused_lstm_cell_reference(
            input_gates.cpu(), hidden_gates.cpu(), c.cpu(), input_bias.cpu(), hidden_bias.cpu()
        )

        # Compare results (with higher tolerance for float16)
        self.assertRtolEqual(npu_hy.cpu().float().numpy(), cpu_hy.float().numpy(), prec=1.e-2)
        self.assertRtolEqual(npu_cy.cpu().float().numpy(), cpu_cy.float().numpy(), prec=1.e-2)
        self.assertRtolEqual(npu_workspace.cpu().float().numpy(), cpu_workspace.float().numpy(), prec=1.e-2)

    @SkipIfNotGteCANNVersion("9.0.0")
    @SupportedDevices(["Ascend910B", "Ascend910C"])
    def test_thnn_fused_lstm_cell_no_bias(self):
        """Test _thnn_fused_lstm_cell without bias."""
        batch_size = 16
        hidden_size = 8
        
        # Create input tensors on NPU (no bias)
        input_gates = torch.randn(batch_size, 4 * hidden_size, dtype=torch.float32, device="npu")
        hidden_gates = torch.randn(batch_size, 4 * hidden_size, dtype=torch.float32, device="npu")
        c = torch.randn(batch_size, hidden_size, dtype=torch.float32, device="npu")
        
        # NPU forward
        npu_hy, npu_cy, npu_workspace = torch.ops.aten._thnn_fused_lstm_cell(
            input_gates, hidden_gates, c, None, None
        )
        
        # CPU reference
        cpu_hy, cpu_cy, cpu_workspace = self.thnn_fused_lstm_cell_reference(
            input_gates.cpu(), hidden_gates.cpu(), c.cpu(), None, None
        )
        
        # Compare results
        self.assertRtolEqual(npu_hy.cpu().numpy(), cpu_hy.numpy(), prec=1.e-3)
        self.assertRtolEqual(npu_cy.cpu().numpy(), cpu_cy.numpy(), prec=1.e-3)
        self.assertRtolEqual(npu_workspace.cpu().numpy(), cpu_workspace.numpy(), prec=1.e-3)

    @SkipIfNotGteCANNVersion("9.0.0")
    @SupportedDevices(["Ascend910B", "Ascend910C"])
    def test_thnn_fused_lstm_cell_large_batch(self):
        """Test _thnn_fused_lstm_cell with large batch size."""
        batch_size = 256
        hidden_size = 64

        # Create input tensors on NPU
        input_gates = torch.randn(batch_size, 4 * hidden_size, dtype=torch.float32, device="npu")
        hidden_gates = torch.randn(batch_size, 4 * hidden_size, dtype=torch.float32, device="npu")
        c = torch.randn(batch_size, hidden_size, dtype=torch.float32, device="npu")
        input_bias = torch.randn(4 * hidden_size, dtype=torch.float32, device="npu")
        hidden_bias = torch.randn(4 * hidden_size, dtype=torch.float32, device="npu")
        
        # NPU forward
        npu_hy, npu_cy, npu_workspace = torch.ops.aten._thnn_fused_lstm_cell(
            input_gates, hidden_gates, c, input_bias, hidden_bias
        )

        # CPU reference
        cpu_hy, cpu_cy, cpu_workspace = self.thnn_fused_lstm_cell_reference(
            input_gates.cpu(), hidden_gates.cpu(), c.cpu(), input_bias.cpu(), hidden_bias.cpu()
        )

        # Compare results
        self.assertRtolEqual(npu_hy.cpu().numpy(), cpu_hy.numpy(), prec=1.e-3)
        self.assertRtolEqual(npu_cy.cpu().numpy(), cpu_cy.numpy(), prec=1.e-3)
        self.assertRtolEqual(npu_workspace.cpu().numpy(), cpu_workspace.numpy(), prec=1.e-3)


if __name__ == "__main__":
    run_tests()

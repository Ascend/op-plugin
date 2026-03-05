import copy
import torch
import torch.nn as nn
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices, SkipIfNotGteCANNVersion


class TestLSTMCellForwardBackward(TestCase):
    """Test LSTMCell forward and backward through torch.nn.LSTMCell.
    
    This test validates that the aclnn implementation of _thnn_fused_lstm_cell
    and _thnn_fused_lstm_cell_backward_impl works correctly when called through
    the standard torch.nn.LSTMCell interface.
    """

    def _test_lstm_cell_forward_backward(self, batch_size, input_size, hidden_size, 
                                          dtype, has_bias=True, prec=1.e-3):
        """Helper function to test LSTMCell forward and backward.

        Args:
            batch_size: Batch size for input
            input_size: Input feature dimension
            hidden_size: Hidden state dimension
            dtype: Data type (torch.float32 or torch.float16)
            has_bias: Whether to use bias in LSTMCell
            prec: Relative tolerance for comparison
        """
        # Create LSTMCell on CPU and NPU
        cpu_lstm = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size, bias=has_bias)
        npu_lstm = copy.deepcopy(cpu_lstm).npu()

        # Create input and hidden states
        input_cpu = torch.randn(batch_size, input_size, dtype=torch.float32, requires_grad=True)
        h0_cpu = torch.randn(batch_size, hidden_size, dtype=torch.float32, requires_grad=True)
        c0_cpu = torch.randn(batch_size, hidden_size, dtype=torch.float32, requires_grad=True)

        # Convert to NPU with specified dtype
        input_npu = input_cpu.to(dtype).npu()
        input_npu.requires_grad_(True)

        h0_npu = h0_cpu.to(dtype).npu()
        h0_npu.requires_grad_(True)

        c0_npu = c0_cpu.to(dtype).npu()
        c0_npu.requires_grad_(True)

        # Forward pass on CPU
        h_cpu, c_cpu = cpu_lstm(input_cpu, (h0_cpu, c0_cpu))

        # Forward pass on NPU
        h_npu, c_npu = npu_lstm(input_npu, (h0_npu, c0_npu))

        # Compare forward results
        self.assertRtolEqual(
            h_cpu.detach().numpy(), 
            h_npu.cpu().to(torch.float32).detach().numpy(), 
            prec=prec
        )
        self.assertRtolEqual(
            c_cpu.detach().numpy(), 
            c_npu.cpu().to(torch.float32).detach().numpy(), 
            prec=prec
        )

        # Backward pass - create gradients for output
        grad_h = torch.randn_like(h_cpu)
        grad_c = torch.randn_like(c_cpu)

        # CPU backward using autograd.grad
        cpu_grads = torch.autograd.grad(
            outputs=(h_cpu, c_cpu),
            inputs=(input_cpu, h0_cpu, c0_cpu),
            grad_outputs=(grad_h, grad_c),
            retain_graph=True
        )
        cpu_input_grad = cpu_grads[0]
        cpu_h0_grad = cpu_grads[1]
        cpu_c0_grad = cpu_grads[2]

        # NPU backward using autograd.grad
        grad_h_npu = grad_h.to(dtype).npu()
        grad_c_npu = grad_c.to(dtype).npu()
        npu_grads = torch.autograd.grad(
            outputs=(h_npu, c_npu),
            inputs=(input_npu, h0_npu, c0_npu),
            grad_outputs=(grad_h_npu, grad_c_npu),
            retain_graph=True
        )
        npu_input_grad = npu_grads[0]
        npu_h0_grad = npu_grads[1]
        npu_c0_grad = npu_grads[2]

        # Compare gradients for input
        self.assertRtolEqual(
            cpu_input_grad.numpy(), 
            npu_input_grad.cpu().to(torch.float32).numpy(), 
            prec=prec
        )

        # Compare gradients for hidden states
        self.assertRtolEqual(
            cpu_h0_grad.numpy(), 
            npu_h0_grad.cpu().to(torch.float32).numpy(), 
            prec=prec
        )
        self.assertRtolEqual(
            cpu_c0_grad.numpy(), 
            npu_c0_grad.cpu().to(torch.float32).numpy(), 
            prec=prec
        )

    @SkipIfNotGteCANNVersion("9.0.0")
    @SupportedDevices(["Ascend910B", "Ascend910C"])
    def test_lstm_cell_forward_backward_float32(self):
        """Test LSTMCell forward and backward with float32."""
        self._test_lstm_cell_forward_backward(
            batch_size=32, input_size=64, hidden_size=32,
            dtype=torch.float32, has_bias=True, prec=1.e-3
        )

    @SkipIfNotGteCANNVersion("9.0.0")
    @SupportedDevices(["Ascend910B", "Ascend910C"])
    def test_lstm_cell_forward_backward_float16(self):
        """Test LSTMCell forward and backward with float16."""
        self._test_lstm_cell_forward_backward(
            batch_size=64, input_size=128, hidden_size=64,
            dtype=torch.float16, has_bias=True, prec=1.e-2
        )

    @SkipIfNotGteCANNVersion("9.0.0")
    @SupportedDevices(["Ascend910B", "Ascend910C"])
    def test_lstm_cell_forward_backward_fp32_no_bias(self):
        """Test LSTMCell forward and backward with float16 and without bias."""
        self._test_lstm_cell_forward_backward(
            batch_size=16, input_size=32, hidden_size=16,
            dtype=torch.float32, has_bias=False, prec=1.e-3
        )

    @SkipIfNotGteCANNVersion("9.0.0")
    @SupportedDevices(["Ascend910B", "Ascend910C"])
    def test_lstm_cell_forward_backward_large_batch(self):
        """Test LSTMCell forward and backward with large batch."""
        self._test_lstm_cell_forward_backward(
            batch_size=256, input_size=512, hidden_size=256,
            dtype=torch.float32, has_bias=True, prec=1.e-3
        )

    @SkipIfNotGteCANNVersion("9.0.0")
    @SupportedDevices(["Ascend910B", "Ascend910C"])
    def test_lstm_cell_forward_backward_fp16_no_bias(self):
        """Test LSTMCell forward and backward with float16 and no bias."""
        self._test_lstm_cell_forward_backward(
            batch_size=48, input_size=96, hidden_size=48,
            dtype=torch.float16, has_bias=False, prec=1.e-2
        )


if __name__ == "__main__":
    run_tests()

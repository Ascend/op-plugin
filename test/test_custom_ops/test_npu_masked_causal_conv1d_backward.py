import unittest
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


# Golden reference — aligned with test_masked_causal_conv1d_backward.py in pta/
def masked_causal_conv1d_backward_golden(grad_output, x, weight, mask=None):
    out_dtype = grad_output.dtype
    go = grad_output.float().cpu().clone()   # [S, B, H]
    x_f32 = x.float().cpu()                 # [S, B, H]
    w_f32 = weight.float().cpu()            # [3, H]

    if mask is not None:
        mask_cpu = mask.cpu()               # [B, S]
        go[~mask_cpu.transpose(0, 1)] = 0.0

    # grad_input
    grad_in = go * w_f32[2].unsqueeze(0).unsqueeze(0)
    grad_in[:-1] += go[1:] * w_f32[1].unsqueeze(0).unsqueeze(0)
    grad_in[:-2] += go[2:] * w_f32[0].unsqueeze(0).unsqueeze(0)

    # grad_weight
    grad_w0 = (go[2:] * x_f32[:-2]).sum(dim=0).sum(dim=0)
    grad_w1 = (go[1:] * x_f32[:-1]).sum(dim=0).sum(dim=0)
    grad_w2 = (go     * x_f32     ).sum(dim=0).sum(dim=0)
    grad_weight = torch.stack([grad_w0, grad_w1, grad_w2], dim=0)

    return grad_in.to(out_dtype), grad_weight.to(out_dtype)


class TestNpuMaskedCausalConv1dBackward(TestCase):
    @unittest.skip("Skip test_npu_masked_causal_conv1d_backward now")
    @SupportedDevices(['Ascend950'])
    def test_npu_masked_causal_conv1d_backward_bf16_with_mask(self):
        S, B, H, W = 512, 4, 128, 3
        dtype = torch.bfloat16

        grad_output = torch.randn(S, B, H, dtype=dtype)
        x = torch.randn(S, B, H, dtype=dtype)
        weight = torch.randn(W, H, dtype=dtype)
        mask = torch.rand(B, S) > 0.3

        golden_grad_in, golden_grad_w = masked_causal_conv1d_backward_golden(
            grad_output, x, weight, mask
        )

        grad_in_npu, grad_w_npu = torch_npu.npu_masked_causal_conv1d_backward(
            grad_output.npu(), x.npu(), weight.npu(), mask=mask.npu()
        )
        torch.npu.synchronize()

        self.assertRtolEqual(grad_in_npu.cpu(), golden_grad_in)
        self.assertRtolEqual(grad_w_npu.cpu(), golden_grad_w)

    @unittest.skip("Skip test_npu_masked_causal_conv1d_backward now")
    @SupportedDevices(['Ascend950'])
    def test_npu_masked_causal_conv1d_backward_fp16_with_mask(self):
        S, B, H, W = 512, 4, 128, 3
        dtype = torch.float16

        grad_output = torch.randn(S, B, H, dtype=dtype)
        x = torch.randn(S, B, H, dtype=dtype)
        weight = torch.randn(W, H, dtype=dtype)
        mask = torch.rand(B, S) > 0.3

        golden_grad_in, golden_grad_w = masked_causal_conv1d_backward_golden(
            grad_output, x, weight, mask
        )

        grad_in_npu, grad_w_npu = torch_npu.npu_masked_causal_conv1d_backward(
            grad_output.npu(), x.npu(), weight.npu(), mask=mask.npu()
        )
        torch.npu.synchronize()

        self.assertRtolEqual(grad_in_npu.cpu(), golden_grad_in)
        self.assertRtolEqual(grad_w_npu.cpu(), golden_grad_w)

    @unittest.skip("Skip test_npu_masked_causal_conv1d_backward now")
    @SupportedDevices(['Ascend950'])
    def test_npu_masked_causal_conv1d_backward_bf16_no_mask(self):
        S, B, H, W = 1024, 2, 192, 3
        dtype = torch.bfloat16

        grad_output = torch.randn(S, B, H, dtype=dtype)
        x = torch.randn(S, B, H, dtype=dtype)
        weight = torch.randn(W, H, dtype=dtype)

        golden_grad_in, golden_grad_w = masked_causal_conv1d_backward_golden(
            grad_output, x, weight, mask=None
        )

        grad_in_npu, grad_w_npu = torch_npu.npu_masked_causal_conv1d_backward(
            grad_output.npu(), x.npu(), weight.npu()
        )
        torch.npu.synchronize()

        self.assertRtolEqual(grad_in_npu.cpu(), golden_grad_in)
        self.assertRtolEqual(grad_w_npu.cpu(), golden_grad_w)

    @unittest.skip("Skip test_npu_masked_causal_conv1d_backward now")
    @SupportedDevices(['Ascend950'])
    def test_npu_masked_causal_conv1d_autograd(self):
        """Verify backward is invoked correctly via autograd."""
        S, B, H, W = 256, 2, 128, 3
        dtype = torch.bfloat16

        x = torch.randn(S, B, H, dtype=dtype, requires_grad=True).npu()
        weight = torch.randn(W, H, dtype=dtype, requires_grad=True).npu()
        mask = (torch.rand(B, S) > 0.3).npu()

        out = torch_npu.npu_masked_causal_conv1d(x, weight, mask=mask)
        out.sum().backward()
        torch.npu.synchronize()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(weight.grad)
        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual(weight.grad.shape, weight.shape)


if __name__ == "__main__":
    run_tests()

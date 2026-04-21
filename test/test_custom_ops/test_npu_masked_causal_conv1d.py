import unittest
import torch
import torch.nn.functional as F
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


# Golden reference — aligned with test_masked_causal_conv1d.py in pta/
def _fwd_f32(x_f32, weight_f32, mask):
    S, B, H = x_f32.shape
    W = weight_f32.shape[0]
    conv = torch.nn.Conv1d(in_channels=H, out_channels=H,
                           kernel_size=W, groups=H, bias=False)
    conv.weight.data = weight_f32.transpose(0, 1).unsqueeze(1)
    pad = torch.zeros((B, H, W - 1), dtype=torch.float32)
    conv_input = torch.cat([pad, x_f32.permute(1, 2, 0)], dim=-1)
    out = conv(conv_input).permute(0, 2, 1)  # [B, S, H]
    if mask is not None:
        out[~mask] = 0.0
    return out.transpose(0, 1).contiguous()  # [S, B, H]


def masked_causal_conv1d_golden(x, weight, mask=None):
    out_dtype = x.dtype
    y = _fwd_f32(x.float().cpu(), weight.float().cpu(),
                 mask.cpu() if mask is not None else None)
    return y.to(out_dtype)


class TestNpuMaskedCausalConv1d(TestCase):
    @unittest.skip("Skip test_npu_masked_causal_conv1d now")
    @SupportedDevices(['Ascend950'])
    def test_npu_masked_causal_conv1d_bf16_with_mask(self):
        S, B, H, W = 512, 4, 128, 3
        dtype = torch.bfloat16

        x = torch.randn(S, B, H, dtype=dtype)
        weight = torch.randn(W, H, dtype=dtype)
        mask = torch.rand(B, S) > 0.3  # bool [B, S]

        golden = masked_causal_conv1d_golden(x, weight, mask)

        out_npu = torch_npu.npu_masked_causal_conv1d(
            x.npu(), weight.npu(), mask=mask.npu()
        )
        torch.npu.synchronize()

        self.assertRtolEqual(out_npu.cpu(), golden)

    @unittest.skip("Skip test_npu_masked_causal_conv1d now")
    @SupportedDevices(['Ascend950'])
    def test_npu_masked_causal_conv1d_fp16_with_mask(self):
        S, B, H, W = 512, 4, 128, 3
        dtype = torch.float16

        x = torch.randn(S, B, H, dtype=dtype)
        weight = torch.randn(W, H, dtype=dtype)
        mask = torch.rand(B, S) > 0.3

        golden = masked_causal_conv1d_golden(x, weight, mask)

        out_npu = torch_npu.npu_masked_causal_conv1d(
            x.npu(), weight.npu(), mask=mask.npu()
        )
        torch.npu.synchronize()

        self.assertRtolEqual(out_npu.cpu(), golden)

    @unittest.skip("Skip test_npu_masked_causal_conv1d now")
    @SupportedDevices(['Ascend950'])
    def test_npu_masked_causal_conv1d_bf16_no_mask(self):
        S, B, H, W = 1024, 2, 192, 3
        dtype = torch.bfloat16

        x = torch.randn(S, B, H, dtype=dtype)
        weight = torch.randn(W, H, dtype=dtype)

        golden = masked_causal_conv1d_golden(x, weight, mask=None)

        out_npu = torch_npu.npu_masked_causal_conv1d(
            x.npu(), weight.npu()
        )
        torch.npu.synchronize()

        self.assertRtolEqual(out_npu.cpu(), golden)


if __name__ == "__main__":
    run_tests()

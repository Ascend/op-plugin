import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


# 以下 4 个算子没有反向实现，已在 AutogradPrivateUse1 上注册
# autogradNotImplementedFallback：前向正常，对输出调用 .backward() 时应抛出
# RuntimeError: derivative for 'npu::xxx' is not implemented。
# 仅在 A5 / Ascend950 上验证，其它机型由 SupportedDevices 自动跳过。
class TestAutogradErrorFallback(TestCase):

    def _leaf(self, *shape, dtype=torch.float16):
        x = torch.randn(*shape, dtype=dtype).npu()
        x.requires_grad_(True)
        return x

    @SupportedDevices(['Ascend950'])
    def test_npu_add_rms_norm_backward_raises(self):
        x1 = self._leaf(4, 16)
        x2 = self._leaf(4, 16)
        gamma = self._leaf(16)
        out = torch_npu.npu_add_rms_norm(x1, x2, gamma, 1e-6)[0]
        self.assertRaisesRegex(
            RuntimeError, "not implemented",
            lambda: out.float().sum().backward())

    @SupportedDevices(['Ascend950'])
    def test_npu_apply_rotary_pos_emb_backward_raises(self):
        q = self._leaf(1, 4, 2, 16)
        k = self._leaf(1, 4, 2, 16)
        cos = torch.randn(1, 4, 1, 16, dtype=torch.float16).npu()
        sin = torch.randn(1, 4, 1, 16, dtype=torch.float16).npu()
        out = torch_npu.npu_apply_rotary_pos_emb(q, k, cos, sin, "BSND", "half")[0]
        self.assertRaisesRegex(
            RuntimeError, "not implemented",
            lambda: out.float().sum().backward())

    @SupportedDevices(['Ascend950'])
    def test_npu_interleave_rope_backward_raises(self):
        x = self._leaf(1, 4, 2, 16)
        cos = torch.randn(1, 4, 1, 16, dtype=torch.float16).npu()
        sin = torch.randn(1, 4, 1, 16, dtype=torch.float16).npu()
        out = torch_npu.npu_interleave_rope(x, cos, sin)
        self.assertRaisesRegex(
            RuntimeError, "not implemented",
            lambda: out.float().sum().backward())

    @SupportedDevices(['Ascend950'])
    def test_npu_moe_gating_top_k_softmax_backward_raises(self):
        x = self._leaf(8, 16)
        out = torch_npu.npu_moe_gating_top_k_softmax(x, None, 2)[0]
        self.assertRaisesRegex(
            RuntimeError, "not implemented",
            lambda: out.float().sum().backward())


if __name__ == "__main__":
    run_tests()

import unittest

import torch

import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


# _nested_tensor_from_mask 返回 NestedTensor，当前 NPU 尚未完整支持嵌套张量
# （提交说明亦标注 "npu incomplete support nested tensor"）：CPU fallback 将 CPU 侧
# 生成的嵌套张量拷贝回 NPU 时缺少 NestedTensornpu 的 _to_copy 后端，会抛
# NotImplementedError。待 NPU 支持嵌套张量后，移除本 skip 并恢复 CPU/NPU 对比。
@unittest.skip("NPU does not support nested tensor output yet")
class TestNestedTensorFromMask(TestCase):
    # ATen 算子 _nested_tensor_from_mask(t, mask, mask_check=True) -> Tensor：
    # 按 padding mask（True == 保留，False == padding）把 dense tensor 转为 nested tensor，
    # 供上层 SDPA / Transformer 的 nested-tensor 快速路径使用。
    # 通过 to_padded_tensor 把 nested 转回 dense 后做数值比对。

    def _op(self, t, mask, mask_check=True):
        return torch.ops.aten._nested_tensor_from_mask(t, mask, mask_check)

    def _make_input(self, batch, seq_len, dim, valid_len):
        # 每行前 valid_len 个位置 True（保留），右侧 False（padding），左对齐。
        t = torch.randn(batch, seq_len, dim)
        mask = torch.zeros((batch, seq_len), dtype=torch.bool)
        if valid_len > 0:
            mask[:, :valid_len] = True
        return t, mask

    def _padded(self, nested):
        return torch.nested.to_padded_tensor(nested, 0.0)

    def test_basic(self):
        batch, seq_len, dim, valid_len = 2, 8, 16, 5
        t, mask = self._make_input(batch, seq_len, dim, valid_len)

        cpu_out = self._op(t, mask)
        npu_out = self._op(t.npu(), mask.npu())

        self.assertTrue(cpu_out.is_nested)
        self.assertTrue(npu_out.is_nested)
        self.assertEqual(self._padded(npu_out).cpu(), self._padded(cpu_out))

    def test_fully_valid(self):
        batch, seq_len, dim = 2, 8, 16
        t, mask = self._make_input(batch, seq_len, dim, valid_len=seq_len)

        cpu_out = self._op(t, mask)
        npu_out = self._op(t.npu(), mask.npu())

        self.assertTrue(cpu_out.is_nested)
        self.assertTrue(npu_out.is_nested)
        self.assertEqual(self._padded(npu_out).cpu(), self._padded(cpu_out))


if __name__ == "__main__":
    run_tests()

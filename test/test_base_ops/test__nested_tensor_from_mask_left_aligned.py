import numpy as np
import torch

import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestNestedTensorFromMaskLeftAligned(TestCase):
    # ATen 算子 _nested_tensor_from_mask_left_aligned(t, mask) -> bool：
    # 探测 mask 是否"左对齐"（每行 True 连续且无空洞，仅右侧为 padding），
    # 用于判断上层 SDPA 能否走 nested-tensor 快速路径。
    # 注意此处 mask 语义与 attention 相反：True == 有效（非 padding），False == padding。
    # NPU 实现委托 ATen 原生函数（组合算子实现），语义与 CPU 一致：
    # 左对齐（含全有效）返回 True，带空洞返回 False。

    def _op(self, t, mask):
        return torch.ops.aten._nested_tensor_from_mask_left_aligned(t, mask)

    def _left_aligned_mask(self, batch, seq_len, valid_len):
        # True 在每行左侧连续排列，右侧为 False（padding），即左对齐。
        mask = np.zeros((batch, seq_len), dtype=np.bool_)
        if valid_len > 0:
            mask[:, :valid_len] = True
        return mask

    def _non_left_aligned_mask(self, batch, seq_len):
        # 每行 True 不连续（首尾各一个 True），存在空洞，非左对齐。
        mask = np.zeros((batch, seq_len), dtype=np.bool_)
        mask[:, 0] = True
        mask[:, -1] = True
        return mask

    def _make_input(self, batch, seq_len, dim, mask):
        return torch.randn(batch, seq_len, dim), torch.from_numpy(mask)

    def test_left_aligned_mask(self):
        batch, seq_len, dim = 2, 8, 16
        mask = self._left_aligned_mask(batch, seq_len, valid_len=5)
        t, m = self._make_input(batch, seq_len, dim, mask)

        cpu_out = self._op(t, m)
        npu_out = self._op(t.npu(), m.npu())

        # 左对齐 mask：CPU 与 NPU 均返回 True。
        self.assertEqual(cpu_out, True)
        self.assertEqual(npu_out, cpu_out)

    def test_non_left_aligned_mask(self):
        batch, seq_len, dim = 2, 8, 16
        mask = self._non_left_aligned_mask(batch, seq_len)
        t, m = self._make_input(batch, seq_len, dim, mask)

        cpu_out = self._op(t, m)
        npu_out = self._op(t.npu(), m.npu())

        # 带空洞 mask：CPU 与 NPU 均返回 False。
        self.assertEqual(cpu_out, False)
        self.assertEqual(npu_out, cpu_out)

    def test_fully_valid_mask(self):
        batch, seq_len, dim = 2, 8, 16
        mask = self._left_aligned_mask(batch, seq_len, valid_len=seq_len)
        t, m = self._make_input(batch, seq_len, dim, mask)

        cpu_out = self._op(t, m)
        npu_out = self._op(t.npu(), m.npu())

        # 全 True 也属于左对齐：CPU 与 NPU 均返回 True。
        self.assertEqual(cpu_out, True)
        self.assertEqual(npu_out, cpu_out)


if __name__ == "__main__":
    run_tests()

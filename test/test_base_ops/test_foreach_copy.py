import unittest
import random
import torch
import torch_npu
import numpy as np
import traceback
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestForeachCopy(TestCase):
    @SupportedDevices(['Ascend910B'])
    def test_foreach_copy_out_bfloat16_shpae_tensor_num(self):
        q1 = torch.rand([2,3,4], device="npu").to(torch.float16)
        q2 = torch.rand([2,3,4], device="npu").to(torch.float16)
        k1 = torch.zeros([1,2,3], device="cpu", dtype=torch.int64)
        k2 = torch.zeros([1,2,3], device="cpu", dtype=torch.int64)
        dst_tensors = []
        src_tensors = []
        dst_tensors.append(q1)
        dst_tensors.append(k1)
        src_tensors.append(q2)
        src_tensors.append(k2)

        try:
            torch._foreach_copy_(dst_tensors, src_tensors)
        except Exception:
            traceback.print_exc()
            raise AssertionError("foreach copy failed, test won't pass")

    @SupportedDevices(['Ascend910B'])
    def test_foreach_copy_non_contiguous_src_d2d(self):
        # src 非连续 (slice), dst 连续, D2D
        base = torch.rand([4, 8], device="npu")
        src = base[:, 0:4]  # shape (4,4), 非连续
        dst = torch.zeros([4, 4], device="npu")
        self.assertFalse(src.is_contiguous())
        self.assertTrue(dst.is_contiguous())

        try:
            torch._foreach_copy_([dst], [src])
        except Exception:
            traceback.print_exc()
            raise AssertionError("non-contiguous src D2D foreach copy failed")
        self.assertRtolEqual(dst.cpu(), base[:, 0:4].cpu())

    @SupportedDevices(['Ascend910B'])
    def test_foreach_copy_non_contiguous_dst_d2d(self):
        # dst 非连续 (slice), src 连续, D2D
        base = torch.zeros([4, 8], device="npu")
        dst = base[:, 0:4]  # shape (4,4), 非连续
        src = torch.rand([4, 4], device="npu")
        self.assertFalse(dst.is_contiguous())
        self.assertTrue(src.is_contiguous())

        try:
            torch._foreach_copy_([dst], [src])
        except Exception:
            traceback.print_exc()
            raise AssertionError("non-contiguous dst D2D foreach copy failed")
        self.assertRtolEqual(base[:, 0:4].cpu(), src.cpu())

    @SupportedDevices(['Ascend910B'])
    def test_foreach_copy_non_contiguous_both_d2d(self):
        # src 和 dst 都非连续, D2D
        base_src = torch.rand([4, 8], device="npu")
        base_dst = torch.zeros([4, 8], device="npu")
        src = base_src[:, 0:4]
        dst = base_dst[:, 0:4]
        self.assertFalse(src.is_contiguous())
        self.assertFalse(dst.is_contiguous())

        try:
            torch._foreach_copy_([dst], [src])
        except Exception:
            traceback.print_exc()
            raise AssertionError("non-contiguous both D2D foreach copy failed")
        self.assertRtolEqual(base_dst[:, 0:4].cpu(), base_src[:, 0:4].cpu())

    @SupportedDevices(['Ascend910B'])
    def test_foreach_copy_non_contiguous_transpose_d2d(self):
        # src 非连续 (transpose), dst 连续, D2D
        src = torch.rand([4, 8], device="npu").transpose(0, 1)  # shape (8,4), 非连续
        dst = torch.zeros([8, 4], device="npu")
        self.assertFalse(src.is_contiguous())

        try:
            torch._foreach_copy_([dst], [src])
        except Exception:
            traceback.print_exc()
            raise AssertionError("non-contiguous transpose src foreach copy failed")
        self.assertRtolEqual(dst.cpu(), src.cpu())

    @SupportedDevices(['Ascend910B'])
    def test_foreach_copy_non_contiguous_src_d2h(self):
        # src 非连续 (slice), dst 连续, D2H
        base = torch.rand([4, 8], device="npu")
        src = base[:, 0:4]  # shape (4,4), 非连续
        dst = torch.zeros([4, 4], device="cpu")
        self.assertFalse(src.is_contiguous())

        try:
            torch._foreach_copy_([dst], [src])
        except Exception:
            traceback.print_exc()
            raise AssertionError("non-contiguous src D2H foreach copy failed")
        self.assertRtolEqual(dst, base[:, 0:4].cpu())

    @SupportedDevices(['Ascend910B'])
    def test_foreach_copy_non_contiguous_src_h2d(self):
        # src 非连续 (slice), dst 连续, H2D
        base = torch.rand([4, 8], device="cpu")
        src = base[:, 0:4]  # shape (4,4), 非连续
        dst = torch.zeros([4, 4], device="npu")
        self.assertFalse(src.is_contiguous())

        try:
            torch._foreach_copy_([dst], [src])
        except Exception:
            traceback.print_exc()
            raise AssertionError("non-contiguous src H2D foreach copy failed")
        self.assertRtolEqual(dst.cpu(), base[:, 0:4])

    @SupportedDevices(['Ascend910B'])
    def test_foreach_copy_non_contiguous_strided(self):
        # src 非连续 (strided), dst 连续, D2D
        base = torch.rand([2, 4, 8], device="npu")
        src = base[0, 0:2, 0:4]  # shape (2,4), 非连续
        dst = torch.zeros([2, 4], device="npu")
        self.assertFalse(src.is_contiguous())

        try:
            torch._foreach_copy_([dst], [src])
        except Exception:
            traceback.print_exc()
            raise AssertionError("non-contiguous strided src foreach copy failed")
        self.assertRtolEqual(dst.cpu(), base[0, 0:2, 0:4].cpu())

    @SupportedDevices(['Ascend910B'])
    def test_foreach_copy_mixed_contiguous_non_contiguous(self):
        # 混合场景: 一个非连续 tensor + 一个连续 tensor 在同一个列表中
        base_src = torch.rand([4, 8], device="npu")
        base_dst = torch.zeros([4, 8], device="npu")
        src_non_contig = base_src[:, 0:4]  # 非连续
        dst_non_contig = base_dst[:, 0:4]  # 非连续

        src_contig = torch.rand([2, 3], device="npu")  # 连续
        dst_contig = torch.zeros([2, 3], device="npu")  # 连续

        self.assertFalse(src_non_contig.is_contiguous())
        self.assertFalse(dst_non_contig.is_contiguous())
        self.assertTrue(src_contig.is_contiguous())
        self.assertTrue(dst_contig.is_contiguous())

        try:
            torch._foreach_copy_([dst_non_contig, dst_contig], [src_non_contig, src_contig])
        except Exception:
            traceback.print_exc()
            raise AssertionError("mixed contiguous/non-contiguous foreach copy failed")
        self.assertRtolEqual(base_dst[:, 0:4].cpu(), base_src[:, 0:4].cpu())
        self.assertRtolEqual(dst_contig.cpu(), src_contig.cpu())


if __name__ == "__main__":
    run_tests()

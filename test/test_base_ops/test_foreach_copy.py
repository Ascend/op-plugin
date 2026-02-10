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


if __name__ == "__main__":
    run_tests()
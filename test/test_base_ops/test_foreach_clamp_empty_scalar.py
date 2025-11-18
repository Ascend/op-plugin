import unittest
import torch
import torch_npu
import hypothesis
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestForeachClampEmptyScalar(TestCase):

    @SupportedDevices(['Ascend910B'])
    def test_foreach_clamp_max_empty_scalar_list(self):
        tensors1 = []
        tensors2 = []
        device = torch.device("npu")
        with self.assertRaisesRegex(
            RuntimeError, "Tensor list must have at least one tensor."
        ):
            torch._foreach_clamp_max_(tensors1, tensors2)

        with self.assertRaisesRegex(
            RuntimeError, "Tensor list must have at least one tensor."
        ):
            torch._foreach_clamp_min_(tensors1, tensors2)

        # One empty list
        tensors1.append(torch.tensor([1], device=device))
        with self.assertRaisesRegex(
            RuntimeError,
            "Tensor list must have same number of elements as scalar list.",
        ):
            torch._foreach_clamp_max_(tensors1, tensors2)

        with self.assertRaisesRegex(
            RuntimeError,
            "Tensor list must have same number of elements as scalar list.",
        ):
            torch._foreach_clamp_min_(tensors1, tensors2)

if __name__ == "__main__":
    run_tests()

import itertools

import torch
import torch_npu  # noqa: F401
from torch_npu.testing.testcase import TestCase, run_tests


class TestMaxMinEmptyDim(TestCase):
    def test_empty_reduction_dimension(self):
        for op_name, shape, dim, keepdim, use_out in itertools.product(
            ("max", "min"), ((0, 3), (2, 0, 3)), (0, -2), (False, True), (False, True)
        ):
            # Both dimension spellings select the zero-sized axis.
            dim = dim if len(shape) == 2 else (1 if dim == 0 else -2)
            with self.subTest(
                op=op_name, shape=shape, dim=dim, keepdim=keepdim, out=use_out
            ):
                for device in ("cpu", "npu"):
                    x = torch.empty(shape, device=device)
                    kwargs = {"dim": dim, "keepdim": keepdim}
                    if use_out:
                        values = torch.full((1,), 17.0, device=device)
                        indices = torch.full((1,), 19, dtype=torch.int64, device=device)
                        kwargs["out"] = (values, indices)
                    with self.assertRaisesRegex(
                        IndexError, "Expected reduction dim .*non-zero size"
                    ):
                        getattr(torch, op_name)(x, **kwargs)
                    if use_out:
                        self.assertEqual(values.cpu(), torch.tensor([17.0]))
                        self.assertEqual(indices.cpu(), torch.tensor([19]))

    def test_empty_non_reduction_dimension(self):
        for op_name, dim, keepdim, use_out in itertools.product(
            ("max", "min"), (1, -2), (False, True), (False, True)
        ):
            with self.subTest(op=op_name, dim=dim, keepdim=keepdim, out=use_out):
                cpu = torch.empty((0, 3, 2))
                expected = getattr(torch, op_name)(cpu, dim=dim, keepdim=keepdim)
                kwargs = {"dim": dim, "keepdim": keepdim}
                if use_out:
                    kwargs["out"] = (
                        torch.empty_like(expected.values, device="npu"),
                        torch.empty_like(expected.indices, device="npu"),
                    )
                actual = getattr(torch, op_name)(cpu.npu(), **kwargs)
                self.assertEqual(actual.values.cpu(), expected.values)
                self.assertEqual(actual.indices.cpu(), expected.indices)

    def test_scalar_and_nonempty(self):
        for op_name, cpu, dim, keepdim in itertools.product(
            ("max", "min"),
            (torch.tensor(3.0), torch.tensor([[1.0, 4.0], [3.0, 2.0]])),
            (0, -1),
            (False, True),
        ):
            with self.subTest(op=op_name, shape=cpu.shape, dim=dim, keepdim=keepdim):
                expected = getattr(torch, op_name)(cpu, dim=dim, keepdim=keepdim)
                actual = getattr(torch, op_name)(cpu.npu(), dim=dim, keepdim=keepdim)
                self.assertEqual(actual.values.cpu(), expected.values)
                self.assertEqual(actual.indices.cpu(), expected.indices)

    def test_invalid_dimension(self):
        for op_name, dim in itertools.product(("max", "min"), (-3, 2)):
            with self.subTest(op=op_name, dim=dim), self.assertRaises(IndexError):
                getattr(torch, op_name)(torch.empty((0, 3), device="npu"), dim=dim)


if __name__ == "__main__":
    run_tests()

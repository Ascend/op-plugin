import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestLdexp(TestCase):
    def assert_ldexp_equal(self, cpu_result, npu_result):
        self.assertRtolEqual(
            cpu_result.to(torch.float32).numpy(),
            npu_result.cpu().to(torch.float32).numpy(),
        )

    def test_ldexp_tensor_variants(self):
        cases = (
            (torch.float16, torch.int64),
            (torch.bfloat16, torch.int32),
            (torch.float32, torch.int64),
        )
        values = torch.tensor(
            [[0.5, -1.25, 2.0], [4.0, -0.125, 1.5]], dtype=torch.float32
        )
        exponents = torch.tensor([[0, 1, -2], [3, -3, 2]], dtype=torch.int64)

        for value_dtype, exponent_dtype in cases:
            cpu_values = values.to(value_dtype)
            cpu_exponents = exponents.to(exponent_dtype)
            npu_values = cpu_values.npu()
            npu_exponents = cpu_exponents.npu()

            self.assert_ldexp_equal(
                torch.ldexp(cpu_values, cpu_exponents),
                torch.ldexp(npu_values, npu_exponents),
            )
            self.assert_ldexp_equal(
                cpu_values.ldexp(cpu_exponents),
                npu_values.ldexp(npu_exponents),
            )

            cpu_out = torch.empty_like(cpu_values)
            npu_out = torch.empty_like(npu_values)
            torch.ldexp(cpu_values, cpu_exponents, out=cpu_out)
            torch.ldexp(npu_values, npu_exponents, out=npu_out)
            self.assert_ldexp_equal(cpu_out, npu_out)

            cpu_inplace = cpu_values.clone()
            npu_inplace = npu_values.clone()
            cpu_inplace.ldexp_(cpu_exponents)
            npu_inplace.ldexp_(npu_exponents)
            self.assert_ldexp_equal(cpu_inplace, npu_inplace)

    def test_ldexp_broadcast_and_non_contiguous(self):
        cpu_values = torch.tensor(
            [[0.5, 1.0, 2.0], [-4.0, 0.25, 8.0]], dtype=torch.float32
        ).t()
        cpu_exponents = torch.tensor([[-2, 0, 3]], dtype=torch.int64).t()
        npu_values = cpu_values.npu()
        npu_exponents = cpu_exponents.npu()

        self.assertFalse(cpu_values.is_contiguous())
        self.assertFalse(npu_values.is_contiguous())
        self.assert_ldexp_equal(
            torch.ldexp(cpu_values, cpu_exponents),
            torch.ldexp(npu_values, npu_exponents),
        )

        cpu_out = torch.empty((2, 3), dtype=torch.float32).t()
        npu_out = torch.empty((2, 3), dtype=torch.float32, device="npu").t()
        torch.ldexp(cpu_values, cpu_exponents, out=cpu_out)
        torch.ldexp(npu_values, npu_exponents, out=npu_out)
        self.assertFalse(cpu_out.is_contiguous())
        self.assertFalse(npu_out.is_contiguous())
        self.assert_ldexp_equal(cpu_out, npu_out)

        cpu_scalar_exponent = torch.tensor(-2, dtype=torch.int64)
        npu_scalar_exponent = cpu_scalar_exponent.npu()
        self.assert_ldexp_equal(
            torch.ldexp(cpu_values, cpu_scalar_exponent),
            torch.ldexp(npu_values, npu_scalar_exponent),
        )

    def test_ldexp_special_values(self):
        cpu_values = torch.tensor(
            [0.0, -0.0, float("inf"), -float("inf"), 1.0, -1.0],
            dtype=torch.float32,
        )
        cpu_exponents = torch.tensor([10, -10, 3, -3, 127, -149], dtype=torch.int64)
        npu_result = torch.ldexp(cpu_values.npu(), cpu_exponents.npu()).cpu()
        cpu_result = torch.ldexp(cpu_values, cpu_exponents)

        self.assertEqual(torch.isnan(npu_result), torch.isnan(cpu_result))
        self.assertEqual(torch.isinf(npu_result), torch.isinf(cpu_result))
        self.assertEqual(torch.signbit(npu_result), torch.signbit(cpu_result))
        finite_mask = torch.isfinite(cpu_result) & torch.isfinite(npu_result)
        self.assertRtolEqual(
            cpu_result[finite_mask].numpy(),
            npu_result[finite_mask].numpy(),
        )

    def test_ldexp_backward(self):
        cpu_values = torch.tensor(
            [0.5, -1.25, 2.0], dtype=torch.float32, requires_grad=True
        )
        cpu_exponents = torch.tensor([0, 2, -3], dtype=torch.int64)
        npu_values = cpu_values.detach().npu().requires_grad_(True)
        npu_exponents = cpu_exponents.npu()

        torch.ldexp(cpu_values, cpu_exponents).sum().backward()
        torch.ldexp(npu_values, npu_exponents).sum().backward()
        self.assert_ldexp_equal(cpu_values.grad, npu_values.grad)


if __name__ == "__main__":
    run_tests()

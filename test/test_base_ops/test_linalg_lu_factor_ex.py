import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestLinalgLuFactorEx(TestCase):
    def test_linalg_lu_factor_ex_backward_amp(self):
        torch.manual_seed(1234)
        input_x = torch.randn(3, 5, dtype=torch.float32).npu()
        input_x.requires_grad_(True)

        with torch.npu.amp.autocast():
            lu, _, _ = torch.linalg.lu_factor_ex(input_x)
            grad = torch.autograd.grad(lu, input_x, torch.ones_like(lu), retain_graph=True)[0]

        self.assertEqual(lu.shape, input_x.shape)
        self.assertEqual(grad.shape, input_x.shape)
        self.assertEqual(grad.dtype, input_x.dtype)


if __name__ == "__main__":
    run_tests()

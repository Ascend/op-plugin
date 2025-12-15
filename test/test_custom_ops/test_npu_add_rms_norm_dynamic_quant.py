import unittest
import math
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestAddRmsNormDynamicQuant(TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)

    def compare(self, a: torch.Tensor, b: torch.Tensor, benchmark: float) -> bool:
        a = a.reshape(-1).cpu()
        b = b.reshape(-1).cpu()
        diff_abs = torch.abs(a - b)
        if diff_abs.numel() == 0:
            return True
        max_diff_abs = diff_abs.max().item()
        if max_diff_abs == 0:
            return True

        rel_error = 0
        abs_error = 0
        for i in range(a.numel()):
            ai = float(a[i].item())
            bi = float(b[i].item())
            diff = abs(ai - bi)

            if ai == 0.0 and bi == 0.0:
                diff_rel_item = 0.0
            elif ai == 0.0 or bi == 0.0:
                diff_rel_item = 1.0
            else:
                diff_rel_item = diff / abs(ai)

            if abs(ai) < 1 and diff > benchmark:
                abs_error += 1
            elif abs(ai) >= 1 and diff_rel_item > benchmark:
                rel_error += 1

            if (rel_error + abs_error) > 10:
                break

        return (rel_error + abs_error) == 0

    def npu_add_rms_norm_dynamic_quant_golden(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        gamma: torch.Tensor,
        smooth_scale1: torch.Tensor = None,
        smooth_scale2: torch.Tensor = None,
        beta: torch.Tensor = None,
        epsilon: float = 1e-6,
        output_mask=None,
    ):
        assert x1.shape == x2.shape
        last_dim = x1.shape[-1]
        assert gamma.dim() == 1 and gamma.shape[0] == last_dim
        if smooth_scale1 is not None:
            assert smooth_scale1.dim() == 1 and smooth_scale1.shape[0] == last_dim
            smooth_scale1 = smooth_scale1.detach().cpu()
        if smooth_scale2 is not None:
            assert smooth_scale2.dim() == 1 and smooth_scale2.shape[0] == last_dim
            smooth_scale2 = smooth_scale2.detach().cpu()
        if beta is not None:
            assert beta.dim() == 1 and beta.shape[0] == last_dim
            beta = beta.detach().cpu()

        x1_fp = x1.detach().cpu().to(torch.float32)
        x2_fp = x2.detach().cpu().to(torch.float32)
        gamma_fp = gamma.detach().cpu().to(torch.float32)

        x = x1_fp + x2_fp
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + float(epsilon))
        y = (x / rms) * gamma_fp
        if beta is not None:
            y = y + beta.to(torch.float32)

        def row_max_abs(t: torch.Tensor):
            return t.abs().amax(dim=-1, keepdim=True)

        no_mask = (output_mask is None)
        mask0 = True if no_mask else bool(output_mask[0])
        mask1 = None if no_mask else bool(output_mask[1])

        if mask0:
            input1 = y if smooth_scale1 is None else y * smooth_scale1.to(torch.float32)
            scale1 = row_max_abs(input1) / 127.0
            scale1 = torch.where(scale1 > 0, scale1, torch.ones_like(scale1))
            q1 = torch.round(input1 / scale1).to(torch.int32)
            q1 = torch.clamp(q1, -128, 127).to(torch.int8)
            s1_out = scale1.squeeze(-1).to(torch.float32).contiguous()
        else:
            q1 = None
            s1_out = None

        compute_branch2 = False
        if no_mask:
            if (smooth_scale1 is not None) and (smooth_scale2 is not None):
                compute_branch2 = True
            else:
                compute_branch2 = False
        else:
            if mask1:
                compute_branch2 = True
            else:
                compute_branch2 = False

        if compute_branch2:
            input2 = y if smooth_scale2 is None else y * smooth_scale2.to(torch.float32)
            scale2 = row_max_abs(input2) / 127.0
            scale2 = torch.where(scale2 > 0, scale2, torch.ones_like(scale2))
            q2 = torch.round(input2 / scale2).to(torch.int32)
            q2 = torch.clamp(q2, -128, 127).to(torch.int8)
            s2_out = scale2.squeeze(-1).to(torch.float32).contiguous()
        else:
            q2 = None
            s2_out = None

        x_out = x.to(x1.dtype)
        return q1, q2, x_out, s1_out, s2_out

    def _run_and_check(
        self,
        x1,
        x2,
        gamma,
        smooth_scale1=None,
        smooth_scale2=None,
        beta=None,
        epsilon=1e-6,
        output_mask=None,
    ):
        for t in [x1, x2, gamma, smooth_scale1, smooth_scale2, beta]:
            if t is None:
                continue
            assert t.dtype in (torch.float16, torch.bfloat16)

        x1_n = x1.npu()
        x2_n = x2.npu()
        gamma_n = gamma.npu()
        s1_n = smooth_scale1.npu() if smooth_scale1 is not None else None
        s2_n = smooth_scale2.npu() if smooth_scale2 is not None else None
        beta_n = beta.npu() if beta is not None else None
        eps_f = float(epsilon)
        if output_mask is None:
            output_mask = [True, True]

        y1_npu, y2_npu, x_out_npu, s1_npu, s2_npu = torch_npu.npu_add_rms_norm_dynamic_quant(
            x1_n, x2_n, gamma_n,
            smooth_scale1=s1_n,
            smooth_scale2=s2_n,
            beta=beta_n,
            epsilon=eps_f,
            output_mask=output_mask,
        )

        y1_cpu, y2_cpu, x_out_cpu, s1_cpu, s2_cpu = self.npu_add_rms_norm_dynamic_quant_golden(
            x1, x2, gamma, smooth_scale1, smooth_scale2, beta, epsilon, output_mask
        )

        self.assertEqual(x_out_npu.dtype, x1.dtype)
        self.assertEqual(tuple(x_out_npu.shape), tuple(x1.shape))

        if output_mask[0]:
            self.assertEqual(y1_npu.dtype, torch.int8)
            self.assertEqual(s1_npu.dtype, torch.float32)
            self.assertEqual(tuple(y1_npu.shape), tuple(x1.shape))
            self.assertEqual(tuple(s1_npu.shape), tuple(x1.shape[:-1]))
        else:
            self.assertTrue(isinstance(y1_npu, torch.Tensor))
            self.assertTrue(y1_npu.numel() == 1, f"Expected y1 numel=1 when mask[0]=False, got {y1_npu.numel()}")
            self.assertTrue(isinstance(s1_npu, torch.Tensor))
            self.assertTrue(s1_npu.numel() == 1, f"Expected s1 numel=1 when mask[0]=False, got {s1_npu.numel()}")

        if output_mask[1]:
            self.assertEqual(y2_npu.dtype, torch.int8)
            self.assertEqual(s2_npu.dtype, torch.float32)
            self.assertEqual(tuple(y2_npu.shape), tuple(x1.shape))
            self.assertEqual(tuple(s2_npu.shape), tuple(x1.shape[:-1]))
        else:
            self.assertTrue(isinstance(y2_npu, torch.Tensor))
            self.assertTrue(y2_npu.numel() == 1, f"Expected y2 numel=1 when mask[1]=False, got {y2_npu.numel()}")
            self.assertTrue(isinstance(s2_npu, torch.Tensor))
            self.assertTrue(s2_npu.numel() == 1, f"Expected s2 numel=1 when mask[1]=False, got {s2_npu.numel()}")

        benchmark = math.pow(2, -7)

        x_out_npu_flat = x_out_npu.reshape(-1).cpu().to(torch.float32)
        x_out_cpu_flat = x_out_cpu.reshape(-1).cpu().to(torch.float32)
        self.assertTrue(self.compare(x_out_cpu_flat, x_out_npu_flat, benchmark))

        if output_mask[0]:
            y1_diff = (y1_npu.cpu().to(torch.int16) - y1_cpu.cpu().to(torch.int16)).abs()
            self.assertTrue(int(y1_diff.max()) <= 1, f"max |y1_npu - y1_ref| = {int(y1_diff.max())} > 1")

            s1_npu_flat = s1_npu.reshape(-1).cpu().to(torch.float32)
            s1_cpu_flat = s1_cpu.reshape(-1).cpu().to(torch.float32)
            self.assertTrue(self.compare(s1_cpu_flat, s1_npu_flat, benchmark))

        if output_mask[1]:
            y2_diff = (y2_npu.cpu().to(torch.int16) - y2_cpu.cpu().to(torch.int16)).abs()
            self.assertTrue(int(y2_diff.max()) <= 1, f"max |y2_npu - y2_ref| = {int(y2_diff.max())} > 1")

            s2_npu_flat = s2_npu.reshape(-1).cpu().to(torch.float32)
            s2_cpu_flat = s2_cpu.reshape(-1).cpu().to(torch.float32)
            self.assertTrue(self.compare(s2_cpu_flat, s2_npu_flat, benchmark))

        for t in [x_out_npu]:
            tt = t.float()
            self.assertFalse(torch.isnan(tt).any().item())
            self.assertFalse(torch.isinf(tt).any().item())

        if output_mask[0]:
            for t in [s1_npu]:
                tt = t.float()
                self.assertFalse(torch.isnan(tt).any().item())
                self.assertFalse(torch.isinf(tt).any().item())

        if output_mask[1]:
            for t in [s2_npu]:
                tt = t.float()
                self.assertFalse(torch.isnan(tt).any().item())
                self.assertFalse(torch.isinf(tt).any().item())


    @SupportedDevices(['Ascend910B'])
    def test_forward_various_shapes(self):
        shape_list = [
            [2, 8],
            [3, 16],
            [4, 5, 32],
            [2, 3, 4, 24],
            [2, 3, 4, 5, 64],
            [2, 3, 4, 5, 6, 128],
            [2, 3, 4, 5, 6, 7, 256],
            [2, 3, 4, 5, 6, 7, 8, 512],
        ]
        for x_shape in shape_list:
            last_dim = x_shape[-1]
            x1 = torch.randn(x_shape, dtype=torch.float16, device='npu')
            x2 = torch.randn(x_shape, dtype=torch.float16, device='npu')
            gamma = torch.ones(last_dim, dtype=torch.float16, device='npu')
            beta = torch.zeros(last_dim, dtype=torch.float16, device='npu')
            smooth_scale1 = torch.ones(last_dim, dtype=torch.float16, device='npu')
            smooth_scale2 = torch.ones(last_dim, dtype=torch.float16, device='npu')

            self._run_and_check(x1, x2, gamma, smooth_scale1, smooth_scale2, beta)


    @SupportedDevices(['Ascend910B'])
    def test_forward_various_shapes_bf16(self):
        shape_list = [
            [2, 8],
            [3, 16],
            [4, 5, 32],
            [2, 3, 4, 24],
            [2, 3, 4, 5, 64],
            [2, 3, 4, 5, 6, 128],
            [2, 3, 4, 5, 6, 7, 256],
            [2, 3, 4, 5, 6, 7, 8, 512],
        ]
        for x_shape in shape_list:
            last_dim = x_shape[-1]
            x1 = torch.randn(x_shape, dtype=torch.bfloat16, device='npu')
            x2 = torch.randn(x_shape, dtype=torch.bfloat16, device='npu')
            gamma = torch.ones(last_dim, dtype=torch.bfloat16, device='npu')
            beta = torch.zeros(last_dim, dtype=torch.bfloat16, device='npu')
            smooth_scale1 = torch.ones(last_dim, dtype=torch.bfloat16, device='npu')
            smooth_scale2 = torch.ones(last_dim, dtype=torch.bfloat16, device='npu')

            self._run_and_check(x1, x2, gamma, smooth_scale1, smooth_scale2, beta)


    @SupportedDevices(['Ascend910B'])
    def test_forward_output_mask_true_false_fp16(self):
        x_shape = [2, 3, 32]
        last_dim = x_shape[-1]

        x1 = torch.randn(x_shape, dtype=torch.float16, device='npu')
        x2 = torch.randn(x_shape, dtype=torch.float16, device='npu')
        gamma = torch.ones(last_dim, dtype=torch.float16, device='npu')
        beta = torch.zeros(last_dim, dtype=torch.float16, device='npu')
        smooth_scale1 = torch.ones(last_dim, dtype=torch.float16, device='npu')
        smooth_scale2 = None

        self._run_and_check(
            x1, x2, gamma,
            smooth_scale1=smooth_scale1,
            smooth_scale2=smooth_scale2,
            beta=beta,
            output_mask=[True, False],
        )
if __name__ == "__main__":
    run_tests()
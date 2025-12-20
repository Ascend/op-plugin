import unittest
import math
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


def npu_attention_update_golden(lse_list, local_out_list, update_type=0):
    if not isinstance(lse_list, (list, tuple)):
        raise TypeError("lse_list must be a list or tuple.")
    if not isinstance(local_out_list, (list, tuple)):
        raise TypeError("local_out_list must be a list or tuple.")
    if len(lse_list) == 0:
        raise ValueError("lse_list must not be empty.")
    if len(local_out_list) == 0:
        raise ValueError("local_out_list must not be empty.")
    if len(lse_list) != len(local_out_list):
        raise ValueError("lse_list and local_out_list must have the same length.")
    if update_type != 0 and update_type != 1:
        raise ValueError("update_type must be 0 or 1.")

    N = None
    H = None
    lse_cpu = []
    out_cpu = []
    out_dtype = torch.float32

    for i, (lse_i, out_i) in enumerate(zip(lse_list, local_out_list)):
        if not isinstance(lse_i, torch.Tensor):
            raise TypeError(f"lse[{i}] must be a torch.Tensor, got {type(lse_i)}")
        if not isinstance(out_i, torch.Tensor):
            raise TypeError(f"local_out[{i}] must be a torch.Tensor, got {type(out_i)}")
        if lse_i.dtype != torch.float32:
            raise ValueError(f"lse[{i}] must be float32, got {lse_i.dtype}")
        if out_i.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"local_out[{i}] must be float32, float16 or bfloat16, got {out_i.dtype}")
        if lse_i.dim() != 1:
            raise ValueError(f"lse[{i}] must be 1D [N], got shape {list(lse_i.shape)}")
        if out_i.dim() != 2:
            raise ValueError(f"local_out[{i}] must be 2D [N, H], got shape {list(out_i.shape)}")

        if N is None:
            N = lse_i.size(0)
            H = out_i.size(1)
            out_dtype = out_i.dtype
        else:
            if lse_i.size(0) != N:
                raise ValueError(f"lse[{i}].size(0) must be {N}, got {lse_i.size(0)}")
            if out_i.size(0) != N or out_i.size(1) != H:
                raise ValueError(f"local_out[{i}] must be [{N}, {H}], got {list(out_i.shape)}")

        lse_cpu.append(lse_i.detach().to("cpu", dtype=torch.float32))
        out_cpu.append(out_i.detach().to("cpu", dtype=torch.float32))

    lse_stack = torch.stack(lse_cpu, dim=0)
    out_stack = torch.stack(out_cpu, dim=0)

    lse_max, _ = torch.max(lse_stack, dim=0)
    exp_terms = torch.exp(lse_stack - lse_max.unsqueeze(0))
    lse_sum = torch.sum(exp_terms, dim=0)
    lse_m = lse_max + torch.log(lse_sum + 1e-20)

    weights = torch.exp(lse_stack - lse_m.unsqueeze(0))
    O = torch.sum(out_stack * weights.unsqueeze(-1), dim=0)
    return O.contiguous().to(out_dtype), lse_m


class TestNpuAttentionUpdate(TestCase):
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

    def _run_and_check(self, N=4, H=32, K=2, update_type=0, dtype=torch.float32, atol=1e-4, rtol=1e-4, benchmark=2**-10):
        if not isinstance(N, int):
            raise TypeError(f"N must be an int, got {type(N)}")
        if not isinstance(H, int):
            raise TypeError(f"H must be an int, got {type(H)}")
        if not isinstance(K, int):
            raise TypeError(f"K must be an int, got {type(K)}")
        if N <= 0:
            raise ValueError(f"N must be > 0, got {N}")
        if H <= 0:
            raise ValueError(f"H must be > 0, got {H}")
        if K <= 0:
            raise ValueError(f"K must be > 0, got {K}")
        if H % 8 != 0:
            raise ValueError(f"H must be divisible by 8, got {H}")
        if H > 512:
            raise ValueError(f"H must be less than 512, got {H}")

        device = 'npu'

        lse_list = [torch.randn(N, dtype=torch.float32, device=device) for _ in range(K)]
        local_out_list = [torch.randn(N, H, dtype=dtype, device=device) for _ in range(K)]

        out_npu, lse_out_npu = torch_npu.npu_attention_update(lse_list, local_out_list, update_type)

        out_cpu, lse_out_cpu = npu_attention_update_golden([t.to("cpu") for t in lse_list],
                                              [t.to("cpu") for t in local_out_list],
                                              update_type=update_type)

        self.assertEqual(tuple(out_npu.shape), (N, H))
        self.assertEqual(tuple(out_cpu.shape), (N, H))
        self.assertEqual(out_npu.dtype, dtype)
        self.assertEqual(out_cpu.dtype, dtype)

        out_npu_cpu = out_npu.to("cpu")
        self.assertTrue(torch.allclose(out_npu_cpu, out_cpu, atol=atol, rtol=rtol),
                        f"allclose failed, max_abs={float((out_npu_cpu - out_cpu).abs().max())}")

        for t in [out_npu_cpu, out_cpu]:
            self.assertFalse(torch.isnan(t).any().item())
            self.assertFalse(torch.isinf(t).any().item())

    @unittest.skip("skip until CANN is updated to support aclnnAttentionUpdate")
    @SupportedDevices(['Ascend910B'])
    def test_forward_min_case(self):
        cases = [
            (4, 32, 2),
            (8, 64, 2),
            (16, 128, 3),
            (32, 256, 2),
            (64, 64, 4),
            (120, 32, 2),
            (240, 128, 2),
        ]
        dtype_configs = [
            (torch.float32, 1e-4, 1e-4),
            (torch.float16, 1e-3, 1e-3),
            (torch.bfloat16, 4e-3, 4e-3),
        ]
        for (N, H, K) in cases:
            for dtype, atol, rtol in dtype_configs:
                self._run_and_check(N=N, H=H, K=K, update_type=0, dtype=dtype, atol=atol, rtol=rtol)

if __name__ == "__main__":
    run_tests()
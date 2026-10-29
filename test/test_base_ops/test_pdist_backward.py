import unittest

import numpy as np
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


def torch_version_ge(major, minor):
    parts = torch.__version__.split("+")[0].split(".")[:2]
    nums = []
    for part in parts:
        digits = ""
        for ch in part:
            if not ch.isdigit():
                break
            digits += ch
        nums.append(int(digits) if digits else 0)
    return tuple(nums) >= (major, minor)


@unittest.skipIf(not torch_version_ge(2, 14), "_pdist_backward NPU kernel requires PyTorch 2.14+")
class TestPdistBackward(TestCase):
    def cpu_first_order_grad(self, x, v, p):
        stype = x.dtype
        if stype == torch.float16:
            x = x.float()
            v = v.float()
        x = x.clone().requires_grad_(True)
        output = torch.nn.functional.pdist(x, p)
        grad = torch.autograd.grad(output, x, grad_outputs=v)[0]
        if stype == torch.float16:
            grad = grad.half()
        return grad.detach().numpy()

    def npu_first_order_grad(self, x, v, p):
        x = x.clone().requires_grad_(True)
        output = torch.nn.functional.pdist(x, p)
        grad = torch.autograd.grad(output, x, grad_outputs=v)[0]
        return grad.detach().cpu().numpy()

    def cpu_second_order_grads(self, x, v, p):
        x = x.clone().requires_grad_(True)
        v = v.clone().requires_grad_(True)
        output = torch.nn.functional.pdist(x, p)
        grad = torch.autograd.grad(output, x, grad_outputs=v, create_graph=True)[0]
        return [g.detach().numpy() for g in torch.autograd.grad(grad.sum(), (x, v))]

    def npu_second_order_grads(self, x, v, p):
        x = x.clone().requires_grad_(True)
        v = v.clone().requires_grad_(True)
        output = torch.nn.functional.pdist(x, p)
        grad = torch.autograd.grad(output, x, grad_outputs=v, create_graph=True)[0]
        return [g.detach().cpu().numpy() for g in torch.autograd.grad(grad.sum(), (x, v))]

    def run_first_order(self, item, p):
        cpu_input, npu_input = create_common_tensor(item, -2, 2)
        n = item[2][0]
        v = torch.from_numpy(np.random.uniform(-2, 2, (n * (n - 1) // 2,)).astype(np.float32))
        cpu_grad = self.cpu_first_order_grad(cpu_input, v, p)
        v_npu = v.to("npu")
        if npu_input.dtype == torch.float16:
            v_npu = v_npu.half()
        npu_grad = self.npu_first_order_grad(npu_input, v_npu, p)
        self.assertRtolEqual(cpu_grad, npu_grad)

    def run_second_order(self, cpu_input, npu_input, v, p):
        cpu_grads = self.cpu_second_order_grads(cpu_input, v, p)
        npu_grads = self.npu_second_order_grads(npu_input, v.to("npu"), p)
        for name, cpu_grad, npu_grad in zip(("self", "v"), cpu_grads, npu_grads):
            with self.subTest(grad=name):
                self.assertRtolEqual(cpu_grad, npu_grad)

    def test_pdist_backward_common_shape_format(self):
        # fp16 covers the sign-only gradient (p == 1.0); fp32 covers
        # representative finite p values.
        shape_format = [
            [[np.float16, 2, (6, 8)], 1.0],
            [[np.float32, 2, (6, 8)], 0.0],
            [[np.float32, 2, (6, 8)], 0.5],
            [[np.float32, 2, (6, 8)], 1.0],
            [[np.float32, 2, (6, 8)], 1.5],
            [[np.float32, 2, (6, 8)], 2.0],
            [[np.float32, 2, (6, 8)], 3.0],
        ]
        for item in shape_format:
            with self.subTest(shape=item[0][2], dtype=item[0][0], p=item[1]):
                np.random.seed(10086)
                self.run_first_order(item[0], item[1])

    def test_pdist_backward_edge_shape(self):
        # No pair to accumulate (n <= 1), empty input and empty feature dim.
        # The gradient w.r.t. self is mathematically zero for every shape
        # below, so compare the NPU result against zeros directly.
        shape_format = [
            [np.float32, 2, (1, 5)],
            [np.float32, 2, (0, 3)],
            [np.float32, 2, (4, 0)],
            [np.float32, 2, (0, 0)],
        ]
        for item in shape_format:
            with self.subTest(shape=item[2]):
                np.random.seed(10086)
                _, npu_input = create_common_tensor(item, -2, 2)
                n = item[2][0]
                v = torch.from_numpy(np.random.uniform(-2, 2, (n * (n - 1) // 2,)).astype(np.float32))
                v_npu = v.to("npu")
                if npu_input.dtype == torch.float16:
                    v_npu = v_npu.half()
                npu_grad = self.npu_first_order_grad(npu_input, v_npu, 2.0)
                self.assertEqual(npu_grad, np.zeros_like(npu_grad))

    def test_pdist_backward_duplicate_rows(self):
        # Duplicate rows produce zero distances and exercise the masking
        # convention shared by the CPU kernel and the NPU implementation.
        for p in [0.5, 1.0, 1.5, 2.0]:
            with self.subTest(p=p):
                np.random.seed(10086)
                cpu_input = torch.from_numpy(np.random.uniform(-2, 2, (7, 4)).astype(np.float32))
                cpu_input[3] = cpu_input[0]
                cpu_input[5] = cpu_input[2]
                npu_input = cpu_input.to("npu")
                v = torch.from_numpy(np.random.uniform(-2, 2, (7 * 6 // 2,)).astype(np.float32))
                cpu_grad = self.cpu_first_order_grad(cpu_input, v, p)
                npu_grad = self.npu_first_order_grad(npu_input, v.to("npu"), p)
                self.assertRtolEqual(cpu_grad, npu_grad)

    def test_pdist_gradgrad_common_shape_format(self):
        shape_format = [
            [[np.float32, 2, (6, 8)], 1.0],
            [[np.float32, 2, (6, 8)], 1.5],
            [[np.float32, 2, (6, 8)], 2.0],
            [[np.float32, 2, (6, 8)], 3.0],
        ]
        for item in shape_format:
            with self.subTest(shape=item[0][2], p=item[1]):
                np.random.seed(10086)
                cpu_input, npu_input = create_common_tensor(item[0], -2, 2)
                n = item[0][2][0]
                v = torch.from_numpy(np.random.uniform(-2, 2, (n * (n - 1) // 2,)).astype(np.float32))
                self.run_second_order(cpu_input, npu_input, v, item[1])

    def test_pdist_gradgrad_inf(self):
        # Covers the p == inf case in a dedicated test so it is reported
        # separately from the regular p values.
        shape_format = [
            [[np.float32, 2, (5, 3)], np.inf],
        ]
        for item in shape_format:
            with self.subTest(shape=item[0][2], p=item[1]):
                np.random.seed(10086)
                cpu_input, npu_input = create_common_tensor(item[0], -2, 2)
                n = item[0][2][0]
                v = torch.from_numpy(np.random.uniform(-2, 2, (n * (n - 1) // 2,)).astype(np.float32))
                self.run_second_order(cpu_input, npu_input, v, item[1])

    def test_pdist_gradgrad_small_p(self):
        # For p < 1 the second-order terms contain |diff|^(p-2), which is too
        # singular for plain random data; separate the rows so that no
        # pairwise difference is close to zero.
        np.random.seed(10086)
        n = 5
        cpu_input = torch.from_numpy(np.random.uniform(-2, 2, (n, 3)).astype(np.float32))
        sep = torch.arange(n, dtype=torch.float32).view(n, 1) * 50
        cpu_input = cpu_input + sep
        npu_input = cpu_input.to("npu")
        v = torch.from_numpy(np.random.uniform(-2, 2, (n * (n - 1) // 2,)).astype(np.float32))
        for p in [0.5, 0.75]:
            with self.subTest(p=p):
                self.run_second_order(cpu_input, npu_input, v, p)

    def test_pdist_gradgrad_duplicate_rows(self):
        for p in [1.5, 2.0]:
            with self.subTest(p=p):
                np.random.seed(10086)
                cpu_input = torch.from_numpy(np.random.uniform(-2, 2, (6, 4)).astype(np.float32))
                cpu_input[2] = cpu_input[0]
                npu_input = cpu_input.to("npu")
                v = torch.from_numpy(np.random.uniform(-2, 2, (6 * 5 // 2,)).astype(np.float32))
                self.run_second_order(cpu_input, npu_input, v, p)


if __name__ == "__main__":
    run_tests()

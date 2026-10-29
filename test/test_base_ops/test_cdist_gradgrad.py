import unittest

import numpy as np
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


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


class TestCdistGradGrad(TestCase):
    def generate_data(self, min_n, max_n, shape_x, shape_y, src_type):
        np.random.seed(10086)
        x1 = np.random.uniform(min_n, max_n, shape_x).astype(src_type)
        x2 = np.random.uniform(min_n, max_n, shape_y).astype(src_type)
        return x1, x2

    def second_order_grads(self, device, x1, x2, v, p, wrt="x1"):
        x1 = torch.tensor(x1, device=device, requires_grad=True)
        x2 = torch.tensor(x2, device=device, requires_grad=True)
        v = torch.tensor(v, device=device, requires_grad=True)
        output = torch.cdist(x1, x2, p)
        target = x1 if wrt == "x1" else x2
        grad = torch.autograd.grad(output, target, grad_outputs=v, create_graph=True)[0]
        return [g.detach().cpu().numpy() for g in torch.autograd.grad(grad.sum(), (x1, x2, v))]

    @unittest.skipIf(not torch_version_ge(2, 14), "cdist double backward requires PyTorch 2.14+")
    def test_cdist_gradgrad_common_shape(self):
        # Covers 2D and batch-matched 3D inputs.
        shape_items = [
            [np.float32, (5, 10), (4, 10), (5, 4)],
            [np.float32, (20, 5, 10), (20, 4, 10), (20, 5, 4)],
        ]
        p_ranges = [1.0, 1.5, 2.0, 2.5, 3.0]
        for item in shape_items:
            for p in p_ranges:
                with self.subTest(shape=item[1], p=p):
                    input1, input2 = self.generate_data(-1, 1, item[1], item[2], item[0])
                    v = np.random.uniform(-1, 1, item[3]).astype(item[0])
                    cpu_grads = self.second_order_grads('cpu', input1, input2, v, p)
                    npu_grads = self.second_order_grads('npu', input1, input2, v, p)
                    for name, cpu_grad, npu_grad in zip(("x1", "x2", "v"), cpu_grads, npu_grads):
                        with self.subTest(grad=name):
                            self.assertRtolEqual(cpu_grad, npu_grad)

    @unittest.skipIf(not torch_version_ge(2, 14), "cdist double backward requires PyTorch 2.14+")
    def test_cdist_gradgrad_inf(self):
        # Covers the p == inf case in a dedicated test so it is reported
        # separately from the regular p values.
        shape_items = [
            [np.float32, (5, 10), (4, 10), (5, 4)],
            [np.float32, (20, 5, 10), (20, 4, 10), (20, 5, 4)],
        ]
        p_ranges = [np.inf]
        for item in shape_items:
            for p in p_ranges:
                with self.subTest(shape=item[1], p=p):
                    input1, input2 = self.generate_data(-1, 1, item[1], item[2], item[0])
                    v = np.random.uniform(-1, 1, item[3]).astype(item[0])
                    cpu_grads = self.second_order_grads('cpu', input1, input2, v, p)
                    npu_grads = self.second_order_grads('npu', input1, input2, v, p)
                    for name, cpu_grad, npu_grad in zip(("x1", "x2", "v"), cpu_grads, npu_grads):
                        with self.subTest(grad=name):
                            self.assertRtolEqual(cpu_grad, npu_grad)

    @unittest.skipIf(not torch_version_ge(2, 14), "cdist double backward requires PyTorch 2.14+")
    def test_cdist_gradgrad_small_p(self):
        # For p < 1 the second-order terms contain |diff|^(p-2), which is too
        # singular for plain random data; keep the point clouds apart so that
        # no pairwise difference is close to zero.
        for p in [0.5, 0.75]:
            with self.subTest(p=p):
                np.random.seed(10086)
                input1 = np.random.uniform(-1, 1, (4, 3)).astype(np.float32)
                input2 = (np.random.uniform(0, 1, (5, 3)) + 20).astype(np.float32)
                v = np.random.uniform(-1, 1, (4, 5)).astype(np.float32)
                cpu_grads = self.second_order_grads('cpu', input1, input2, v, p)
                npu_grads = self.second_order_grads('npu', input1, input2, v, p)
                for name, cpu_grad, npu_grad in zip(("x1", "x2", "v"), cpu_grads, npu_grads):
                    with self.subTest(grad=name):
                        self.assertRtolEqual(cpu_grad, npu_grad)

    @unittest.skipIf(not torch_version_ge(2, 14), "cdist double backward requires PyTorch 2.14+")
    def test_cdist_gradgrad_duplicate_points(self):
        # Duplicate rows produce zero distances and exercise the masking
        # convention of the second-order formulas.
        for p in [1.5, 2.0]:
            with self.subTest(p=p):
                np.random.seed(10086)
                input1 = np.random.uniform(-1, 1, (6, 4)).astype(np.float32)
                input1[2] = input1[0]
                input1[4] = input1[1]
                input2 = np.random.uniform(-1, 1, (5, 4)).astype(np.float32)
                input2[3] = input2[0]
                v = np.random.uniform(-1, 1, (6, 5)).astype(np.float32)
                cpu_grads = self.second_order_grads('cpu', input1, input2, v, p)
                npu_grads = self.second_order_grads('npu', input1, input2, v, p)
                for name, cpu_grad, npu_grad in zip(("x1", "x2", "v"), cpu_grads, npu_grads):
                    with self.subTest(grad=name):
                        self.assertRtolEqual(cpu_grad, npu_grad)

    @unittest.skipIf(not torch_version_ge(2, 14), "cdist double backward requires PyTorch 2.14+")
    def test_cdist_gradgrad_wrt_x2(self):
        # Differentiate the x2 slot first to exercise the
        # grad.mT().contiguous() path of the backward formula under
        # create_graph=True.
        for p in [1.5, 2.0]:
            with self.subTest(p=p):
                input1, input2 = self.generate_data(-1, 1, (5, 10), (4, 10), np.float32)
                np.random.seed(20086)
                v = np.random.uniform(-1, 1, (5, 4)).astype(np.float32)
                cpu_grads = self.second_order_grads('cpu', input1, input2, v, p, wrt="x2")
                npu_grads = self.second_order_grads('npu', input1, input2, v, p, wrt="x2")
                for name, cpu_grad, npu_grad in zip(("x1", "x2", "v"), cpu_grads, npu_grads):
                    with self.subTest(grad=name):
                        self.assertRtolEqual(cpu_grad, npu_grad)


if __name__ == "__main__":
    run_tests()

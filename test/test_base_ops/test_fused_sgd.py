import unittest
import copy
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

@unittest.skip("skip now")
def fused_sgd_ref(params, grads, momentum_buffer_list, weight_decay, momentum, lr,
                  dampening, nesterov, maximize, is_first_step,
                  grad_scale=None, found_inf=None):
    if found_inf is not None and int(found_inf.item()) == 1:
        return
    grad_scale_val = grad_scale.item() if grad_scale is not None else 1.0
    for i in range(len(params)):
        d_p = grads[i].clone()
        if grad_scale is not None:
            d_p = d_p / grad_scale_val
        if weight_decay != 0:
            d_p = d_p.add(params[i], alpha=weight_decay)
        if momentum != 0:
            buf = momentum_buffer_list[i]
            if is_first_step:
                buf.copy_(d_p)
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf
        if maximize:
            params[i].add_(d_p, alpha=lr)
        else:
            params[i].add_(d_p, alpha=-lr)

@unittest.skip("skip now")
class TestFusedSgd(TestCase):

    def _gen_tensors(self, shapes, dtype=torch.float32, seed=42):
        torch.manual_seed(seed)
        cpu_params = [torch.randn(s, dtype=dtype) for s in shapes]
        cpu_grads = [torch.randn(s, dtype=dtype) for s in shapes]
        cpu_bufs = [torch.randn(s, dtype=dtype) for s in shapes]
        npu_params = [p.npu() for p in cpu_params]
        npu_grads = [g.npu() for g in cpu_grads]
        npu_bufs = [b.npu() for b in cpu_bufs]
        cpu_params = [p.clone() for p in cpu_params]
        cpu_grads = [g.clone() for g in cpu_grads]
        cpu_bufs = [b.clone() for b in cpu_bufs]
        return cpu_params, cpu_grads, cpu_bufs, npu_params, npu_grads, npu_bufs

    def _run_and_compare(self, shapes, dtype, weight_decay, momentum, lr,
                         dampening, nesterov, maximize, is_first_step,
                         grad_scale=None, found_inf=None, prec=None):
        cpu_p, cpu_g, cpu_b, npu_p, npu_g, npu_b = self._gen_tensors(shapes, dtype)

        torch._fused_sgd_(cpu_p, cpu_g, cpu_b, weight_decay=weight_decay,
                          momentum=momentum, lr=lr, dampening=dampening,
                          nesterov=nesterov, maximize=maximize,
                          is_first_step=is_first_step,
                          grad_scale=grad_scale, found_inf=found_inf)

        npu_grad_scale = grad_scale.npu() if grad_scale is not None else None
        npu_found_inf = found_inf.npu() if found_inf is not None else None
        torch._fused_sgd_(npu_p, npu_g, npu_b, weight_decay=weight_decay,
                          momentum=momentum, lr=lr, dampening=dampening,
                          nesterov=nesterov, maximize=maximize,
                          is_first_step=is_first_step,
                          grad_scale=npu_grad_scale, found_inf=npu_found_inf)

        for c_p, n_p, c_b, n_b in zip(cpu_p, npu_p, cpu_b, npu_b):
            self.assertRtolEqual(c_p, n_p.cpu(), prec=1e-3)
            self.assertRtolEqual(c_b, n_b.cpu(), prec=1e-3)

    def test_fused_sgd_first_step(self):
        self._run_and_compare(
            shapes=[(4, 4), (8, 8)], dtype=torch.float32,
            weight_decay=0.0, momentum=0.9, lr=0.01,
            dampening=0.0, nesterov=False, maximize=False,
            is_first_step=True)

    def test_fused_sgd_subsequent_step(self):
        self._run_and_compare(
            shapes=[(4, 4), (8, 8)], dtype=torch.float32,
            weight_decay=0.0, momentum=0.9, lr=0.01,
            dampening=0.0, nesterov=False, maximize=False,
            is_first_step=False)

    def test_fused_sgd_weight_decay(self):
        self._run_and_compare(
            shapes=[(4, 4), (2, 3, 3)], dtype=torch.float32,
            weight_decay=0.01, momentum=0.9, lr=0.01,
            dampening=0.0, nesterov=False, maximize=False,
            is_first_step=False)

    def test_fused_sgd_nesterov(self):
        self._run_and_compare(
            shapes=[(4, 4), (8, 8)], dtype=torch.float32,
            weight_decay=0.0, momentum=0.9, lr=0.01,
            dampening=0.0, nesterov=True, maximize=False,
            is_first_step=False)

    def test_fused_sgd_maximize(self):
        self._run_and_compare(
            shapes=[(4, 4), (8, 8)], dtype=torch.float32,
            weight_decay=0.0, momentum=0.9, lr=0.01,
            dampening=0.0, nesterov=False, maximize=True,
            is_first_step=False)

    def test_fused_sgd_dampening(self):
        self._run_and_compare(
            shapes=[(4, 4), (8, 8)], dtype=torch.float32,
            weight_decay=0.0, momentum=0.9, lr=0.01,
            dampening=0.5, nesterov=False, maximize=False,
            is_first_step=False)

    def test_fused_sgd_nesterov_with_weight_decay(self):
        self._run_and_compare(
            shapes=[(4, 4), (8, 8)], dtype=torch.float32,
            weight_decay=0.001, momentum=0.9, lr=0.01,
            dampening=0.0, nesterov=True, maximize=True,
            is_first_step=False)

    def test_fused_sgd_found_inf_skip(self):
        cpu_p, cpu_g, cpu_b, npu_p, npu_g, npu_b = self._gen_tensors(
            [(4, 4), (8, 8)], torch.float32)
        cpu_p_before = [p.clone() for p in cpu_p]
        cpu_b_before = [b.clone() for b in cpu_b]
        npu_p_before = [p.clone() for p in npu_p]
        npu_b_before = [b.clone() for b in npu_b]

        found_inf = torch.tensor(1, dtype=torch.int32)
        fused_sgd_ref(cpu_p, cpu_g, cpu_b, weight_decay=0.0, momentum=0.9, lr=0.01, dampening= 0.0,
                      nesterov=False, maximize=False, is_first_step=False, grad_scale=None, found_inf=found_inf)
        torch._fused_sgd_(npu_p, npu_g, npu_b, weight_decay=0.0,
                          momentum=0.9, lr=0.01, dampening=0.0,
                          nesterov=False, maximize=False,
                          is_first_step=False, grad_scale=None,
                          found_inf=found_inf.npu())

        for c_before, c_after in zip(cpu_p_before, cpu_p):
            self.assertRtolEqual(c_before, c_after, prec=1e-2)
        for n_before, n_after in zip(npu_p_before, npu_p):
            self.assertRtolEqual(n_before.cpu(), n_after.cpu(), prec=1e-2)

    def test_fused_sgd_found_inf_zero(self):
        found_inf = torch.tensor(0, dtype=torch.float32)
        self._run_and_compare(
            shapes=[(4, 4), (8, 8)], dtype=torch.float32,
            weight_decay=0.0, momentum=0.9, lr=0.01,
            dampening=0.0, nesterov=False, maximize=False,
            is_first_step=False, found_inf=found_inf)

    def test_fused_sgd_grad_scale(self):
        grad_scale = torch.tensor(4.0, dtype=torch.float32)
        self._run_and_compare(
            shapes=[(4, 4), (8, 8)], dtype=torch.float32,
            weight_decay=0.0, momentum=0.9, lr=0.01,
            dampening=0.0, nesterov=False, maximize=False,
            is_first_step=False, grad_scale=grad_scale)

    def test_fused_sgd_grad_scale_with_weight_decay(self):
        grad_scale = torch.tensor(2.0, dtype=torch.float32)
        self._run_and_compare(
            shapes=[(4, 4), (8, 8)], dtype=torch.float32,
            weight_decay=0.01, momentum=0.9, lr=0.01,
            dampening=0.0, nesterov=True, maximize=False,
            is_first_step=False, grad_scale=grad_scale)

    def test_fused_sgd_single_tensor(self):
        self._run_and_compare(
            shapes=[(16, 16)], dtype=torch.float32,
            weight_decay=0.01, momentum=0.9, lr=0.05,
            dampening=0.1, nesterov=False, maximize=False,
            is_first_step=False)

    def test_fused_sgd_multi_tensor(self):
        self._run_and_compare(
            shapes=[(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)], dtype=torch.float32,
            weight_decay=0.001, momentum=0.9, lr=0.01,
            dampening=0.0, nesterov=False, maximize=False,
            is_first_step=False)

    def test_fused_sgd_momentum_non_positive_error(self):
        npu_p = [torch.randn(4, 4, dtype=torch.float32).npu()]
        npu_g = [torch.randn(4, 4, dtype=torch.float32).npu()]
        npu_b = [torch.randn(4, 4, dtype=torch.float32).npu()]
        with self.assertRaises(RuntimeError):
            torch._fused_sgd_(npu_p, npu_g, npu_b, weight_decay=0.0,
                              momentum=0.0, lr=0.01, dampening=0.0,
                              nesterov=False, maximize=False,
                              is_first_step=True)

@unittest.skip("skip now")
class TestFusedSgdTensorLr(TestCase):

    def _gen_tensors(self, shapes, dtype=torch.float32, seed=42):
        torch.manual_seed(seed)
        cpu_params = [torch.randn(s, dtype=dtype) for s in shapes]
        cpu_grads = [torch.randn(s, dtype=dtype) for s in shapes]
        cpu_bufs = [torch.randn(s, dtype=dtype) for s in shapes]
        npu_params = [p.npu() for p in cpu_params]
        npu_grads = [g.npu() for g in cpu_grads]
        npu_bufs = [b.npu() for b in cpu_bufs]
        cpu_params = [p.clone() for p in cpu_params]
        cpu_grads = [g.clone() for g in cpu_grads]
        cpu_bufs = [b.clone() for b in cpu_bufs]
        return cpu_params, cpu_grads, cpu_bufs, npu_params, npu_grads, npu_bufs

    def _run_and_compare(self, shapes, dtype, weight_decay, momentum, lr_val,
                         dampening, nesterov, maximize, is_first_step,
                         grad_scale=None, found_inf=None, prec=None):
        cpu_p, cpu_g, cpu_b, npu_p, npu_g, npu_b = self._gen_tensors(shapes, dtype)

        cpu_lr = torch.tensor(lr_val, dtype=torch.float32)
        torch._fused_sgd_(cpu_p, cpu_g, cpu_b, weight_decay=weight_decay, momentum=momentum, lr=cpu_lr,
                          dampening=dampening, nesterov=nesterov, maximize=maximize, is_first_step=is_first_step,
                          grad_scale=grad_scale, found_inf=found_inf)

        npu_lr = cpu_lr.npu()
        npu_grad_scale = grad_scale.npu() if grad_scale is not None else None
        npu_found_inf = found_inf.npu() if found_inf is not None else None
        torch._fused_sgd_(npu_p, npu_g, npu_b, weight_decay=weight_decay,
                                    momentum=momentum, lr=npu_lr,
                                    dampening=dampening, nesterov=nesterov,
                                    maximize=maximize, is_first_step=is_first_step,
                                    grad_scale=npu_grad_scale,
                                    found_inf=npu_found_inf)

        for c_p, n_p, c_b, n_b in zip(cpu_p, npu_p, cpu_b, npu_b):
            self.assertRtolEqual(c_p, n_p.cpu(), prec=1e-3)
            self.assertRtolEqual(c_b, n_b.cpu(), prec=1e-3)

    def test_fused_sgd_tensor_lr_first_step(self):
        self._run_and_compare(
            shapes=[(4, 4), (8, 8)], dtype=torch.float32,
            weight_decay=0.0, momentum=0.9, lr_val=0.01,
            dampening=0.0, nesterov=False, maximize=False,
            is_first_step=True)

    def test_fused_sgd_tensor_lr_subsequent_step(self):
        self._run_and_compare(
            shapes=[(4, 4), (8, 8)], dtype=torch.float32,
            weight_decay=0.01, momentum=0.9, lr_val=0.01,
            dampening=0.0, nesterov=False, maximize=False,
            is_first_step=False)

    def test_fused_sgd_tensor_lr_nesterov(self):
        self._run_and_compare(
            shapes=[(4, 4), (8, 8)], dtype=torch.float32,
            weight_decay=0.001, momentum=0.9, lr_val=0.05,
            dampening=0.0, nesterov=True, maximize=False,
            is_first_step=False)

    def test_fused_sgd_tensor_lr_maximize(self):
        self._run_and_compare(
            shapes=[(4, 4), (8, 8)], dtype=torch.float32,
            weight_decay=0.0, momentum=0.9, lr_val=0.01,
            dampening=0.0, nesterov=False, maximize=True,
            is_first_step=False)

    def test_fused_sgd_tensor_lr_found_inf_skip(self):
        cpu_p, cpu_g, cpu_b, npu_p, npu_g, npu_b = self._gen_tensors(
            [(4, 4), (8, 8)], torch.float32)
        cpu_p_before = [p.clone() for p in cpu_p]
        npu_p_before = [p.clone() for p in npu_p]

        found_inf = torch.tensor(1, dtype=torch.float32)
        torch._fused_sgd_(cpu_p, cpu_g, cpu_b, weight_decay=0.0, momentum=0.9, lr=0.01, dampening=0.0,
                      nesterov=False, maximize=False, is_first_step=False, grad_scale=None, found_inf=found_inf)
        npu_lr = torch.tensor(0.01, dtype=torch.float32).npu()
        torch._fused_sgd_(npu_p, npu_g, npu_b, weight_decay=0.0,
                                    momentum=0.9, lr=npu_lr, dampening=0.0,
                                    nesterov=False, maximize=False,
                                    is_first_step=False, grad_scale=None,
                                    found_inf=found_inf.npu())

        for c_before, c_after in zip(cpu_p_before, cpu_p):
            self.assertRtolEqual(c_before, c_after)
        for n_before, n_after in zip(npu_p_before, npu_p):
            self.assertRtolEqual(n_before.cpu(), n_after.cpu())

    def test_fused_sgd_tensor_lr_grad_scale(self):
        grad_scale = torch.tensor([4.0], dtype=torch.float32)
        self._run_and_compare(
            shapes=[(4, 4), (8, 8)], dtype=torch.float32,
            weight_decay=0.0, momentum=0.9, lr_val=0.01,
            dampening=0.0, nesterov=False, maximize=False,
            is_first_step=False, grad_scale=grad_scale)

    def test_fused_sgd_tensor_lr_momentum_non_positive_error(self):
        npu_p = [torch.randn(4, 4, dtype=torch.float32).npu()]
        npu_g = [torch.randn(4, 4, dtype=torch.float32).npu()]
        npu_b = [torch.randn(4, 4, dtype=torch.float32).npu()]
        npu_lr = torch.tensor(0.01, dtype=torch.float32).npu()
        with self.assertRaises(RuntimeError):
            torch._fused_sgd_(npu_p, npu_g, npu_b, weight_decay=0.0,
                                        momentum=0.0, lr=npu_lr,
                                        dampening=0.0, nesterov=False,
                                        maximize=False, is_first_step=True)

    def test_fused_sgd_tensor_lr_device_check(self):
        npu_p = [torch.randn(4, 4, dtype=torch.float32).npu()]
        npu_g = [torch.randn(4, 4, dtype=torch.float32).npu()]
        npu_b = [torch.randn(4, 4, dtype=torch.float32).npu()]
        cpu_lr = torch.tensor(0.01, dtype=torch.float32)
        with self.assertRaises(RuntimeError):
            torch._fused_sgd_(npu_p, npu_g, npu_b, weight_decay=0.0,
                                        momentum=0.9, lr=cpu_lr,
                                        dampening=0.0, nesterov=False,
                                        maximize=False, is_first_step=True)


if __name__ == "__main__":
    run_tests()

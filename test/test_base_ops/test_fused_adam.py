# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestFusedAdam(TestCase):
    """UT for torch._fused_adam_ (native op, in-place TensorList update)."""

    def _gen_tensors(self, shapes, dtype=torch.float32, seed=0):
        """Generate CPU and NPU TensorLists with identical data."""
        torch.manual_seed(seed)
        cpu_self = [torch.randn(s, dtype=dtype) for s in shapes]
        cpu_grads = [torch.randn(s, dtype=dtype) for s in shapes]
        cpu_exp_avgs = [torch.zeros(s, dtype=dtype) for s in shapes]
        cpu_exp_avg_sqs = [torch.zeros(s, dtype=dtype) for s in shapes]
        cpu_max_exp_avg_sqs = [torch.zeros(s, dtype=dtype) for s in shapes]
        npu_self = [p.npu() for p in cpu_self]
        npu_grads = [g.npu() for g in cpu_grads]
        npu_exp_avgs = [m.npu() for m in cpu_exp_avgs]
        npu_exp_avg_sqs = [v.npu() for v in cpu_exp_avg_sqs]
        npu_max_exp_avg_sqs = [mv.npu() for mv in cpu_max_exp_avg_sqs]
        cpu_self = [p.clone() for p in cpu_self]
        cpu_grads = [g.clone() for g in cpu_grads]
        cpu_exp_avgs = [m.clone() for m in cpu_exp_avgs]
        cpu_exp_avg_sqs = [v.clone() for v in cpu_exp_avg_sqs]
        cpu_max_exp_avg_sqs = [mv.clone() for mv in cpu_max_exp_avg_sqs]
        return (cpu_self, cpu_grads, cpu_exp_avgs, cpu_exp_avg_sqs, cpu_max_exp_avg_sqs,
                npu_self, npu_grads, npu_exp_avgs, npu_exp_avg_sqs, npu_max_exp_avg_sqs)

    def _run_and_compare(self, shapes, dtype, lr, beta1, beta2, weight_decay,
                         eps, amsgrad, maximize, state_steps_values,
                         grad_scale=None, found_inf=None, prec=1e-3):
        """CPU reference vs NPU, assert all in-place updated tensors."""
        (cpu_self, cpu_grads, cpu_exp_avgs, cpu_exp_avg_sqs, cpu_max_exp_avg_sqs,
         npu_self, npu_grads, npu_exp_avgs, npu_exp_avg_sqs, npu_max_exp_avg_sqs) = \
            self._gen_tensors(shapes, dtype)

        state_steps_cpu = [torch.tensor([v], dtype=torch.int64) for v in state_steps_values]
        state_steps_npu = [s.npu() for s in state_steps_cpu]

        npu_grad_scale = grad_scale.npu() if grad_scale is not None else None
        npu_found_inf = found_inf.npu() if found_inf is not None else None

        max_exp_avg_sqs_cpu = cpu_max_exp_avg_sqs if amsgrad else []
        max_exp_avg_sqs_npu = npu_max_exp_avg_sqs if amsgrad else []

        torch._fused_adam_(
            cpu_self, cpu_grads, cpu_exp_avgs, cpu_exp_avg_sqs,
            max_exp_avg_sqs_cpu, state_steps_cpu,
            lr=lr, beta1=beta1, beta2=beta2, weight_decay=weight_decay,
            eps=eps, amsgrad=amsgrad, maximize=maximize,
            grad_scale=None, found_inf=None)

        torch._fused_adam_(
            npu_self, npu_grads, npu_exp_avgs, npu_exp_avg_sqs,
            max_exp_avg_sqs_npu, state_steps_npu,
            lr=lr, beta1=beta1, beta2=beta2, weight_decay=weight_decay,
            eps=eps, amsgrad=amsgrad, maximize=maximize,
            grad_scale=npu_grad_scale, found_inf=npu_found_inf)

        # Assertion Completeness Iron Rule: every updated tensor must be value-compared
        for c_p, n_p in zip(cpu_self, npu_self):
            self.assertRtolEqual(c_p, n_p.cpu(), prec=prec)
        for c_m, n_m in zip(cpu_exp_avgs, npu_exp_avgs):
            self.assertRtolEqual(c_m, n_m.cpu(), prec=prec)
        for c_v, n_v in zip(cpu_exp_avg_sqs, npu_exp_avg_sqs):
            self.assertRtolEqual(c_v, n_v.cpu(), prec=prec)
        if amsgrad:
            for c_mv, n_mv in zip(cpu_max_exp_avg_sqs, npu_max_exp_avg_sqs):
                self.assertRtolEqual(c_mv, n_mv.cpu(), prec=prec)
        if grad_scale is not None:
            for c_g, n_g in zip(cpu_grads, npu_grads):
                self.assertRtolEqual(c_g, n_g.cpu(), prec=prec)

    # === Positive test cases (Coverage Matrix) ===

    def test_basic_adam(self, device="npu"):
        """Basic Adam: amsgrad=False, maximize=False, no grad_scale/found_inf, fp32."""
        torch.manual_seed(0)
        self._run_and_compare(
            shapes=[(4, 4), (8, 8)], dtype=torch.float32,
            lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.0,
            eps=1e-8, amsgrad=False, maximize=False,
            state_steps_values=[1, 1])

    def test_adam_amsgrad(self, device="npu"):
        """Adam with AMSGrad variant: amsgrad=True, max_exp_avg_sqs provided."""
        torch.manual_seed(0)
        self._run_and_compare(
            shapes=[(4, 4), (8, 8)], dtype=torch.float32,
            lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.0,
            eps=1e-8, amsgrad=True, maximize=False,
            state_steps_values=[1, 1])

    def test_adam_maximize(self, device="npu"):
        """Adam with maximize=True (gradient ascent)."""
        torch.manual_seed(0)
        self._run_and_compare(
            shapes=[(4, 4), (8, 8)], dtype=torch.float32,
            lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.0,
            eps=1e-8, amsgrad=False, maximize=True,
            state_steps_values=[1, 1])

    def test_adam_weight_decay(self, device="npu"):
        """Adam with weight_decay > 0."""
        torch.manual_seed(0)
        self._run_and_compare(
            shapes=[(4, 4), (2, 3, 3)], dtype=torch.float32,
            lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01,
            eps=1e-8, amsgrad=False, maximize=False,
            state_steps_values=[1, 1])

    def test_adam_multi_tensor(self, device="npu"):
        """Adam with 5 tensor groups in TensorList."""
        torch.manual_seed(0)
        self._run_and_compare(
            shapes=[(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)], dtype=torch.float32,
            lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.001,
            eps=1e-8, amsgrad=False, maximize=False,
            state_steps_values=[1, 1, 1, 1, 1])

    def test_adam_state_step_2(self, device="npu"):
        """Adam with state_step=2 (second iteration step)."""
        torch.manual_seed(0)
        self._run_and_compare(
            shapes=[(4, 4), (8, 8)], dtype=torch.float32,
            lr=0.01, beta1=0.9, beta2=0.999, weight_decay=0.0,
            eps=1e-8, amsgrad=False, maximize=False,
            state_steps_values=[2, 2])

    def test_adam_found_inf_skip(self, device="npu"):
        """Adam with found_inf=1: skip update, all tensors unchanged."""
        torch.manual_seed(0)
        shapes = [(4, 4), (8, 8)]
        dtype = torch.float32
        (cpu_self_before, cpu_grads, cpu_exp_avgs, cpu_exp_avg_sqs, cpu_max_exp_avg_sqs,
         npu_self_before, npu_grads, npu_exp_avgs, npu_exp_avg_sqs, npu_max_exp_avg_sqs) = \
            self._gen_tensors(shapes, dtype)

        cpu_self = [p.clone() for p in cpu_self_before]
        npu_self = [p.clone() for p in npu_self_before]
        state_steps_cpu = [torch.tensor([1], dtype=torch.int64) for _ in shapes]
        state_steps_npu = [s.npu() for s in state_steps_cpu]
        found_inf = torch.tensor(1, dtype=torch.float32)

        torch._fused_adam_(
            cpu_self, cpu_grads, cpu_exp_avgs, cpu_exp_avg_sqs, [],
            state_steps_cpu,
            lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.0,
            eps=1e-8, amsgrad=False, maximize=False,
            grad_scale=None, found_inf=None)

        torch._fused_adam_(
            npu_self, npu_grads, npu_exp_avgs, npu_exp_avg_sqs, [],
            state_steps_npu,
            lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.0,
            eps=1e-8, amsgrad=False, maximize=False,
            grad_scale=None, found_inf=found_inf.npu())

        # When found_inf=1, params should be unchanged
        for c_before, c_after in zip(cpu_self_before, cpu_self):
            self.assertRtolEqual(c_before, c_after, prec=1e-2)
        for n_before, n_after in zip(npu_self_before, npu_self):
            self.assertRtolEqual(n_before.cpu(), n_after.cpu(), prec=1e-2)

    def test_adam_float16(self, device="npu"):
        """Adam with float16 dtype."""
        torch.manual_seed(0)
        self._run_and_compare(
            shapes=[(4, 4), (8, 8)], dtype=torch.float16,
            lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.0,
            eps=1e-8, amsgrad=False, maximize=False,
            state_steps_values=[1, 1],
            prec=1e-3)

    def test_adam_bfloat16(self, device="npu"):
        """Adam with bfloat16 dtype."""
        torch.manual_seed(0)
        self._run_and_compare(
            shapes=[(4, 4), (8, 8)], dtype=torch.bfloat16,
            lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.0,
            eps=1e-8, amsgrad=False, maximize=False,
            state_steps_values=[1, 1],
            prec=1e-2)

    def test_adam_single_tensor(self, device="npu"):
        """Adam with single tensor in TensorList."""
        torch.manual_seed(0)
        self._run_and_compare(
            shapes=[(16, 16)], dtype=torch.float32,
            lr=0.01, beta1=0.9, beta2=0.999, weight_decay=0.01,
            eps=1e-8, amsgrad=True, maximize=False,
            state_steps_values=[3])

    # === Negative test cases (Negative Coverage Matrix) ===

    def test_neg_mismatched_list_size(self, device="npu"):
        """Negative: TensorList size mismatch, expect RuntimeError."""
        torch.manual_seed(0)
        npu_self = [torch.randn(4, 4, dtype=torch.float32).npu() for _ in range(2)]
        npu_grads = [torch.randn(4, 4, dtype=torch.float32).npu() for _ in range(3)]
        npu_exp_avgs = [torch.zeros(4, 4, dtype=torch.float32).npu() for _ in range(2)]
        npu_exp_avg_sqs = [torch.zeros(4, 4, dtype=torch.float32).npu() for _ in range(2)]
        state_steps = [torch.tensor([1], dtype=torch.int64).npu() for _ in range(2)]
        with self.assertRaises(RuntimeError):
            torch._fused_adam_(
                npu_self, npu_grads, npu_exp_avgs, npu_exp_avg_sqs, [],
                state_steps,
                lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.0,
                eps=1e-8, amsgrad=False, maximize=False)


if __name__ == "__main__":
    run_tests()

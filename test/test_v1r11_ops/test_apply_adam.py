# Copyright (c) 2020, Huawei Technologies.All rights reserved.
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

import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


def _output_m_compute(m, beta1_broad, grad):
    """
    _output_m_compute
    """
    input_dtype = m.dtype

    #sneg_one = tvm.const(-1, dtype=input_dtype)
    sneg_one = torch.ones((1), dtype=input_dtype) * -1

    # `formula; beta1 -1`
    #vsub_beta1_1 = tbe.vadds(beta1_broad, sneg_one)
    vsub_beta1_1 = torch.add(beta1_broad, sneg_one)

    # `formula; m - grad`
    vsub_m_grad = torch.sub(m, grad)

    # `formula; (beta1 - 1) * (m - grad)`
    vmul_m = torch.mul(vsub_beta1_1, vsub_m_grad)

    # `formula; m_t = m + (beta1 - 1) * (m - grad)`
    m_t = torch.add(m, vmul_m)

    return m_t


def _output_v_compute(v, beta2, grad):
    """_output_v_compute
    do compute v_t = v + (1 - beta2)*(grad*grad -v)
    """
    input_dtype = v.dtype
    #shape_m_grad = shape_util.shape_to_list(v.shape)
    shape_m_grad = v.shape
    #sneg_one = tvm.const(-1, dtype=input_dtype)
    sneg_one = torch.ones((1), dtype=input_dtype) * -1

    # `formula; broadcast beta2 to vector`
    #beta2_broad = tbe.broadcast(beta2, shape_m_grad)
    beta2_tensor = torch.tensor(beta2, dtype=input_dtype)
    beta2_broad = beta2_tensor.expand_as(v)

    # `formula; beta2 - 1`
    vsub_beta2_1 = torch.add(beta2_broad, sneg_one)

    # `formula; grad * grad`
    vmul_grad_grad = torch.mul(grad, grad)

    # `formula; (v - grad*grad)`
    vsub_v_grad = torch.sub(v, vmul_grad_grad)

    # `formula; (beta2 -1) * (v - grad * grad)`
    vmul_grad = torch.mul(vsub_beta2_1, vsub_v_grad)

    # `formula; v_t = v + (beta2 - 1) * (v - grad * grad)`
    v_t = torch.add(v, vmul_grad)

    return v_t


def _inner_eps_add_sqrt_vt_compute(epsilon, v_t):
    """
    (epsilon + sqrt(v_t) )
    """
    # `formula; sqrt(v_t)`
    sqrt_vt = torch.sqrt(v_t.to(torch.float32))

    # `formula; broadcast epsilon  to vector`
    compute_shape = v_t.shape
    input_dtype = v_t.dtype
    epsilon_tensor = torch.tensor(epsilon, dtype=input_dtype)
    epsilon_broad = epsilon_tensor.expand_as(v_t)

    # `formula; epsilon + sqrt(v_t)`
    v_add_sqrt_v = torch.add(sqrt_vt.to(torch.float16), epsilon_broad)

    return v_add_sqrt_v


def _inner_lr_compute(lr, beta2_power, beta1_power, compute_shape_tensor):
    """
    _inner_lr_compute
    #lr_t = learning_rate * (sqrt(1-beta2_power)) / (1 - beta1_power)
    """

    input_dtype = compute_shape_tensor.dtype

    s_one = torch.ones((1), dtype=input_dtype)

    s_neg_one = torch.ones((1), dtype=input_dtype) * -1

    # `formula; (1 - beta2_power)`
    v_neg_beta2_power = torch.mul(beta2_power, s_neg_one)
    v_add_beta2_power = torch.add(v_neg_beta2_power, s_one)

    # `formula; sqrt(1 - beta2_power)`
    v_sqrt_beta2_power = torch.sqrt(v_add_beta2_power.to(torch.float32))

    # `formula; (1 - beta1_power)`
    v_neg_beta1_power = torch.mul(beta1_power, s_neg_one)
    v_add_beta1_power = torch.add(v_neg_beta1_power, s_one)

    # `formula; learning_rate * (sqrt(1-beta2_power)`
    res = torch.mul(lr, v_sqrt_beta2_power.to(torch.float16))

    # `formula; learning_rate*(sqrt(1-beta2_power))/(1-beta1_power)`
    res = torch.div(res, v_add_beta1_power)
    return res.expand_as(compute_shape_tensor)


# pylint:disable = huawei-too-many-arguments
def _output_var_t_compute_use_nesterov(var, lr_t, m_t, beta1_broad, grad, epsilon, v_t):
    """
    _output_var_t_compute_use_nesterov
    # var_t = var - lr_t * (m_t * beta1 + (1 - beta1) * grad) / (epsilon + sqrt(v_t))
    # var_t = var - lr_t * (m_t * beta1 + (1 - beta1) * grad) / (epsilon + sqrt(v_t))
    """
    input_dtype = var.dtype

    compute_shape = var.shape

    s_one = torch.ones((1), dtype=input_dtype)

    s_neg_one = torch.ones((1), dtype=input_dtype) * -1

    # `formula; m_t * beta1`
    v_muls_mt_beta1 = torch.mul(m_t, beta1_broad)

    # `formula; 1 -beta1`
    v_neg_beta1 = torch.mul(beta1_broad, s_neg_one)
    vsub_1_beta1 = torch.add(v_neg_beta1, s_one)

    # `formula; (1-beta1)* grad`
    v_mul_grad = torch.mul(vsub_1_beta1, grad)

    # `formula; (m_t*beta1 + (1 - beta1)*grad)`
    v_div_left = torch.add(v_muls_mt_beta1, v_mul_grad)

    # `formula; lr_t * (m_t*beta1 + (1 - beta1) * grad)`
    # broadcast lr_t to vector

    lrt_broad = lr_t.expand_as(var)
    v_mul_left = torch.mul(lrt_broad, v_div_left)

    # `formula; (epsilon + sqrt(v_t))`
    v_add_sqrt_v = _inner_eps_add_sqrt_vt_compute(epsilon, v_t)

    # `formula; lr_t * (m_t*beta1 + (1-beta1)*grad / (epsilon + sqrt(v_t))`
    v_div_res = torch.div(v_mul_left, v_add_sqrt_v)

    # `formula; var - lr_t * (m_t*beta1 + (1-beta1)*grad) / (epsilon + sqrt(v_t))`
    v_t = torch.sub(var, v_div_res)

    return v_t


# `var_t = var - lr_t * m_t / (epsilon + sqrt(v_t))`
def _output_var_t_compute(var, lr_t, m_t, epsilon, v_t):
    """
    _output_var_t_compute
    `var_t = var - lr_t * m_t / (epsilon + sqrt(v_t))`
    """
    # `formula; lr_t * m_t`
    v_mul_left = torch.mul(lr_t, m_t)

    # `formula; (epsilon + sqrt(v_t))`
    v_add_sqrt_v = _inner_eps_add_sqrt_vt_compute(epsilon, v_t)

    # `formula; lr_t * m_t /(epsilon + sqrt(v_t))`
    v_div_res = torch.div(v_mul_left, v_add_sqrt_v)

    # `formula; var - lr_t * m_t / (epsilon + sqrt(v_t))`
    v_t = torch.sub(var, v_div_res)

    return v_t


# pylint:disable = huawei-too-many-arguments
def apply_adam_d(beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, use_locking, use_nesterov, var, m, v):


    shape_m_grad = m.shape
    input_dtype = m.dtype

    beta1_tensor = torch.tensor(beta1, dtype=input_dtype)
    beta1_broad = beta1_tensor.expand_as(m)
    m_t = _output_m_compute(m, beta1_broad, grad)
    v_t = _output_v_compute(v, beta2, grad)


    compute_shape = m.shape
    lr_r = _inner_lr_compute(lr, beta2_power, beta1_power, m)

    if use_nesterov is True:
        var_t = _output_var_t_compute_use_nesterov(var, lr_r, m_t, beta1_broad, grad, epsilon, v_t)
    else:
        var_t = _output_var_t_compute(var, lr_r, m_t, epsilon, v_t)

    res = [var_t, m_t, v_t]
    return res


class TestApplyAdam(TestCase):
    def test_apply_adam(self):
        var1 = torch.randn(2, 2, 2, 2, dtype=torch.float32)
        m1 = torch.randn(2, 2, 2, 2, dtype=torch.float32)
        v1 = torch.randn(2, 2, 2, 2, dtype=torch.float32)
        grad1 = torch.randn(2, 2, 2, 2, dtype=torch.float32)
        var2 = var1.to(torch.half)
        m2 = m1.to(torch.half)
        v2 = v1.to(torch.half)
        grad2 = grad1.to(torch.half)
        _, _, v1_c = apply_adam_d(1, 1, 0.2, 0.2, 0.2, 0.2, grad1, False, False, var1, m1, v1)
        _, _, v2_c = apply_adam_d(1, 1, 0.2, 0.2, 0.2, 0.2, grad2, False, False, var2, m2, v2)
        _, _, v1_o = torch_npu.npu_apply_adam(1, 1, 0.2, 0.2, 0.2, 0.2, grad1.to("npu"), False, False, out=(var1.to("npu"), m1.to("npu"), v1.to("npu")))
        _, _, v2_o = torch_npu.npu_apply_adam(1, 1, 0.2, 0.2, 0.2, 0.2, grad2.to("npu"), False, False, out=(var2.to("npu"), m2.to("npu"), v2.to("npu")))

        self.assertRtolEqual(v1_c, v1_o.cpu())
        self.assertRtolEqual(v2_c, v2_o.cpu())

    def test_apply_adam_out_fp32(self):
        var = torch.randn(2, 2, 2, 2, dtype=torch.float32)
        m = torch.randn(2, 2, 2, 2, dtype=torch.float32)
        v = torch.randn(2, 2, 2, 2, dtype=torch.float32)
        grad = torch.randn(2, 2, 2, 2, dtype=torch.float32)
        bt1p = 0.9
        bt2p = 0.9
        lr = 0.2
        bt1 = 0.2
        bt2 = 0.2
        ep = 0.2
        ul = False
        un = False
        var_c, m_c, v_c = apply_adam_d(bt1p, bt2p, lr, bt1, bt2, ep, grad, ul, un, var, m, v)
        var_o, m_o, v_o = torch_npu.npu_apply_adam(bt1p, bt2p, lr, bt1, bt2, ep, grad.to("npu"), ul, un, out=(var.to("npu"), m.to("npu"), v.to("npu")))
        # self.assertRtolEqual(var_c, var_o.cpu())
        self.assertRtolEqual(m_c, m_o.cpu())
        self.assertRtolEqual(v_c, v_o.cpu())


if __name__ == "__main__":
    run_tests()

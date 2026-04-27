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

import itertools
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices

def hc_post_forward(h_out, x, h_post, h_res):
    y = h_post.unsqueeze(-1) * h_out.unsqueeze(-2) + torch.sum(h_res.unsqueeze(-1) * x.unsqueeze(-2), dim=-3)
    return y

class TestNpuMhcPostBackward(TestCase):

    def cpu_op_exec(self, grad_output, x, h_res, h_out, h_post):
        dtype = grad_output.dtype

        x_f32 = x.float().requires_grad_(True)
        h_res_f32 = h_res.float().requires_grad_(True)
        h_out_f32 = h_out.float().requires_grad_(True)
        h_post_f32 = h_post.float().requires_grad_(True)

        y = hc_post_forward(h_out_f32, x_f32, h_post_f32, h_res_f32)
        grad_output_f32 = grad_output.float()
        y.backward(grad_output_f32)

        grad_h_out = h_out_f32.grad.to(dtype)
        grad_x = x_f32.grad.to(dtype)
        grad_h_post = h_post_f32.grad
        grad_h_res = h_res_f32.grad

        return grad_x, grad_h_res, grad_h_out, grad_h_post

    def npu_op_exec(self, grad_output, x, h_res, h_out, h_post):
        grad_x, grad_h_res, grad_h_out, grad_h_post = torch_npu.npu_mhc_post_backward(grad_output, x, h_res, h_out, h_post)
        return grad_x, grad_h_res, grad_h_out, grad_h_post
    
    @SupportedDevices(['Ascend950'])
    def test_npu_mhc_post_backward(self, device="npu"):
        
        grad_output_shape = (1, 4, 1024)
        x_shape = (1, 4, 1024)
        h_res_shape = (1, 4, 4)
        h_out_shape = (1, 1024)
        h_post_shape = (1, 4)
        grad_output = torch.rand(grad_output_shape, dtype=torch.float16)
        x = torch.rand(x_shape, dtype=torch.float16)
        h_res = torch.rand(h_res_shape, dtype=torch.float32)
        h_out = torch.rand(h_out_shape, dtype=torch.float16)
        h_post = torch.rand(h_post_shape, dtype=torch.float32)
        grad_output_npu = grad_output.npu()
        x_npu = x.npu()
        h_res_npu = h_res.npu()
        h_out_npu = h_out.npu()
        h_post_npu = h_post.npu()
        grad_x, grad_h_res, grad_h_out, grad_h_post = self.cpu_op_exec(grad_output, x, h_res, h_out, h_post)
        grad_x_npu, grad_h_res_npu, grad_h_out_npu, grad_h_post_npu = self.npu_op_exec(grad_output_npu, x_npu, h_res_npu, h_out_npu, h_post_npu)
        self.assertRtolEqual(grad_x.numpy(), grad_x_npu.cpu().numpy())
        self.assertRtolEqual(grad_h_res.numpy(), grad_h_res_npu.cpu().numpy())
        self.assertRtolEqual(grad_h_out.numpy(), grad_h_out_npu.cpu().numpy())
        self.assertRtolEqual(grad_h_post.numpy(), grad_h_post_npu.cpu().numpy())

if __name__ == "__main__":
    run_tests()
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

def hc_post(h_out, H_post, H_comb, x):
    h_out_fp32 = h_out.float()
    x_fp32 = x.float()
    h_post_term = H_post.unsqueeze(-1) * h_out_fp32.unsqueeze(2)
    h_comb_term = torch.sum(H_comb.unsqueeze(-1) * x_fp32.unsqueeze(-2), dim=2)
    y = h_post_term + h_comb_term
    return y

class TestNpuMhcPost(TestCase):

    def cpu_op_exec(self, x, h_res, h_out, h_post):

        x_dtype = x.dtype
        x_ndim = x.ndim
        if (x_dtype == 'bfloat16'):
            x = x.to(torch.float32)
            h_out = h_out.to(torch.float32)
            
        if (x_ndim == 3):
            x = torch.unsqueeze(x, dim=0)
            h_res = torch.unsqueeze(h_res, dim=0)
            h_out = torch.unsqueeze(h_out, dim=0)
            h_post = torch.unsqueeze(h_post, dim=0)
        
        h_out_fp32 = h_out.float()
        x_fp32 = x.float()
        h_post_term = h_post.unsqueeze(-1) * h_out_fp32.unsqueeze(-2)
        h_comb_term = torch.sum(h_res.unsqueeze(-1) * x_fp32.unsqueeze(-2), dim=-3)
        y_tensor = h_post_term + h_comb_term

        if (x_ndim == 3):
            y_tensor = torch.squeeze(y_tensor, dim=0)
        y_tensor.to(x_dtype)
        return y_tensor

    def npu_op_exec(self, x, h_res, h_out, h_post):
        y = torch_npu.npu_mhc_post(x, h_res, h_out, h_post)
        return y
    
    @SupportedDevices(['Ascend950'])
    def test_npu_mhc_post(self, device="npu"):
        x_shape = [1,1,4,512]
        h_res_shape = [1,1,4,4]
        h_out_shape = [1,1,512]
        h_post_shape = [1,1,4]
        x = torch.rand(x_shape, dtype=torch.float16)
        h_res = torch.rand(h_res_shape, dtype=torch.float32)
        h_out = torch.rand(h_out_shape, dtype=torch.float16)
        h_post = torch.rand(h_post_shape, dtype=torch.float32)
        x_npu = x.npu()
        h_res_npu = h_res.npu()
        h_out_npu = h_out.npu()
        h_post_npu = h_post.npu()
        y = self.cpu_op_exec(x, h_res, h_out, h_post)
        y_npu = self.npu_op_exec(x_npu, h_res_npu, h_out_npu, h_post_npu)
        self.assertRtolEqual(y.numpy(), y_npu.cpu().numpy())

if __name__ == "__main__":
    run_tests()
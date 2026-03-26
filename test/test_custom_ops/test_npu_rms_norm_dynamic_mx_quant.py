# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.

# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import math
import numpy as np
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices
from torch.testing import assert_close

class TestRmsNormDynamicMxQuant(TestCase):

    # 此处传递的是已经.npu()后的
    def npu_op_exec(self, x, gamma, beta=None, epsilon=1e-6, scale_alg=0, round_mode="rint", dst_type=torch_npu.float8_e5m2):
        return torch_npu.npu_rms_norm_dynamic_mx_quant(x, gamma, beta=beta, epsilon=epsilon, scale_alg=scale_alg, 
                                                       round_mode=round_mode, dst_type=dst_type)

    def golden_op_exec(self, input_tensor):
        if torch.all(torch.eq(input_tensor, 0.0)) and input_tensor.shape == torch.Size([1, 2, 2]):
            device = input_tensor.device
            y = torch.tensor([[[0, 0], [0, 0]]], dtype=torch.float8_e5m2, device=device)
            mxscale = torch.tensor([[[[0, 0]], [[0, 0]]]], dtype=torch.uint8, device=device)
            rstd = torch.tensor([[[1000], [1000]]], dtype=torch.float32, device=device)
            return y, mxscale, rstd

    def generate_input(self, input, value, dtype="float16"):
        if dtype == "float32":
            data_type = torch.float32
        elif dtype == "float16":
            data_type = torch.float16
        elif dtype == "bfloat16":
            data_type = torch.bfloat16
        input_tensor = torch.full(input, value, dtype=data_type)
        return input_tensor

    @SupportedDevices(['Ascend950'])
    def test_npu_rms_norm_quant_float8_e5m2_with_rstd(self, device="npu"):
        x = self.generate_input(input=[1, 2, 2], value=0.0, dtype="bfloat16")
        C = x.shape[-1]
        gamma = self.generate_input(input=[C], value=0.0, dtype="float32")
        beta = self.generate_input(input=[C], value=0.0, dtype="float32")
        x = x.to(device).requires_grad_(True)

        gamma = gamma.to(device)
        beta = beta.to(device)
        eps = 1e-6
        scale_alg=0
        round_mode="rint"
        out_dtype = 23

        golden_output = self.golden_op_exec(x)
        npu_output = self.npu_op_exec(x, gamma, beta=beta, epsilon=eps, scale_alg=scale_alg, round_mode=round_mode, dst_type=out_dtype)
        y = npu_output[0].view([1, 2, 2]).view(torch.uint8)
        mxscale = npu_output[1].view([1, 2, 1, 2]).to(torch.uint8)
        rstd = npu_output[2].view([1, 2, 1]).to(torch.float32)

        assert torch.all(y == golden_output[0].view(torch.uint8))
        assert_close(golden_output[1], mxscale, atol=0.01, rtol=0.001)
        assert torch.all(rstd == golden_output[2].view(torch.float32))


if __name__ == "__main__":
    run_tests()
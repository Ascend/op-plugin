# Copyright (c) 2020, Huawei Technologies.All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at relate links.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torch_npu
import torch.nn.functional as F
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests

def gen_quant_conv2d_golden(fmap_tensor, weight_tensor, cout, stride, padding, dilation, groups):
    # int8 + int8 -> int32 -> dequant -> fp16
    device = torch.device("cpu")
    cpu_out = F.conv2d(fmap_tensor.to(torch.int32).to(device),
                                weight_tensor.to(torch.int32).to(device),
                                None,
                                stride,
                                padding,
                                dilation,
                                groups).cpu().to(torch.int32)
    scale_np = np.random.uniform(1, 2, size=[cout]).astype(np.float32)
    scale_np = np.bitwise_and(scale_np.view(np.uint32), 0xffffe000).view(np.float32)
    scale_np.view(np.uint32).astype(np.uint64) # uint64
    scale_tensor = torch.from_numpy(scale_np.reshape(1, scale_np.shape[0], 1, 1)) # NCHW
    scale_out = torch.multiply(cpu_out, scale_tensor)
    res = scale_out.to(torch.float16)
    return res, scale_tensor

class TestQuantMatmul(TestCase):
    @SupportedDevices(['Ascend910_95', 'Ascend950'])
    def test_npu_quant_conv2d_int8(self):
        torch.manual_seed(0)
        conv_input = torch.randint(-1, 1, (1, 1, 4, 4), dtype=torch.int8)
        weight = torch.randint(-1, 1, (1, 1, 3, 3), dtype=torch.int8)
        cout = 1
        stride = tuple(1,1)
        padding = tuple(0,0)
        dilation = tuple(1,1)
        groups = 1
        offset_x = 0
        round_mode = "rint"
        output_dtype = torch.float16
        bias = None
        offset = None
        input_dtype = None
        weight_dtype = None
        supported_output, scale = gen_quant_conv2d_golden(conv_input, weight, cout, stride, padding, dilation, groups)
        custom_output = torch_npu.npu_quant_matmul(
            conv_input, weight, scale, stride, padding,
            dilation, groups, offset_x, round_mode, output_dtype,
            bias, offset, input_dtype, weight_dtype)
        self.assertRtolEqual(supported_output.float().cpu().numpy(), custom_output.float().cpu().numpy(), 0.01)

if __name__ == "__main__":
    run_tests()

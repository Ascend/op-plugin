# Copyright (c) 2025, Huawei Technologies.All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices


class TestNpuCompress(TestCase):

    def cpu_op_exec(self, input_cpu, weight, compress_block_size, compress_stride, actual_seq_len):
        input_dtype = input_cpu.dtype
        input_cpu = input_cpu.to(torch.float)
        weight = weight.unsqueeze(-1).expand(-1, -1, input_cpu.shape[2])
        weight = weight.to(torch.float)
        output_shape_0 = 0
        pre_seq_len = 0

        for _, x in enumerate(actual_seq_len):
            cur_seq_len = x - pre_seq_len
            pre_seq_len = x
            if cur_seq_len >= compress_block_size:
                output_shape_0 += (cur_seq_len - compress_block_size) // compress_stride + 1
        output = torch.zeros((output_shape_0, input_cpu.shape[1], input_cpu.shape[2]), dtype=input_dtype)

        output = output.to(torch.float)
        token_idx = 0
        for sample_id, seq_len in enumerate(actual_seq_len):
            if sample_id != 0:
                seq_len = actual_seq_len[sample_id] - actual_seq_len[sample_id - 1]
            # 跳短Sample
            if seq_len < compress_block_size:
                continue

            for start in range(0, seq_len, compress_stride):
                # 用于跳结尾
                if (start + compress_block_size > seq_len):
                    break

                if sample_id:
                    start_in_all_sample = start + actual_seq_len[sample_id - 1]
                else:
                    start_in_all_sample = start + 0

                output[token_idx] = torch.sum(input_cpu[start_in_all_sample:start_in_all_sample + compress_block_size] * weight, axis=0)
                token_idx += 1

        output = output.to(input_dtype)
        return output


    def npu_op_exec(self, input_npu, weight, compress_block_size, compress_stride, actual_seq_len):
        output = torch_npu.npu_nsa_compress(input_npu, weight, compress_block_size, compress_stride, actual_seq_len=actual_seq_len)
        return output

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_nsa_compress(self):
        actual_seq_len = np.random.randint(0, 100, [48])
        actual_seq_len = np.cumsum(actual_seq_len).astype(np.int64)
        head_num = 4
        head_dim = 128
        compress_block_size = 16
        compress_stride = 16
        input_ori = torch.randn(actual_seq_len[-1], head_num, head_dim, dtype=torch.float16)
        weight = torch.randn(compress_block_size, head_num, dtype=torch.float16)

        npuout = self.npu_op_exec(input_ori.npu(), weight.npu(), compress_block_size, compress_stride, actual_seq_len)
        gloden_out = self.cpu_op_exec(input_ori, weight, compress_block_size, compress_stride, actual_seq_len)

        self.assertRtolEqual(gloden_out.numpy(), npuout.cpu().numpy())

if __name__ == "__main__":
    run_tests()

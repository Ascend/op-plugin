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


class TestNpuCompressInfer(TestCase):
    # pylint:disable = huawei-too-many-arguments
    def get_last_kv(self, input_kv, act_seq_lens, block_table, batch_idx, compress_block_size, page_block_size):
        act_seq_len = act_seq_lens[batch_idx]
        block_num = (act_seq_len - 1) // page_block_size
        tile_act_seq_len = act_seq_len - block_num * page_block_size
        if tile_act_seq_len < compress_block_size:
            cur_block_idx = block_table[batch_idx, block_num]
            pre_block_idx = block_table[batch_idx, block_num - 1]
            input_kv_cur = input_kv[cur_block_idx, 0:tile_act_seq_len]
            input_kv_pre = input_kv[pre_block_idx, page_block_size - compress_block_size + tile_act_seq_len:page_block_size]
            return torch.concatenate((input_kv_pre, input_kv_cur), axis=0)

        block_idx = block_table[batch_idx, block_num]
        return input_kv[block_idx, tile_act_seq_len - compress_block_size:tile_act_seq_len]

    # pylint:disable = huawei-too-many-arguments
    def cpu_op_exec(self, input_cpu, weight, slot_mapping, compress_block_size, compress_stride, page_block_size, block_table, actual_seq_len, cache):
        input_dtype = input_cpu.dtype
        input_cpu = input_cpu.to(torch.float32)
        weight = weight.to(torch.float32)
        cache = cache.to(torch.float32)
        batch_size = slot_mapping.shape[0]
        for batch_idx in range(batch_size):
            if(actual_seq_len[batch_idx] >= compress_block_size and (actual_seq_len[batch_idx] - compress_block_size) % compress_stride == 0):
                last_kv = get_last_kv(input_cpu, actual_seq_len, block_table, batch_idx, compress_block_size, page_block_size)
                weight_last_kv = last_kv * weight.unsqueeze(2)
                compress_last_kv = weight_last_kv.sum(axis=0)
                cache[slot_mapping[batch_idx]] = compress_last_kv

        return cache.to(input_dtype)

    # pylint:disable = huawei-too-many-arguments
    def npu_op_exec(self, input_npu, weight, slot_mapping, compress_block_size, compress_stride, page_block_size, block_table, actual_seq_len, cache):
        output = torch_npu.npu_nsa_compress_infer(input_npu, weight, slot_mapping, compress_block_size, compress_stride, page_block_size, block_table=block_table, actual_seq_len=actual_seq_len, cache=cache)
        return output

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_nsa_compress_infer(self):
        input_ori = torch.randn(1, 128, 1, 192, dtype=torch.float16)
        weight = torch.randn(32, 1, dtype=torch.float16)
        slot_mapping = torch.randn([1]).int()
        compress_block_size = 32
        compress_stride = 16
        page_block_size = 128
        block_table = torch.randn([1, 1]).int()
        actual_seq_len = [43]
        cache = torch.zeros([1, 1, 192], dtype=torch.float16)

        npuout = self.npu_op_exec(input_ori.npu(), weight.npu(), slot_mapping.npu(), compress_block_size, compress_stride, page_block_size, block_table.npu(), actual_seq_len, cache.npu())
        gloden_out = self.cpu_op_exec(input_ori, weight, slot_mapping, compress_block_size, compress_stride, page_block_size, block_table, actual_seq_len, cache)

        self.assertRtolEqual(gloden_out.numpy(), npuout.cpu().numpy())

if __name__ == "__main__":
    run_tests()

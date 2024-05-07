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


class TestNpuMoeInitRouting(TestCase):

    def cpu_op_exec(self, x, row_idx, expert_idx, active_num):
        num_rows = x.shape[0]
        hidden_size = x.shape[-1]
        k = expert_idx.shape[-1]
        sort_expert_for_source_row = np.argsort(
            expert_idx.reshape((-1,)), axis=-1, kind="stable")
        expanded_expert_idx = np.sort(
            expert_idx.reshape((-1,)), axis=-1)

        expanded_dst_to_src_row = np.take_along_axis(
            row_idx.reshape((-1,)), sort_expert_for_source_row, axis=-1)
        expanded_row_idx = np.zeros(expanded_dst_to_src_row.shape).astype(np.int32)
        expanded_row_idx[expanded_dst_to_src_row] = np.arange(
            expanded_dst_to_src_row.shape[-1])
        active_num = min(active_num, num_rows) * k
        expanded_x = x[expanded_dst_to_src_row[:active_num] % num_rows, :]
        return expanded_x, expanded_row_idx, expanded_expert_idx

    def npu_op_exec(self, x, row_idx, expert_idx, active_num):
        expanded_x, expanded_row_idx, expanded_expert_idx = torch_npu.npu_moe_init_routing(x, row_idx, expert_idx,
                                                                                           active_num)
        return expanded_x, expanded_row_idx, expanded_expert_idx
    
    @SupportedDevices(['Ascend910B'])
    def test_npu_noe_init_routing(self, device="npu"):
        n_list = [10, 430, 520]
        k_list = [2, 4, 5, 9]
        col_list = [200, 1256, 5120]
        dtype_list = [np.float16, np.float32]
        for n, k, col, dtype in itertools.product(n_list, k_list, col_list, dtype_list):
            x = np.random.uniform(-1, 1, size=(n, col)).astype(dtype)
            row_idx = np.arange(n * k).reshape([k, n]).transpose(1, 0).astype(np.int32)
            expert_idx = np.random.randint(0, 100, size=(n, k)).astype(np.int32)
            x_npu = torch.from_numpy(x).npu()
            row_idx_npu = torch.from_numpy(row_idx).contiguous().npu()
            expert_idx_npu = torch.from_numpy(expert_idx).npu()
            active_num = n
            
            expanded_x, expanded_row_idx, expanded_expert_idx = self.cpu_op_exec(x, row_idx, expert_idx, active_num)
            expanded_x_npu, expanded_row_idx_npu, expanded_expert_idx_npu = self.npu_op_exec(x_npu, row_idx_npu,
                                                                                        expert_idx_npu, active_num)
            self.assertRtolEqual(expanded_x, expanded_x_npu.cpu().numpy())
            self.assertRtolEqual(expanded_row_idx, expanded_row_idx_npu.cpu().numpy())
            self.assertRtolEqual(expanded_expert_idx, expanded_expert_idx_npu.cpu().numpy())

if __name__ == "__main__":
    run_tests()

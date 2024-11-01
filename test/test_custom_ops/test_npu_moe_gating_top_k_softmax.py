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


class TestNpuMoeGatingTopKSoftmax(TestCase):

    def softmax_func(self, x, axis=None):
        is_fp16 = x.dtype == np.float16
        x = x.astype(np.float32)
        x_max = x.max(axis=-1, keepdims=True)
        x_sub = x - x_max
        y = np.exp(x_sub)
        x_sum = y.sum(axis=-1, keepdims=True)
        if (x_sum == 0).any():
            ans = 0
        else:
            ans = y / x_sum
        if is_fp16:
            ans = ans.astype(np.float16)
            x_max = x_max.astype(np.float16)
            x_sum = x_sum.astype(np.float16)
        return ans, x_max, x_sum

    def cpu_op_exec(self, x, finished_optional, k):
        num_expert = x.shape[-1]
        softmax, _, _, = self.softmax_func(x, -1)
        expert_idx = np.argsort(-softmax, axis=-1, kind='stable')
        expert_idx = expert_idx[:, :k]
        y = np.take_along_axis(softmax, expert_idx, axis=-1)
        if finished_optional is not None:
            finished_optional = finished_optional.reshape(finished_optional.shape[0], 1)
            finished_optional = np.tile(finished_optional, (1, k))
            expert_idx = np.where(finished_optional, num_expert, expert_idx)
        row_idx = np.arange(y.shape[0] * y.shape[1]).reshape(y.shape[1], y.shape[0]).transpose(1, 0)
        if x.dtype == np.float16:
            y = y.astype(np.float16)
        return y, expert_idx.astype(np.int32), row_idx.astype(np.int32)

    def npu_op_exec(self, x, finished, k):
        y, expert_idx, row_idx = torch_npu.npu_moe_gating_top_k_softmax(x, finished, k)
        return y, expert_idx, row_idx

    @SupportedDevices(['Ascend910B'])
    def test_npu_noe_init_routing(self, device="npu"):
        n_list = [10, 430, 520]
        k_list = [2, 4, 5, 9]
        col_list = [200, 1256, 5120]
        dtype_list = [np.float16, np.float32]
        for n, k, col, dtype in itertools.product(n_list, k_list, col_list, dtype_list):
            x = np.random.uniform(-1, 1, size=(n, col)).astype(dtype)
            finished = np.random.uniform(-1, 1, size=(n,)).astype(bool)
            x_npu = torch.from_numpy(x).npu()
            finished_npu = torch.from_numpy(finished).npu()

            y, expert_idx, row_idx = self.cpu_op_exec(x, finished, k)
            y_npu, expert_idx_npu, row_idx_npu = self.npu_op_exec(x_npu, finished_npu, k)
            self.assertRtolEqual(y, y_npu.cpu().numpy())
            self.assertRtolEqual(expert_idx, expert_idx_npu.cpu().numpy())
            self.assertRtolEqual(row_idx, row_idx_npu.cpu().numpy())


if __name__ == "__main__":
    run_tests()

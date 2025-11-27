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

import itertools
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices


class TestNpuMoeGatingTopKSoftmaxV2(TestCase):

    def softmax_func(self, x, axis=None):
        m = np.max(x, axis=axis, keepdims=True)
        e = np.exp(x - m)
        return e / np.sum(e, axis=axis, keepdims=True)

    def cpu_op_exec(self, x, k, finished, renorm, output_softmax):
        num_expert = x.shape[-1]
        leading_shape = x.shape[:-1]
        x_2d = x

        if x_2d.ndim == 3:
            x_2d = x_2d.reshape(-1, num_expert)
            if finished is not None:
                finished = finished.flatten()
        else:
            if finished is not None and finished.ndim > 1:
                finished = finished.reshape(-1)
        
        softmax_full_f32 = self.softmax_func(x_2d.astype(np.float32), axis=-1)

        if renorm == 1:
            indices = np.argsort(-x_2d, axis=-1, kind='stable')[:, :k]
            values = np.take_along_axis(x_2d, indices, axis=-1)
            out = self.softmax_func(values.astype(np.float32), axis=-1).astype(x.dtype)
        else:
            indices = np.argsort(-softmax_full_f32, axis=-1, kind='stable')[:, :k]
            out = np.take_along_axis(softmax_full_f32, indices, axis=-1).astype(x.dtype)
        
        indices = indices.astype(np.int32)
        if finished is not None:
            finished_expanded = np.tile(finished.reshape(-1, 1), (1, k))
            indices = np.where(finished_expanded, num_expert, indices)

        out = out.reshape(*leading_shape, k)
        indices = indices.reshape(*leading_shape, k)

        if renorm == 0 and output_softmax:
            softmax_full = softmax_full_f32.reshape(*leading_shape, num_expert)
            return out, indices, softmax_full
        else:
            empty_softmax = np.array([], dtype=softmax_full_f32.dtype)
            return out, indices, empty_softmax

    def npu_op_exec(self, x, k, finished, renorm, output_softmax):
        y, indices, softmax_result = torch_npu.npu_moe_gating_top_k_softmax_v2(
            x=x, k=k, finished=finished, renorm=renorm, output_softmax=output_softmax)
        return y, indices, softmax_result

    @SupportedDevices(['Ascend910B'])
    def test_npu_noe_gating_top_k_softmax_v2(self, device="npu"):
        n_list = [10, 430, 520]
        k_list = [2, 4, 5, 9]
        col_list = [200, 1256, 5120]
        flag_list = [True, False]
        renorm_list = [0, 1]
        dtype_list = [np.float16, np.float32]
        for n, k, col, flag, renorm, dtype in itertools.product(n_list, k_list, col_list, flag_list, renorm_list, dtype_list):
            x = np.random.uniform(-1, 1, size=(n, col)).astype(dtype)
            finished = np.random.uniform(-1, 1, size=(n,)).astype(bool)
            x_npu = torch.from_numpy(x).npu()
            finished_npu = torch.from_numpy(finished).npu()

            try:
                y, expert_idx, output_softmax = self.cpu_op_exec(x, k, finished, renorm, flag)
                y_npu, expert_idx_npu, output_softmax_npu = self.npu_op_exec(x_npu, k, finished_npu, renorm, flag)
                self.assertRtolEqual(y, y_npu.cpu().numpy())
                self.assertRtolEqual(expert_idx, expert_idx_npu.cpu().numpy())
                self.assertRtolEqual(output_softmax, output_softmax_npu.cpu().numpy())

            except Exception as e:
                raise AssertionError(f"Task failed unecpectedly: {e}") from e

            


if __name__ == "__main__":
    run_tests()
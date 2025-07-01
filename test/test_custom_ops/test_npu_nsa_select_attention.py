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
import numpy as np
import torch
import torch_npu
import tqdm
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices


class TestNpuCompress(TestCase):

    def tsoftmax(self, x):
        x_max = torch.max(x, dim=-1, keepdims=True)[0]
        x_sub = x.sub(x_max)
        y = torch.exp(x_sub)
        x_sum = y.sum(dim=-1, keepdims=True)
        softmax_res = y.div(x_sum)
        return softmax_res, x_max, x_sum


    def simpleSoftmax(self, x, x_max, x_sum):
        x_sub = x.sub(x_max)
        y = torch.exp(x_sub)
        softmax_res = y.div(x_sum)
        return softmax_res

    def tsoftmax_grad(self, p, dp, out, outGrad):
        muls = outGrad * out
        muls_res = muls.sum(dim=-1, keepdims=True)
        sub_res = dp - muls_res
        res = sub_res * p
        return res


    def get_tnd_idx(self, actual_q_len, t_idx):
        b_idx = 0
        while t_idx >= actual_q_len[b_idx]:
            b_idx += 1
        if b_idx == 0:
            s1_offset = 0
        else:
            s1_offset = actual_q_len[b_idx - 1]
        s1_idx = t_idx - s1_offset

        return b_idx, s1_idx

    def get_currentB_index(self, BS_index, actual_seq_len):
        for i, x in enumerate(actual_seq_len):
            if BS_index < x:
                return i
        raise RuntimeError(f"BS_index is greater than max(actual_seq_len).")

    # pylint:disable = huawei-too-many-arguments
    def cpu_op_exec(self, query, key, value, topk_indices, scale_value, head_num, select_block_size, select_block_count, atten_mask, actual_seq_qlen, actual_seq_kvlen):
        def compute_select_attention(query, key, value, topK_index, atten_mask, actual_seq_qlen=None, actual_seq_kvlen=None, scalar_value=1.0, block_size=64, block_count=16):
            """
            完整的注意力计算流程
            Args:
                query: 查询张量 [BS1, N1, queryKD]
                key: 键张量 [BS2, N2, queryKD]
                value: 值张量 [BS2, N2, VD]
                topK_index: 索引张量 [BS1, N2, block_count]
                atten_mask: 索引张量 [S1, S2]
            Returns:
                output: 最终输出结果 [BS1, N1, VD]
            """
            # 步骤1：处理QKV，以及top_k的维度，将维度保持一致
            ori_dtype = query.dtype
            query = query.float()
            key = key.float()
            value = value.float()
            BS1, N1, QKD = query.shape
            BS2, N2, VD = value.shape
            G = N1 // N2
            query = query.reshape(BS1, N2 * G, QKD) # [BS1, N1, QKD]
            key = key # [BS2, N2, QKD]
            value = value # [BS2, N2, VD]
            topK_index = topK_index # [BS1, N2, 16]
            N1 = N2 * G
            key_reshaped = key.view(BS2 // block_size, block_size, N2, QKD)
            value_reshaped = value.view(BS2 // block_size, block_size, N2, VD)
            query_reshaped = query.view(BS1, N2, G, QKD)

            # output
            output = torch.zeros(BS1, N2, G, VD, dtype=torch.float32, device=query_reshaped.device)
            softmax_max = torch.zeros(BS1, N2, G, 1, dtype=torch.float32, device=query_reshaped.device)
            softmax_sum = torch.zeros(BS1, N2, G, 1, dtype=torch.float32, device=query_reshaped.device)

            for bs1_index in tqdm.tqdm(range(BS1)):
                b_index = get_currentB_index(bs1_index, actual_seq_qlen)
                s1_index = bs1_index if b_index == 0 else bs1_index - actual_seq_qlen[b_index - 1]
                start_BS2_index = 0 if b_index == 0 else actual_seq_kvlen[b_index - 1]
                start_block_index = start_BS2_index // block_size
                current_S2_size = actual_seq_kvlen[b_index] - start_BS2_index
                current_block_count = current_S2_size // block_size
                end_block_index = start_block_index + current_block_count
                if start_BS2_index % block_size != 0:
                    raise RuntimeError(f"当前golden只支持S2 {block_size}对齐.")

                for n2_index in range(N2):
                    topk_index_ = topK_index[bs1_index, n2_index, :]
                    selectedK = torch.index_select(key_reshaped[start_block_index:end_block_index, :, n2_index, :], 0, topk_index_) # (block_count, block_size, QKD)
                    selectedK = selectedK.reshape(block_count * block_size, QKD)
                    qk_ = torch.matmul(query_reshaped[bs1_index, n2_index, :, :], selectedK.transpose(-1, -2)) # (G, block_count * block_size)
                    qk_ = torch.mul(qk_, scalar_value)
                    softmax_res, x_max, x_sum = softmax_torch(qk_)
                    selectedV = torch.index_select(value_reshaped[start_block_index:end_block_index, :, n2_index, :], 0, topk_index_) # (block_count, block_size, VD)
                    selectedV = selectedV.reshape(block_count * block_size, VD)
                    out_ = torch.matmul(softmax_res.to(ori_dtype).float(), selectedV.to(ori_dtype).float()) # (G, VD)

                    output[bs1_index, n2_index] = out_
                    softmax_max[bs1_index, n2_index] = x_max
                    softmax_sum[bs1_index, n2_index] = x_sum

            output = output.reshape(BS1, N1, VD)
            softmax_max = softmax_max.reshape(BS1, N1, 1)
            softmax_sum = softmax_sum.reshape(BS1, N1, 1)

            softmax_max_8 = softmax_max.broadcast_to(BS1, N1, 8)
            softmax_sum_8 = softmax_sum.broadcast_to(BS1, N1, 8)
            return output.cpu().to(ori_dtype), softmax_max_8.cpu(), softmax_sum_8.cpu()

        def softmax_torch(x):
            x_max = torch.max(x, dim=-1, keepdims=True)[0]
            x_sub = x.sub(x_max)
            y = torch.exp(x_sub)
            x_sum = y.sum(dim=-1, keepdims=True)
            ans = torch.softmax(x, dim=-1)
            return ans, x_max, x_sum

        def get_currentB_index(BS_index, actual_seq_len):
            for i in range(actual_seq_len.size(0)):
                if BS_index < actual_seq_len[i]:
                    return i
            raise RuntimeError(f"BS_index is greater than max(actual_seq_len).")

        return compute_select_attention(query, key, value, topk_indices, atten_mask, actual_seq_qlen, actual_seq_kvlen, scale_value, select_block_size, select_block_count)

    # pylint:disable = huawei-too-many-arguments
    def npu_op_exec(self, query, key, value, topk_indices, scale_value, head_num, select_block_size, select_block_count, atten_mask, actual_seq_qlen, actual_seq_kvlen):
        output = torch_npu.npu_nsa_select_attention(query, key, value, topk_indices, scale_value, head_num, select_block_size, select_block_count, atten_mask=None, actual_seq_qlen=actual_seq_qlen, actual_seq_kvlen=actual_seq_kvlen)
        return output

    @SupportedDevices(['Ascend910B'])
    def test_npu_nsa_select_attention(self):
        query = torch.randn([1, 115, 192], dtype=torch.float16)
        key = torch.randn([1088, 115, 192], dtype=torch.float16)
        value = torch.randn([1088, 115, 128], dtype=torch.float16)
        scale_value = 1.0
        head_num = query.size(1)
        select_block_count = 16
        select_block_size = 64
        actual_seq_qlen = [1]
        actual_seq_kvlen = [1088]

        topk_indices = torch.zeros(query.size(0), value.size(1), select_block_count,).to(torch.int)
        for i in range(query.size(0)):
            _, s1_idx = self.get_tnd_idx(actual_seq_qlen, i)
            for j in range(value.size(1)):
                if s1_idx < select_block_count * select_block_size:
                    topk_indices[i][j] = torch.arange(select_block_count)
                else:
                    topk_indices[i][j] = torch.randperm(select_block_count)
                    topk_indices[i][j][random.uniform(0, select_block_count)] = (s1_idx + select_block_size - 1) // select_block_size

        npuout = self.npu_op_exec(query.npu(), key.npu(), value.npu(), topk_indices.npu(), scale_value, head_num, select_block_size, select_block_count, None, actual_seq_qlen, actual_seq_kvlen)
        gloden_out = self.cpu_op_exec(query, key, value, topk_indices, scale_value, head_num, select_block_size, select_block_count, None, torch.tensor(actual_seq_qlen, dtype=torch.int64), torch.tensor(actual_seq_kvlen, dtype=torch.int64))

        self.assertRtolEqual(gloden_out[0].numpy(), npuout[0].cpu().numpy())
        self.assertRtolEqual(gloden_out[1].numpy(), npuout[1].cpu().numpy())
        self.assertRtolEqual(gloden_out[2].numpy(), npuout[2].cpu().numpy())

if __name__ == "__main__":
    run_tests()

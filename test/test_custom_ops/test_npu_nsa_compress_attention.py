# Copyright (c) 2020, Huawei Technologies.All rights reserved.
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
from einops import rearrange
import torch.nn.functional as F
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices


TOPK_CHECK_OUT_SUFFIX = 2


class TestNpuCompressAttention(TestCase):
    def get_top_k_mask(self, S1, S2):
        top_k_masks = torch.eye(S1, S2)
        top_k_masks[:, 0] = 1  # 首列默认选择
        top_k_masks = top_k_masks.to(torch.bool)
        return top_k_masks

    def get_cu_seqlens(self, seqlens_list):
        cu = torch.zeros(len(seqlens_list) + 1, dtype=torch.int32)
        for i in range(len(seqlens_list) + 1):
            cu[i] = sum(seqlens_list[:i])
        return cu

    def broadcastKV(self, n1, n2, kv_tensor, dtype):
        factor = n1 // n2
        kv_shape = kv_tensor.shape
        B = kv_shape[0]
        S = kv_shape[2]
        D = kv_shape[3]
        kv_res = torch.zeros([B, n1, S, D]).to(dtype)
        for i in range(n1):
            j = i // factor
            kv_res[:, i:i + 1, :, :] = kv_tensor[:, j:j + 1, :, :]
        return kv_res

    def tsoftmax(self, x):
        x_max = torch.max(x, dim=-1, keepdims=True)[0]
        x_sub = x.sub(x_max)
        y = torch.exp(x_sub)
        x_sum = y.sum(dim=-1, keepdims=True)
        ans = y.div(x_sum)
        return ans, x_max, x_sum

    def tforward(self, q, k, v, atten_mask, scale):
        qk = torch.matmul(q, k.permute(0, 1, 3, 2)).mul(scale)
        qk = qk + atten_mask.bool() * (-40000.0)
        softmax_res, x_max, x_sum = self.tsoftmax(qk)
        y = torch.matmul(softmax_res, v)
        return y, softmax_res, x_max, x_sum

    # pylint:disable = huawei-too-many-arguments
    def compute_impscore(self, softmax_res, topk_mask, selKVLen, m, n, N2, G):
        s1 = int(softmax_res.shape[-2])  # 1,n1,s1,s2
        s2 = int(softmax_res.shape[-1])  # 1,n1,s1,s2
        outerLoop = selKVLen
        innerLoop = m + n - 1
        trans = softmax_res.permute(0, 1, 3, 2)  # 1,n1,s2,s1
        sum_res = torch.zeros([int(softmax_res.shape[0]), int(softmax_res.shape[1]), outerLoop, s1])  # 1,n1,selKVLen,s1
        for i in range(outerLoop, 0, -1):
            src_idx = i * m
            dst_idx = i - 1
            for j in range(innerLoop):
                if j < innerLoop / 2:
                    times = j + 1
                else:
                    times = innerLoop - j
                times = min(times, n)
                if m * i - j >= s2 or m * i - j < 0:
                    continue
                sum_res[:, :, dst_idx] += trans[:, :, m * i - j] * times
        trans_back = sum_res.permute(0, 1, 3, 2)  # 1, n1, s1, selKVLen
        trans_back = trans_back.view(1, N2, G, s1, -1)
        trans_back = trans_back.sum(dim=2)  # 1, n2, s1, selKVLen
        if topk_mask is None:
            res = trans_back
        else:
            res = trans_back.masked_fill(topk_mask.to(torch.bool), value=23333.0)
        return res

    # pylint:disable = huawei-too-many-arguments
    def generate_golden(self, q, k, v, scale_value, head_num, compress_block_size, compress_stride, select_block_size,
                        select_block_count, topk_mask, atten_mask, actual_seq_qlen, actual_cmp_seq_kvlen, actual_sel_seq_kvlen):
        S1 = q.shape[0]
        N1 = q.shape[1]
        D2 = v.shape[2]
        N2 = v.shape[1]
        B = len(actual_cmp_seq_kvlen)
        m = select_block_size // compress_stride
        n = compress_block_size // compress_stride
        G = N1 // N2

        atten_out = torch.zeros(S1, N1, D2)
        topk_indices = torch.zeros(S1, N2, select_block_count)
        topk_indices_with_n = torch.zeros(S1, N2, select_block_count + TOPK_CHECK_OUT_SUFFIX)
        softmax_max = torch.empty(0)
        softmax_sum = torch.empty(0)

        cu_seqlens_q = self.get_cu_seqlens(actual_seq_qlen)
        cu_seqlens_k = self.get_cu_seqlens(actual_cmp_seq_kvlen)

        for i in range(B):
            qi = q[cu_seqlens_q[i]:cu_seqlens_q[i + 1]]
            ki = k[cu_seqlens_k[i]:cu_seqlens_k[i + 1]]
            vi = v[cu_seqlens_k[i]:cu_seqlens_k[i + 1]]
            qi = rearrange(qi, 's n d -> 1 n s d')
            ki = rearrange(ki, 's n d -> 1 n s d')
            vi = rearrange(vi, 's n d -> 1 n s d')

            if not (N1 == N2):
                ki = self.broadcastKV(N1, N2, ki, ki.dtype)
                vi = self.broadcastKV(N1, N2, vi, vi.dtype)

            cur_seq_q = actual_seq_qlen[i]
            cur_cmp_seq_k = actual_cmp_seq_kvlen[i]
            cur_sel_seq_k = actual_sel_seq_kvlen[i]
            atten_maski = torch.tensor([0]) if atten_mask is None else atten_mask[:(cur_seq_q), :(cur_cmp_seq_k)]


            outi_golden, softmax_resi, x_maxi, x_sumi = self.tforward(qi, ki, vi,
                                                                      atten_maski,
                                                                      scale_value)

            topk_maski = topk_mask if topk_mask is None else topk_mask[:(cur_seq_q), :(cur_sel_seq_k)]
            importanceScore_i = self.compute_impscore(softmax_resi, topk_maski,
                                                      actual_sel_seq_kvlen[i], m, n, N2, G)  # 1, n1, s1, s2 --> 1, n2, s1, selKVLen
            values, topki_total = torch.sort(importanceScore_i, descending=True, stable=True, dim=-1)
            topki = topki_total[:, :, :, :select_block_count]  # 1, n2, s1, select_count
            topki_with_n = topki_total[:, :, :, :select_block_count + TOPK_CHECK_OUT_SUFFIX]  # 1, n2, s1, select_count + TOPK_CHECK_OUT_SUFFIX

            atten_out[cu_seqlens_q[i]:cu_seqlens_q[i + 1]] = rearrange(outi_golden, '1 n s d -> s n d')
            topk_indices[cu_seqlens_q[i]:cu_seqlens_q[i + 1]] = rearrange(topki, '1 n s d -> s n d')  # s1, n2, select_count
            topk_indices_with_n[cu_seqlens_q[i]:cu_seqlens_q[i + 1]] = rearrange(topki_with_n, '1 n s d -> s n d')  # s1, n2, select_count + TOPK_CHECK_OUT_SUFFIX

            x_maxi = x_maxi.broadcast_to(1, N1, actual_seq_qlen[i], 8).contiguous().view(-1)
            x_sumi = x_sumi.broadcast_to(1, N1, actual_seq_qlen[i], 8).contiguous().view(-1)
            softmax_max = torch.cat([softmax_max, x_maxi], dim=0)
            softmax_sum = torch.cat([softmax_sum, x_sumi], dim=0)


        return atten_out, topk_indices.to(torch.int32), softmax_max, softmax_sum, topk_indices_with_n.to(torch.int32)

    # pylint:disable = huawei-too-many-arguments
    def cpu_op_exec(self, query, key, value, scale_value, head_num, compress_block_size, compress_stride, select_block_size, select_block_count, topk_mask, atten_mask, actual_seq_qlen, actual_cmp_seq_kvlen, actual_sel_seq_kvlen):
        output = self.generate_golden(query.to(torch.float32), key.to(torch.float32), value.to(torch.float32), scale_value, head_num, compress_block_size, compress_stride, select_block_size, select_block_count, topk_mask, atten_mask, actual_seq_qlen, actual_cmp_seq_kvlen, actual_sel_seq_kvlen)

        return output

    # pylint:disable = huawei-too-many-arguments
    def npu_op_exec(self, query, key, value, scale_value, head_num, compress_block_size, compress_stride, select_block_size, select_block_count, topk_mask, atten_mask, actual_seq_qlen, actual_cmp_seq_kvlen, actual_sel_seq_kvlen):
        output = torch_npu.npu_nsa_compress_attention(query, key, value, scale_value, head_num, compress_block_size, compress_stride, select_block_size, select_block_count, topk_mask=topk_mask, atten_mask=atten_mask, actual_seq_qlen=actual_seq_qlen, actual_cmp_seq_kvlen=actual_cmp_seq_kvlen, actual_sel_seq_kvlen=actual_sel_seq_kvlen)
        return output


    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_nsa_compress_attention(self):
        query = torch.randn([2633, 7, 8], dtype=torch.float16)
        key = torch.randn([4072, 7, 8], dtype=torch.float16)
        value = torch.randn([4072, 7, 7], dtype=torch.float16)
        scale_value = 0.3
        head_num = 7
        compress_block_size = 32
        compress_stride = 16
        select_block_size = 64
        select_block_count = 16

        topk_mask = self.get_top_k_mask(1384, 567).to(torch.uint8)

        atten_mask = np.triu(np.ones([1384, 2266]), k=1)
        atten_mask = torch.from_numpy(atten_mask).to(torch.uint8)
        cu_actual_seq_qlen = [1384, 1249]
        cu_actual_cmp_seq_kvlen = [2266, 1806]
        cu_actual_sel_seq_kvlen = [567, 452]
        actual_seq_qlen = [1384, 2633]
        actual_cmp_seq_kvlen = [2266, 4072]
        actual_sel_seq_kvlen = [567, 1019]


        npuout = self.npu_op_exec(query.npu(), key.npu(), value.npu(), scale_value, head_num, compress_block_size, compress_stride, select_block_size, select_block_count, topk_mask.npu(), atten_mask.npu(), actual_seq_qlen, actual_cmp_seq_kvlen, actual_sel_seq_kvlen)

        golden_out = self.cpu_op_exec(query.to(torch.float32), key.to(torch.float32), value.to(torch.float32), scale_value, head_num, compress_block_size, compress_stride, select_block_size, select_block_count, topk_mask, atten_mask, cu_actual_seq_qlen, cu_actual_cmp_seq_kvlen, cu_actual_sel_seq_kvlen)

        self.assertRtolEqual(golden_out[0].half().numpy(), npuout[0].cpu().numpy())

if __name__ == "__main__":
    run_tests()

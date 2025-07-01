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

import unittest
import numpy as np
import torch
from einops import rearrange
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices


def tsoftmax(x):
    x_max = torch.max(x, dim=-1, keepdims=True)[0]
    x_sub = x.sub(x_max)
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdims=True)
    softmax_res = y.div(x_sum)
    return softmax_res, x_max, x_sum


def simpleSoftmax(x, x_max, x_sum):
    x_sub = x.sub(x_max)
    y = torch.exp(x_sub)
    softmax_res = y.div(x_sum)
    return softmax_res


def tsoftmax_grad(p, dp, out, outGrad):
    muls = outGrad * out
    muls_res = muls.sum(dim=-1, keepdims=True)
    sub_res = dp - muls_res
    res = sub_res * p
    return res


def get_tnd_idx(actual_q_len, t_idx):
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


class NsaSelectedAttention:
    def __init__(self,
                 query,
                 key,
                 value,
                 out,
                 dy,
                 topkIndices,
                 attenMsk,
                 actualSeqLenQ,
                 actualSeqLenKV,
                 scaleValue=1.0,
                 blockCount=16,
                 blockSize=64,
                 sparseMode=0,
                ):
        self.query = query
        self.key = key
        self.value = value
        self.out = out
        self.dy = dy
        self.topkIndices = topkIndices
        self.attenMsk = attenMsk
        self.actualSeqLenQ = actualSeqLenQ
        self.actualSeqLenKV = actualSeqLenKV
        # param
        self.scaleValue = scaleValue
        self.blockCount = blockCount
        self.blockSize = blockSize
        self.sparseMode = sparseMode
        self.dtype = query.dtype

    def forward(self):
        query = self.query.float()
        key = self.key.float()
        topk_indices = self.topkIndices
        actual_q_len = self.actualSeqLenQ

        # param
        scaleValue = self.scaleValue
        selected_block_count = self.blockCount
        selected_block_size = self.blockSize
        select_s2 = selected_block_size * selected_block_count
        if self.sparseMode == 2:
            atten_enable = True
        else:
            atten_enable = False

        # shape
        T1, N1, D_qk = query.shape
        T2, N2, D_qk = key.shape
        G = N1 // N2
        S1 = max(self.actualSeqLenQ)
        S2 = max(self.actualSeqLenKV)
        B = len(self.actualSeqLenQ)
        # reshape
        query = query.reshape(T1, N2, G, D_qk)
        key = key.reshape(T2, N2, 1, D_qk)
        key = rearrange(key, 'b s n d ->  b n s d').reshape(B, N2, -1, selected_block_size, D_qk)

        x_max_out = torch.zeros(T1, N2, G, 1).to(torch.float)
        x_sum_out = torch.zeros(T1, N2, G, 1).to(torch.float)
        for i in range(T1):
            b_idx, s1_idx = get_tnd_idx(actual_q_len, i)

            for n2_idx in range(N2):
                topk = topk_indices[i][n2_idx]
                q_cal = query[i][n2_idx]
                k_cal = torch.index_select(key[b_idx][n2_idx], 0, topk).reshape(selected_block_count * selected_block_size, D_qk)

                # attenmask
                if atten_enable:
                    if s1_idx < select_s2:
                        atten_msk_cal = torch.ones(select_s2)
                        atten_msk_cal[0:s1_idx + 1] = 0
                    else:
                        atten_msk = torch.ones(S2)
                        atten_msk[0:select_s2 + 1] = 0
                        atten_msk = atten_msk.reshape(-1, selected_block_size)
                        atten_msk_cal = torch.index_select(atten_msk, 0, topk).reshape(select_s2)
                    atten_msk_cal = atten_msk_cal.repeat(G, 1)
                qk = torch.matmul(q_cal, k_cal.permute(1, 0)).mul(scaleValue)
                if atten_enable:
                    qk = qk + atten_msk_cal * (-2e35)
                _, x_max, x_sum = tsoftmax(qk)
                x_max_out[i][n2_idx] = x_max
                x_sum_out[i][n2_idx] = x_sum

        x_max_out = x_max_out.expand(T1, N2, G, 8)
        x_sum_out = x_sum_out.expand(T1, N2, G, 8)

        self.softmaxMax = x_max_out.reshape(T1, N2 * G, 8).float()
        self.softmaxSum = x_sum_out.reshape(T1, N2 * G, 8).float()

        return self.out, self.softmaxMax, self.softmaxSum


    def backward(self):
        query = self.query.float()
        key = self.key.float()
        value = self.value.float()
        out = self.out.float()
        dy = self.dy.float()
        softmaxMax = self.softmaxMax.float()
        softmaxSum = self.softmaxSum.float()
        topk_indices = self.topkIndices
        actual_q_len = self.actualSeqLenQ
        actual_kv_len = self.actualSeqLenKV

        # param
        scaleValue = self.scaleValue
        selected_block_count = self.blockCount
        selected_block_size = self.blockSize
        select_s2 = selected_block_size * selected_block_count
        if self.sparseMode == 2:
            atten_enable = True
        else:
            atten_enable = False

        # shape
        T1, N1, D_qk = query.shape
        T2, N2, D_v = value.shape
        G = N1 // N2
        S1 = max(self.actualSeqLenQ)
        S2 = max(self.actualSeqLenKV)
        B = len(self.actualSeqLenQ)
        # reshape
        query = query.reshape(T1, N2, G, D_qk)
        key = key.reshape(T2, N2, 1, D_qk)
        value = value.reshape(T2, N2, 1, D_v)
        out = out.reshape(T1, N2, G, D_v)
        dy = dy.reshape(T1, N2, G, D_v)
        softmaxMax = softmaxMax.reshape(T1, N2, G, 8)
        softmaxSum = softmaxSum.reshape(T1, N2, G, 8)


        dq_out = torch.zeros(T1, N2, G, D_qk).to(torch.float)
        x_max_out = torch.zeros(T1, N2, G, 1).to(torch.float)
        x_sum_out = torch.zeros(T1, N2, G, 1).to(torch.float)
        dk_out = torch.zeros(B, N2, S2, D_qk).reshape(B, N2, -1, selected_block_size, D_qk).to(torch.float)
        dv_out = torch.zeros(B, N2, S2, D_v).reshape(B, N2, -1, selected_block_size, D_v).to(torch.float)

        k_tmp = key.reshape(B, S2, N2, D_qk)
        v_tmp = value.reshape(B, S2, N2, D_v)
        k_tmp = rearrange(k_tmp, 'b s n d ->  b n s d').reshape(B, N2, -1, selected_block_size, D_qk)
        v_tmp = rearrange(v_tmp, 'b s n d ->  b n s d').reshape(B, N2, -1, selected_block_size, D_v)

        for i in range(T1):
            b_idx, s1_idx = get_tnd_idx(actual_q_len, i)

            for n2_idx in range(N2):
                # gather
                topk = topk_indices[i][n2_idx]
                q_cal = query[i][n2_idx]
                out_cal = out[i][n2_idx]
                dy_cal = dy[i][n2_idx]
                k_cal = torch.index_select(k_tmp[b_idx][n2_idx], 0, topk).reshape(selected_block_count * selected_block_size, D_qk)
                v_cal = torch.index_select(v_tmp[b_idx][n2_idx], 0, topk).reshape(selected_block_count * selected_block_size, D_v)

                if atten_enable:
                    if s1_idx < select_s2:
                        atten_msk_cal = torch.ones(select_s2)
                        atten_msk_cal[0:s1_idx + 1] = 0
                    else:
                        atten_msk = torch.ones(S2)
                        atten_msk[0:select_s2 + 1] = 0
                        atten_msk = atten_msk.reshape(-1, selected_block_size)
                        atten_msk_cal = torch.index_select(atten_msk, 0, topk).reshape(select_s2)
                    atten_msk_cal = atten_msk_cal.repeat(G, 1)
                # fag cal
                qk = torch.matmul(q_cal, k_cal.permute(1, 0)).mul(scaleValue)
                if atten_enable:
                    qk = qk + atten_msk_cal * (-2e35)

                x_max = softmaxMax[i][n2_idx][:, [0]].reshape(-1, 1)
                x_sum = softmaxSum[i][n2_idx][:, [0]].reshape(-1, 1)
                softmax_res = simpleSoftmax(qk, x_max, x_sum)

                dp = torch.matmul(dy_cal, v_cal.permute(1, 0))
                softmax_grad_res = (tsoftmax_grad(softmax_res, dp, out_cal, dy_cal))
                dq = torch.matmul(softmax_grad_res, k_cal)
                dk = torch.matmul(softmax_grad_res.permute(1, 0), q_cal)
                dv = torch.matmul(softmax_res.permute(1, 0), dy_cal)
                dk = dk.reshape(selected_block_count, selected_block_size, D_qk)
                dv = dv.reshape(selected_block_count, selected_block_size, D_v)

                #scatter
                dq_out[i][n2_idx] = dq
                x_max_out[i][n2_idx] = x_max
                x_sum_out[i][n2_idx] = x_sum
                for kk in range(selected_block_count):
                    dk_out[b_idx][n2_idx][topk[kk]] += dk[kk]
                    dv_out[b_idx][n2_idx][topk[kk]] += dv[kk]


        dq_out = dq_out * scaleValue
        dk_out = dk_out * scaleValue

        dk_out = dk_out.reshape(B, N2, S2, D_qk)
        dv_out = dv_out.reshape(B, N2, S2, D_v)
        dk_out = rearrange(dk_out, 'b n s d ->  b s n d')
        dv_out = rearrange(dv_out, 'b n s d ->  b s n d')

        dq_out = dq_out.reshape(T1, N1, D_qk)
        dk_out = dk_out.reshape(T2, N2, D_qk)
        dv_out = dv_out.reshape(T2, N2, D_v)

        x_max_out = x_max_out.expand(T1, N2, G, 8).reshape(T1, N2 * G, 8)
        x_sum_out = x_sum_out.expand(T1, N2, G, 8).reshape(T1, N2 * G, 8)

        return dq_out, dk_out, dv_out


class TestNpuCompress(TestCase):

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_nsa_select_attention_grad(self):
        query = torch.randn([1, 1, 192], dtype=torch.bfloat16)
        key = torch.randn([1024, 1, 192], dtype=torch.bfloat16)
        value = torch.randn([1024, 1, 192], dtype=torch.bfloat16)
        attention_out = torch.randn([1, 1, 192], dtype=torch.bfloat16)
        grad = torch.randn([1, 1, 192], dtype=torch.bfloat16)
        actual_seq_qlen = [1]
        actual_seq_kvlen = [1024]
        select_block_size = 64
        select_block_count = 16
        sparse_mode = 2
        scale_value = 0.01
        head_num = query.shape[1]
        atten_mask = None

        BS1, N1, QKD = query.shape
        BS2, N2, VD = value.shape
        G = N1 // N2
        N1 = N2 * G

        if sparse_mode == 2:
            topk_indices = torch.zeros(BS1, N2, select_block_count,).to(torch.int)
            for i in range(BS1):
                _, s1_idx = get_tnd_idx(actual_seq_qlen, i)
                for j in range(N2):
                    if s1_idx < select_block_count * select_block_size:
                        topk_indices[i][j] = torch.arange(select_block_count)
                    else:
                        topk_indices[i][j] = torch.randperm(select_block_count)
                        topk_indices[i][j][random.uniform(0, select_block_count)] = (s1_idx + select_block_size - 1) // select_block_size
        else:
            topk_indices = torch.randint(low=0, high=S2 // select_block_size, size=(T1, N2, select_block_count,))

        select_attention = NsaSelectedAttention(
            query=query,
            key=key,
            value=value,
            out=attention_out,
            dy=grad,
            topkIndices=topk_indices,
            attenMsk=None,
            actualSeqLenQ=actual_seq_qlen,
            actualSeqLenKV=actual_seq_kvlen,
            scaleValue=scale_value,
            blockSize=select_block_size,
            blockCount=select_block_count,
            sparseMode=sparse_mode
        )

        attention_out, softmax_max, softmax_sum = select_attention.forward()
        golden = select_attention.backward()

        npuout = torch_npu.npu_nsa_select_attention_grad(grad.npu(), query.npu(), key.npu(), value.npu(), attention_out.npu(), softmax_max.npu(), softmax_sum.npu(), topk_indices.npu(), scale_value, head_num, select_block_size, select_block_count, atten_mask=atten_mask, actual_seq_qlen=actual_seq_qlen, actual_seq_kvlen=actual_seq_kvlen)

        self.assertRtolEqual(golden[0].float().numpy(), npuout[0].float().cpu().numpy(), 0.004, 0.004)
        self.assertRtolEqual(golden[1].float().numpy(), npuout[1].float().cpu().numpy(), 0.004, 0.004)
        self.assertRtolEqual(golden[2].float().numpy(), npuout[2].float().cpu().numpy(), 0.004, 0.004)

if __name__ == "__main__":
    run_tests()

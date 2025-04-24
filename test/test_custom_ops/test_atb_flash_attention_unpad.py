import math
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestFaUnPad(TestCase):
    def gen_seq_len(self, batch, max_seq, variate_seq=False):
        if variate_seq:
            num = max_seq // 16
            seqlen_aligned_arange = np.arange(1, num) * 16
            if batch > num:
                seqlen_aligned_remain = np.random.randint(1, max_seq, size=(batch - num))
                seqlen_aligned_remain[:] = ((seqlen_aligned_remain[:] + 15) // 16) * 16
                seqlen_aligned = np.concatenate((seqlen_aligned_arange, seqlen_aligned_remain), 0)
            else:
                seqlen_aligned = seqlen_aligned_arange
            sp_list = np.random.randint(0, 15, size=(num - 1))
            seqlen = seqlen_aligned - sp_list
            seqlen = seqlen[-batch:]
            seqlen_aligned = seqlen_aligned[-batch:]
        else:
            max_seq_aligned = (max_seq + 15) // 16 * 16
            sp_list = np.ones((batch,)) * (max_seq_aligned - max_seq)
            sp_list = sp_list.astype(np.int32)
            seqlen = np.ones((batch,)) * max_seq
            seqlen = seqlen.astype(np.int32)
            seqlen_aligned = np.ones((batch,)) * max_seq_aligned
            seqlen_aligned = seqlen_aligned.astype(np.int32)

        ntokens = seqlen.sum()
        return seqlen, seqlen_aligned, ntokens

    def group_matmul(self, heads, group_num, A, B):
        group_head = heads // group_num
        score = None
        for i in range(group_num):
            group_score = np.matmul(A[i * group_head: (i + 1) * group_head, :, :].astype(np.float32),
                                    B[i:(i + 1), :, :].astype(np.float32)).astype(np.float16)
            if score is None:
                score = group_score
            else:
                score = np.concatenate((score, group_score), 0)
        return score

    def calc_expect_func(self, batch, seqlen, heads, embed, group_num=32):
        variate_seq = False
        is_decoder = False
        max_seq = 2048
        src_type = 'float16'
        fp32 = True
        if is_decoder:
            q_seqlen, q_seqlen_aligned, q_ntokens = self.gen_seq_len(batch, 1, variate_seq)
            kv_seqlen, kv_seqlen_aligned, kv_ntokens = self.gen_seq_len(batch, seqlen, variate_seq)
        else:
            q_seqlen, q_seqlen_aligned, q_ntokens = self.gen_seq_len(batch, seqlen, variate_seq)
            kv_seqlen, kv_seqlen_aligned, kv_ntokens = q_seqlen, q_seqlen_aligned, q_ntokens

        max_s = np.max(q_seqlen)
        ntokens2 = (q_seqlen * kv_seqlen).sum()
        embed_v = np.random.randint(1, embed)

        q = np.random.uniform(-1.0, 1.0, size=(q_ntokens, heads * embed)).astype(np.float16)
        k = np.random.uniform(-1.0, 1.0, size=(kv_ntokens, group_num * embed)).astype(np.float16)
        v = np.random.uniform(-1.0, 1.0, size=(kv_ntokens, group_num * embed_v)).astype(np.float16)

        q_offset = 0
        k_offset = 0
        v_offset = 0

        s = None
        _p = None
        out = None

        for idx in range(batch):
            q_s = q_seqlen[idx]
            kv_s = kv_seqlen[idx]
            q_slice = q[q_offset:q_offset + q_s][:]
            q_slice = q_slice.reshape(q_s, heads, embed)
            q_slice = np.transpose(q_slice, (1, 0, 2))
            k_slice = k[k_offset:k_offset + kv_s][:]
            k_slice = k_slice.reshape(kv_s, group_num, embed)
            k_slice = np.transpose(k_slice, (1, 0, 2))
            k_slice_t = np.transpose(k_slice, (0, 2, 1))
            v_slice = v[v_offset:v_offset + kv_s][:]
            v_slice = v_slice.reshape(kv_s, group_num, embed_v)
            v_slice = np.transpose(v_slice, (1, 0, 2))
            score = self.group_matmul(heads, group_num, q_slice, k_slice_t)
            if s is None:
                s = score.reshape([-1, ])
            else:
                s = np.concatenate((s, score.reshape([-1, ])), 0)

            tor = np.float16(1.0 / math.sqrt(1.0 * embed))
            score = score * tor
            score_max = np.max(score, axis=-1)
            score = score - score_max.reshape((heads, q_s, 1))
            score_exp = np.exp(score.astype(np.float32))
            if not fp32:
                score_sum = np.sum(score_exp.astype(np.float16), axis=-1)
                if _p is None:
                    _p = score_exp.astype(np.float16).reshape([-1, ])
                else:
                    _p = np.concatenate((_p, score_exp.astype(np.float16).reshape([-1, ])), 0)
                p = score_exp.astype(np.float16) / score_sum.reshape((heads, q_s, 1)).astype(np.float16)
                out_sub = self.group_matmul(heads, group_num, p, v_slice)
            else:
                score_sum = np.sum(score_exp, axis=-1)
                if _p is None:
                    _p = score_exp.astype(np.float16).reshape([-1, ])
                else:
                    _p = np.concatenate((_p, score_exp.astype(np.float16).reshape([-1, ])), 0)
                p = score_exp.astype(np.float16)
                out_sub = self.group_matmul(heads, group_num, p, v_slice)
                out_sub = out_sub / score_sum.reshape((heads, q_s, 1)).astype(np.float16)

            out_sub = out_sub.reshape(heads, q_s, embed_v)
            out_sub = np.transpose(out_sub, (1, 0, 2))
            out_sub = np.ascontiguousarray(out_sub)
            if out is None:
                out = out_sub
            else:
                out = np.concatenate((out, out_sub), 0)

            q_offset += q_s
            k_offset += kv_s
            v_offset += kv_s

        q = q.astype(src_type).reshape(-1, heads, embed)
        k = k.astype(src_type).reshape(-1, group_num, embed)
        v = v.astype(src_type).reshape(-1, group_num, embed_v)
        q_len = q_seqlen.astype(np.int32)
        out_expect = out.astype(src_type).reshape(-1, heads, embed_v)

        ret_data = q, k, v, q_len, tor, heads, out_expect
        return ret_data

    @SupportedDevices(['Ascend910B'])
    def test_flash_attention_unpad(self):
        kv_head = 32
        data = self.calc_expect_func(16, 128, 32, 128, group_num=kv_head)
        param_seqlen = data[4].tolist()

        # 检查每个元素是否为 numpy 数组，如果是标量则转换为数组
        in_tensors = []
        for tensor in data:
            if isinstance(tensor, np.ndarray):
                in_tensors.append(torch.from_numpy(tensor))
            else:
                in_tensors.append(torch.tensor(tensor))

        in_tensors = [tensor.npu() for tensor in in_tensors]
        query = in_tensors[0]
        key = in_tensors[1]
        value = in_tensors[2]
        seq_len = in_tensors[3].cpu()
        tor = data[4]
        heads = data[5]
        group_num = kv_head
        cal_out = in_tensors[6]
        out = torch.empty_like(in_tensors[6]).npu()
        torch_npu._npu_flash_attention_unpad(query, key, value, seq_len, tor, heads, group_num, out)
        self.assertRtolEqual(cal_out, out)

if __name__ == '__main__':
    run_tests()

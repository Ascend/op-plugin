import unittest
import torch
import numpy as np
import torch.nn as nn
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestDenseLightningIndexerSoftmaxLse(TestCase):
    def batch_matmul_gqa(self, q_head_num, k_head_num, q, k):
        g_size = q_head_num // k_head_num
        score = None
        for i in range(k_head_num):
            score_tmp = torch.matmul(q[i*g_size:(i+1)*g_size, :, :].to(torch.float32), k[i:i+1, :, :].to(torch.float32))
            score = torch.cat((score, score_tmp), 0) if score is not None else score_tmp
        return score
    
    def _dense_lightning_indexer_softmax_lse(self, query_index, key_index, weights, actual_seq_lengths_query, actual_seq_lengths_key, layout, sparse_mode=3):
        if layout == 'BSND':
            B, q_seq_len, q_head_num, head_dim = query_index.shape
            _, k_seq_len, k_head_num, _ = key_index.shape   #k_head_num=1
        elif layout == 'TND':
            B = len(actual_seq_lengths_query)
            _, q_head_num, head_dim = query_index.shape
            _, k_head_num, _ = key_index.shape
            q_seq_len = actual_seq_lengths_query[0] # each batch has same length
        else:
            print(f"not support layout {layout}, only support BSND and TND")
            return False

        # return results
        softmax_max_out = torch.zeros(B, k_head_num, q_seq_len, dtype=torch.float32)
        softmax_sum_out = torch.zeros(B, k_head_num, q_seq_len, dtype=torch.float32)

        for i in range(B):
            if layout == 'BSND':
                q_seq_len_1batch = q_seq_len
                k_seq_len_1batch = k_seq_len
                q_index_1batch = query_index[i]
                k_index_1batch = key_index[i]
                w = weights[i] 
            else: # TND
                if i == 0:
                    q_seq_len_1batch = actual_seq_lengths_query[i]
                    k_seq_len_1batch = actual_seq_lengths_key[i]
                    q_index_1batch = query_index[0:actual_seq_lengths_query[0], :, :]
                    k_index_1batch = key_index[0:actual_seq_lengths_key[0], :, :]
                    w = weights[0:actual_seq_lengths_query[0], :]
                else:
                    q_seq_len_1batch = actual_seq_lengths_query[i] - actual_seq_lengths_query[i-1]
                    k_seq_len_1batch = actual_seq_lengths_key[i] - actual_seq_lengths_key[i-1]
                    q_index_1batch = query_index[actual_seq_lengths_query[i-1]:actual_seq_lengths_query[i], :, :]
                    k_index_1batch = key_index[actual_seq_lengths_key[i-1]:actual_seq_lengths_key[i], :, :]
                    w = weights[actual_seq_lengths_query[i-1]:actual_seq_lengths_query[i], :]

            q_index_1batch = torch.permute(q_index_1batch, (1, 0, 2)) #(s1, n1, d) -> (n1, s1, d)
            k_index_1batch = torch.permute(k_index_1batch, (1, 2, 0)) #(s2, n2, d) -> (n2, d, s2)
            score = self.batch_matmul_gqa(q_head_num, k_head_num, q_index_1batch, k_index_1batch)  #(n1, s1, s2)
            score = torch.nn.functional.relu(score)

            score = torch.permute(score, (2, 1, 0)) #(s2, s1, k_head_num*g_size)
            p = torch.mul(score, w.to(torch.float32))
            p = torch.permute(p, (1, 0, 2)) #(s1, s2, k_head_num*g_size)
            p_reduce_sum = torch.sum(p, dim=2) #(s1, s2)

            if sparse_mode == 3:
                for s1_idx in range(q_seq_len_1batch):
                    p_reduce_sum[s1_idx, (k_seq_len_1batch - q_seq_len_1batch + s1_idx + 1):] = float('-inf')

            max_out, _ = torch.max(p_reduce_sum, dim=-1, keepdim=False)
            max_res, _ = torch.max(p_reduce_sum, dim=-1, keepdim=True)
            sub_res = p_reduce_sum - max_res
            exp_res = torch.exp(sub_res)
            sum_out = torch.sum(exp_res, dim=-1, keepdim=False)
            softmax_max_out[i, :, :] = max_out
            softmax_sum_out[i, :, :] = sum_out

        return softmax_max_out.reshape(-1, B*k_head_num*q_seq_len_1batch), softmax_sum_out.reshape(-1, B*k_head_num*q_seq_len_1batch)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    def test_bsnd_dense_lightning_indexer_softmax_lse_eager(self):
        b = 20
        s1 = 512
        n1 = 32
        s2 = 2048
        n2 = 1
        d = 128
        layout = 'BSND'

        np.random.seed(0)
        query_index = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, d))).to(torch.bfloat16)
        key_index = torch.tensor(np.random.uniform(-10, 10, (b, s2, n2, d))).to(torch.bfloat16)
        weights = torch.tensor(np.random.uniform(-1, 1, (b, s1, n1))).to(torch.bfloat16)
        actual_seq_lengths_query = None
        actual_seq_lengths_key = None
        sparse_mode = 3
        cpu_out, cpu_out1 = self._dense_lightning_indexer_softmax_lse(query_index, key_index, weights, actual_seq_lengths_query, actual_seq_lengths_key, layout, sparse_mode)

        query_index = query_index.npu()
        key_index = key_index.npu()
        weights = weights.npu()
        npu_out, npu_out1 = torch_npu.npu_dense_lightning_indexer_softmax_lse(query_index, key_index, weights, 
                                                                              actual_seq_qlen=actual_seq_lengths_query, 
                                                                              actual_seq_klen=actual_seq_lengths_key, 
                                                                              layout=layout,
                                                                              sparse_mode=sparse_mode)

        # compare result
        npu_out = npu_out.reshape(-1, b*n2*s1).cpu()
        cpu_out = cpu_out.reshape(-1, b*n2*s1).cpu()
        npu_out1 = npu_out1.reshape(-1, b*n2*s1).cpu()
        cpu_out1 = cpu_out1.reshape(-1, b*n2*s1).cpu()
        self.assertRtolEqual(npu_out, cpu_out, prec=1.e-3)
        self.assertRtolEqual(npu_out1, cpu_out1, prec=1.e-3)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    def test_tnd_dense_lightning_indexer_softmax_lse_eager(self):
        b = 2
        s1 = 512
        n1 = 32
        s2 = 2048
        n2 = 1
        d = 128
        layout = 'TND'

        np.random.seed(3)
        query_index = torch.tensor(np.random.uniform(-10, 10, (b*s1, n1, d))).to(torch.bfloat16)
        key_index = torch.tensor(np.random.uniform(-10, 10, (b*s2, n2, d))).to(torch.bfloat16)
        weights = torch.tensor(np.random.uniform(-1, 1, (b*s1, n1))).to(torch.bfloat16)
        actual_seq_lengths_query = torch.tensor([s1*(i+1) for i in range(b)]).to(torch.int32)
        actual_seq_lengths_key = torch.tensor([s2*(i+1) for i in range(b)]).to(torch.int32)
        sparse_mode = 3
        cpu_out, cpu_out1 = self._dense_lightning_indexer_softmax_lse(query_index, key_index, weights, actual_seq_lengths_query, actual_seq_lengths_key, layout, sparse_mode)

        query_index = query_index.npu()
        key_index = key_index.npu()
        weights = weights.npu()
        actual_seq_lengths_query = actual_seq_lengths_query.npu()
        actual_seq_lengths_key = actual_seq_lengths_key.npu()
        npu_out, npu_out1 = torch_npu.npu_dense_lightning_indexer_softmax_lse(query_index, key_index, weights, 
                                                                              actual_seq_qlen=actual_seq_lengths_query, 
                                                                              actual_seq_klen=actual_seq_lengths_key, 
                                                                              layout=layout,
                                                                              sparse_mode=sparse_mode)

        # compare result
        npu_out = npu_out.reshape(-1, b*n2*s1).cpu()
        cpu_out = cpu_out.reshape(-1, b*n2*s1).cpu()
        npu_out1 = npu_out1.reshape(-1, b*n2*s1).cpu()
        cpu_out1 = cpu_out1.reshape(-1, b*n2*s1).cpu()
        self.assertRtolEqual(npu_out, cpu_out, prec=1.e-3)
        self.assertRtolEqual(npu_out1, cpu_out1, prec=1.e-3)


if __name__ == "__main__":
    run_tests()
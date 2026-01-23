import unittest
import torch
import numpy as np
import torch.nn as nn
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestDenseLightningIndexerGradKLLoss(TestCase):

    def _deal_sparse_mask(self, input_matrix):
        assert len(input_matrix.shape) == 4

        _, _, s1, s2 = input_matrix.shape
        assert s1 <= s2

        res_matrix = input_matrix
        for s1_idx in range(s1):
            res_matrix[:, :, s1_idx, (s2 - s1 + s1_idx + 1):] = -(torch.inf)

        return res_matrix

    def _process_p(self, input_query, input_key, input_softmax_max, input_softmax_sum, scale):
        # Q: [B,S1,N1,D] -> [B,N,S1,D]
        _, _, q_head_num, _  = input_query.shape
        query = input_query.permute(0, 2, 1, 3).contiguous()
        # K: [B,S2,N2,D] -> [B,N2,D,S2]
        key = input_key.permute(0, 2, 3, 1).contiguous()
        # batch matmul
        p_tmp = torch.matmul(query.to(torch.float32), key.to(torch.float32))
        # scale
        p_tmp *= scale

        # sparse mask
        p_sparse = self._deal_sparse_mask(p_tmp)
        # simple softmax
        softmax_max_res = input_softmax_max.permute(0, 1, 3, 2).contiguous().reshape(p_sparse.shape[0], -1, p_sparse.shape[2])
        p_diff = p_sparse - softmax_max_res.unsqueeze(-1)
        p_exp = torch.exp(p_diff)
        softmax_sum_res = input_softmax_sum.permute(0, 1, 3, 2).contiguous().reshape(p_exp.shape[0], -1, p_exp.shape[2])
        p_div = p_exp.div(softmax_sum_res.unsqueeze(-1))
        # reduce sum: (B,N1,S1,S2) -> (B,S1,S2)
        p_reduce = p_div.sum(axis=1, keepdims=False)
        # scale
        p_reduce *= (1 / q_head_num)

        return p_reduce, softmax_max_res, softmax_sum_res
    
    def _process_sy(self, input_query_index, input_key_index, input_weight, input_softmax_max_index, input_softmax_sum_index):
        _, _, N1, D = input_query_index.shape
        _, _, N2, _ = input_key_index.shape
        group_size_index = N1 // N2
        # Q_INDEX: [B,S1,N1,D] -> [B,N1,S1,D]
        query_index = input_query_index.permute(0, 2, 1, 3).contiguous()
        # K_INDEX: [B,S2,N2,D] -> [B,N2*G,D,S2]
        key_index = input_key_index.permute(0, 2, 3, 1).contiguous().repeat(1, group_size_index, 1, 1)
        # batch matmul
        s_tmp = torch.matmul(query_index.to(torch.float32), key_index.to(torch.float32))
        # relu
        s_relu = torch.relu(s_tmp)
        # WEIGHT: [B,S1,N1] -> [B,N1,S1,1]
        weight = input_weight.permute(0, 2, 1).contiguous().unsqueeze(-1)
        # Mul
        s_mul = s_relu * weight.to(torch.float32)
        # reduce
        s_reduce = s_mul.sum(axis=1, keepdims=True)
        # sparse mask
        s_sparse = self._deal_sparse_mask(s_reduce).squeeze(1)
        # simple softmax
        softmax_max_index_res = input_softmax_max_index.squeeze(1)
        s_diff = s_sparse - softmax_max_index_res.unsqueeze(-1)
        s_exp = torch.exp(s_diff)
        softmax_sum_index_res = input_softmax_sum_index.squeeze(1)
        s_div = s_exp.div(softmax_sum_index_res.unsqueeze(-1))

        return s_div, s_relu, softmax_max_index_res, softmax_sum_index_res
    
    def _process_kl_loss(self, p_result, sy_result):
        # clip
        min_value = torch.tensor([1e-8])
        p_result_clip = torch.max(p_result, min_value)
        sy_result_clip = torch.max(sy_result, min_value)
        # log
        p_log = torch.log(p_result_clip)
        sy_log = torch.log(sy_result_clip)
        # sub
        sub_result = p_log - sy_log
        # mul
        mul_result = sub_result * p_result
        # loss
        loss = torch.sum(mul_result)

        return loss


    def _process_dwqk(self, p_result, sy_result, relu_res, input_query_index, input_key_index, input_weight, q_dtype):
        
        _, _, N1, D = input_query_index.shape
        _, _, N2, _ = input_key_index.shape
        group_size = N1 // N2
        # sub
        sub_result = sy_result - p_result
        # mul: (B,S1,1,S2) * (B,S1,N1,S2)
        mul_relu = sub_result.unsqueeze(2) * relu_res.permute(0, 2, 1, 3).contiguous()
        # reduce: [B,S1,N1,S2] -> [B,S1,N1]
        d_weight = mul_relu.sum(axis=-1, keepdims=False).to(q_dtype)

        # WEIGHT: [B,S1,N1] -> [B,N1,S1,1]
        weight = input_weight.permute(0, 2, 1).contiguous().unsqueeze(-1).to(torch.float32)
        # mul: (B,1,S1,S2) * (B,N1,S1,1) = (B,N1,S1,S2)
        mul_weight = sub_result.unsqueeze(1) * weight
        # relu grad: (x > 0) = 1; (x <= 0) = 0
        relu_grad = mul_weight * (relu_res > 0).float()
        # cast for matmul
        relu_grad = relu_grad.to(q_dtype)

        # KEY_INDEX: [B,S2,N2,D] -> [B,N1,S2,D]
        key_index = input_key_index.permute(0, 2, 1, 3).contiguous().repeat(1, group_size, 1, 1)
        # batch matmul (B,N1,S1,S2) @ (B,N1,S2,D) = (B,N1,S1,D)
        d_query_index = torch.matmul(relu_grad.to(torch.float32), key_index.to(torch.float32))
        d_query_index = d_query_index.to(q_dtype)
        # D_QUERY_INDEX: [B,N1,S1,D] -> [B,S1,N1,D]
        d_query_index = d_query_index.permute(0, 2, 1, 3).contiguous()

        # QUERY_INDEX: [B,S1,N1,D] -> [B,N1,S1,D]
        query_index = input_query_index.permute(0,2,1,3).contiguous()
        # RELU_GRAD: [B,N1,S1,S2] -> [B,N1,S2,S1]
        relu_grad = relu_grad.permute(0, 1, 3, 2).contiguous()
        # batch matmul: (B,N1,S2,S1) @ (B,N1,S1,D) = (B,N1,S2,D)
        d_key_index_tmp = torch.matmul(relu_grad.to(torch.float32), query_index.to(torch.float32))
        # reshape: [B,N1,S2,D] -> [B,G,N2,S2,D]
        d_key_index = d_key_index_tmp.reshape(-1, group_size, N2, d_key_index_tmp.shape[2], D)
        # reduce on g_size: [B,G,N2,S2,D] -> [B,N2,S2,D]
        d_key_index = d_key_index.sum(axis=1, keepdims=False)
        # permute: [B,N2,S2,D] -> [B,S2,N2,D]
        d_key_index = d_key_index.permute(0, 2, 1, 3).contiguous().to(q_dtype)

        return d_weight, d_query_index, d_key_index



    def _dense_lightning_indexer_grad_kl_loss(self, query, key, query_index, key_index, weights, softmax_max, softmax_sum, softmax_max_index, softmax_sum_index, scale, query_rope, key_rope):
        
        query = torch.cat((query, query_rope), dim=-1)
        key = torch.cat((key, key_rope), dim=-1)
        # process P
        p_result, softmax_max_result, softmax_sum_result = self._process_p(query, key, softmax_max, softmax_sum, scale)

        # process S'Y
        sy_result, relu_res, softmax_max_index_result, softmax_sum_index_result = self._process_sy(query_index, key_index, weights, softmax_max_index, softmax_sum_index)
        # process kl loss
        loss = self._process_kl_loss(p_result, sy_result)
        # process dw/dq/dk
        q_dtype = query.dtype
        d_weight, d_query_index, d_key_index = self._process_dwqk(p_result, sy_result, relu_res, query_index, key_index, weights, q_dtype)

        return d_query_index, d_key_index, d_weight, loss
    
    def _get_input(self, layout="BSND"):
        torch.manual_seed(0)
        np.random.seed(0)

        q_dtype = torch.float16
        B, N1, N2, N1_index, N2_index, S1, S2, D, Dr = 1, 64, 64, 64, 1, 128, 256, 128, 64
        query = torch.randn(B, S1, N1, D, dtype=q_dtype)
        key = torch.randn(B, S2, N2, D, dtype=q_dtype)
        query_index = torch.randn(B, S1, N1_index, D, dtype=q_dtype)
        key_index = torch.randn(B, S2, N2_index, D, dtype=q_dtype)
        query_rope = torch.randn(B, S1, N1, Dr, dtype=q_dtype)
        key_rope = torch.randn(B, S2, N2, Dr, dtype=q_dtype)
        weights = torch.randn(B, S1, N1_index, dtype=q_dtype)
        softmax_max = (torch.randn(B, N2, S1, 1, dtype=torch.float32).abs() + 0.4) * D  # N1=N2
        softmax_sum = torch.ones(B, N2, S1, 1, dtype=torch.float32)
        softmax_max_index = (torch.randn(B, 1, S1, dtype=torch.float32).abs() + 0.4) * D * N1_index  
        softmax_sum_index = torch.ones(B, 1, S1, dtype=torch.float32)
        actual_seq_qlen = [S1]
        actual_seq_klen = [S2]
        input_list = [query, key, query_index, key_index, weights, softmax_max, softmax_sum, softmax_max_index, softmax_sum_index, query_rope, key_rope, actual_seq_qlen, actual_seq_klen]

        return input_list


    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    def test_dense_lightning_indexer_grad_kl_loss_eager(self):
        sparse_mode = 3
        scale = 1.0
        layout = 'BSND'
        input_list = self._get_input(layout)
        query, key, query_index, key_index, weights, softmax_max, softmax_sum, softmax_max_index, softmax_sum_index, query_rope, key_rope, actual_seq_qlen, actual_seq_klen = input_list
        cpu_out = self._dense_lightning_indexer_grad_kl_loss(query, key, query_index, key_index, weights, softmax_max, softmax_sum, softmax_max_index, 
                                                            softmax_sum_index, scale, query_rope, key_rope)

        for i in range(len(input_list)):
            try:
                input_list[i] = input_list[i].npu()
            except:
                continue
        query, key, query_index, key_index, weights, softmax_max, softmax_sum, softmax_max_index, softmax_sum_index, query_rope, key_rope, actual_seq_qlen, actual_seq_klen = input_list

        npu_out = torch_npu.npu_dense_lightning_indexer_grad_kl_loss(query, key, query_index, key_index, weights, softmax_max, softmax_sum, softmax_max_index, softmax_sum_index, scale,
        query_rope=query_rope, key_rope=key_rope, actual_seq_qlen=actual_seq_qlen, actual_seq_klen=actual_seq_klen, layout=layout, sparse_mode=sparse_mode, pre_tokens=65536, next_tokens=65536)

        # compare result
        dq_cpu, dk_cpu, dw_cpu, loss_cpu = cpu_out
        dq_npu, dk_npu, dw_npu, loss_npu = npu_out
        self.assertRtolEqual(dq_npu, dq_cpu, prec=1.e-3)
        self.assertRtolEqual(dk_npu, dk_cpu, prec=1.e-3)
        self.assertRtolEqual(dw_npu, dw_cpu, prec=1.e-3)
        self.assertRtolEqual(loss_npu[0], loss_cpu, prec=1.e-3)


if __name__ == "__main__":
    run_tests()
import torch
import torch_npu
import numpy as np
import torch.nn as nn
import math
import unittest

from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests



class TestCustomQuantLightningIndexer(TestCase):
    def _get_data_from_pa_cache(self, key, block_table, act_s2):
        block_num, block_size, n2, d = key.shape
        if n2 != 1:
            raise ValueError("n2 only support 1")
        need_blcok_num = (act_s2 + block_size - 1) // block_size
        act_s2_align = need_blcok_num * block_size
        out = torch.zeros((act_s2_align, d), dtype=key.dtype, device=key.device)
        for i in range(need_blcok_num):
            out[i * block_size:(i + 1) * block_size, :] = key[block_table[i], ...].reshape(block_size, d)

        return out[:act_s2, :]


    def _get_k_scale(self, key_dequant_scale, block_table, act_s2):
        block_num, block_size, n2 = key_dequant_scale.shape
        if n2 != 1:
            raise ValueError("n2 only support 1")
        need_blcok_num = (act_s2 + block_size - 1) // block_size
        act_s2_align = need_blcok_num * block_size
        out = torch.zeros((act_s2_align), dtype=key_dequant_scale.dtype, device=key_dequant_scale.device)
        key_dequant_scale = key_dequant_scale.reshape(block_num, block_size)
        for i in range(need_blcok_num):
            out[i * block_size:(i + 1) * block_size] = key_dequant_scale[block_table[i], ...].reshape(block_size)

        return out[:act_s2]


    def _quant_lightning_indexer(self, query, key, weights, query_dequant_scale, key_dequant_scale, actual_seq_lengths_query,
                                actual_seq_lengths_key, block_table,
                                layout_query="BSND", sparse_count=2048, sparse_mode=3):
        batch_size = query.shape[0]
        if layout_query == "TND":
            batch_size = actual_seq_lengths_query.shape[0]
        out_shape = list(query.shape)
        n2 = key.shape[2]
        d = query.shape[-1]
        n1 = query.shape[-2]
        out_shape[-1] = sparse_count
        out_shape[-2] = n2
        out = torch.zeros(out_shape, dtype=torch.int32, device=query.device).reshape(-1, n2, sparse_count) - 1
        act_s1 = 0
        act_s2 = 0
        process_q_len = 0
        for batch_id in range(batch_size):
            if actual_seq_lengths_query is None:
                act_s1 = query.shape[1]
            else:
                if layout_query == "TND":
                    act_s1 = actual_seq_lengths_query[batch_id] - process_q_len
                else:
                    act_s1 = actual_seq_lengths_query[batch_id]
            act_s2 = actual_seq_lengths_key[batch_id]
            # n1, s1, d
            now_q = query.reshape(-1, n1, d)[process_q_len:process_q_len + act_s1, :, :].transpose(0, 1).to(torch.int32)
            now_weights = weights.reshape(-1, n1, 1)[process_q_len:process_q_len + act_s1, :, :]
            now_query_scale = query_dequant_scale.reshape(-1, n1, 1)[process_q_len:process_q_len + act_s1, :, :]
            # s1, n1, 1
            weights_scale = now_weights * now_query_scale # float16
            process_q_len += act_s1
            now_block_table = block_table[batch_id, :]
            # d s2
            now_k = self._get_data_from_pa_cache(key, now_block_table, act_s2).transpose(0, 1).to(torch.int32)
            # s2
            now_k_scale = self._get_k_scale(key_dequant_scale, now_block_table, act_s2).to(torch.float32)
            # n1,s1,d @ d,s2 -> n1,s1,s2
            s_out = (torch.maximum(torch.matmul(now_q, now_k), torch.tensor(0)).to(torch.float32)) / 1024.0
            # n1,s1,s2 -> s1,n1,s2  to fp16
            s_out = s_out.to(torch.float16).transpose(0, 1).to(torch.float32)
            # s1,n1,1 -> s1,1,n1
            weights_scale = weights_scale.transpose(1, 2).to(torch.float32)
            # s1,1,n1 @ s1,n1,s2 -> s1,1,s2 -> s1,s2
            topk_in = torch.bmm(weights_scale, s_out).squeeze(1)
            # s1,s2 * s2
            topk_in = topk_in * now_k_scale
            tmp_s1 = topk_in.shape[0]
            tmp_s2 = topk_in.shape[1]
            if sparse_mode == 3:
                for i in range(tmp_s1):
                    topk_in[-1 - i, tmp_s2 - i:] = float('-inf')
            sorted_value, sorted_indices = torch.sort(topk_in, dim=1, descending=True, stable=True)
            if sparse_mode == 3:
                for i in range(tmp_s1):
                    sorted_indices[-1 - i, tmp_s2 - i:] = -1
            return_s2 = min(sparse_count, tmp_s2)
            out[process_q_len - act_s1:process_q_len, 0, :return_s2] = sorted_indices.to(torch.int32)[:, :return_s2]

        out = out.reshape(out_shape)
        return out

    def cpu_op_exec(self, query, key, weights, query_dequant_scale, key_dequant_scale, actual_seq_lengths_query,
                    actual_seq_lengths_key, block_table, layout_query, sparse_count, sparse_mode):
        output = self._quant_lightning_indexer(query, key, weights, query_dequant_scale, key_dequant_scale,
                                          actual_seq_lengths_query, actual_seq_lengths_key, block_table,
                                          layout_query, sparse_count, sparse_mode)
        output = output.cpu()

        return output

    def npu_op_exec_eager(self, query, key, weights, query_dequant_scale, key_dequant_scale, actual_seq_lengths_query,
                          actual_seq_lengths_key, block_table, query_quant_mode, key_quant_mode, layout_query,
                          layout_key, sparse_count, sparse_mode):
        npu_out = torch_npu.npu_quant_lightning_indexer(query, key, weights, query_dequant_scale, key_dequant_scale,
                                                        actual_seq_lengths_query=actual_seq_lengths_query,
                                                        actual_seq_lengths_key=actual_seq_lengths_key,
                                                        block_table=block_table,
                                                        query_quant_mode=query_quant_mode,
                                                        key_quant_mode=key_quant_mode,
                                                        layout_query=layout_query,
                                                        layout_key=layout_key, sparse_count=sparse_count,
                                                        sparse_mode=sparse_mode)
        npu_out = npu_out.cpu()

        return npu_out

    def quant_lightning_indexer_result(self, layout_query, b, t, s1, s2, act_seq_q, act_seq_k, sparse_mode,
                                       sparse_count = 2048):
        n1 = 64
        n2 = 1
        d = 128
        block_size = 128
        layout_key = 'PA_BSND'
        query_quant_mode = 0
        key_quant_mode = 0
        np.random.seed(0)
        # -------------
        max_block_table_num = (s2 + block_size - 1) // block_size
        block_table = torch.tensor([range(b * max_block_table_num)], dtype = torch.int32).reshape(b, -1)
        key = torch.tensor(np.random.uniform(-128, 127, (b * max_block_table_num, block_size, n2, d))).to(torch.int8)
        key_dequant_scale = torch.tensor(np.random.uniform(0, 10, (b * max_block_table_num, block_size, n2)))
        key_dequant_scale = key_dequant_scale.to(torch.float16)
        if layout_query == 'BSND':
            query = torch.tensor(np.random.uniform(-128, 127, (b, s1, n1, d))).to(torch.int8)
            query_dequant_scale = torch.tensor(np.random.uniform(0, 10, (b, s1, n1))).to(torch.float16)
            weights = torch.tensor(np.random.uniform(0, 0.01, (b, s1, n1))).to(torch.float16)
            actual_seq_lengths_query = torch.tensor(np.random.uniform(s1, s1, (b))).to(torch.int32) \
                                       if act_seq_q is None else torch.tensor(act_seq_q).to(torch.int32)
            actual_seq_lengths_key = torch.tensor(np.random.uniform(s2, s2, (b))).to(torch.int32) \
                                       if act_seq_k is None else torch.tensor(act_seq_k).to(torch.int32)
            print(f"------- test LIQuant BSND case b:{b} s1:{s1} s2:{s2} sparse_mode:{sparse_mode} ----------")
        else:
            query = torch.tensor(np.random.uniform(-128, 127, (t, n1, d))).to(torch.int8)
            query_dequant_scale = torch.tensor(np.random.uniform(0, 10, (t, n1))).to(torch.float16)
            weights = torch.tensor(np.random.uniform(0, 0.01, (t, n1))).to(torch.float16)
            actual_seq_lengths_query = torch.tensor(act_seq_q).to(torch.int32)
            actual_seq_lengths_key = torch.tensor(act_seq_k).to(torch.int32)
            print(f"------- test LIQuant TND case b:{b} t:{t} s2:{s2} act_seq_q:{act_seq_q} act_seq_k:{act_seq_k} ",
                  f"sparse_mode:{sparse_mode} -------")

        cpu_out = self.cpu_op_exec(query.cpu(), key.cpu(), weights.cpu(), query_dequant_scale.cpu(), key_dequant_scale.cpu(),
                                   actual_seq_lengths_query.cpu(), actual_seq_lengths_key.cpu(), block_table.cpu(),
                                   layout_query, sparse_count, sparse_mode)

        npu_eager_out = self.npu_op_exec_eager(query.npu(), key.npu(), weights.npu(), query_dequant_scale.npu(), key_dequant_scale.npu(),
                                               actual_seq_lengths_query.npu(), actual_seq_lengths_key.npu(), block_table.npu(),
                                               query_quant_mode, key_quant_mode,
                                               layout_query, layout_key, sparse_count, sparse_mode)
        res = npu_eager_out.equal(cpu_out)
        self.assertRtolEqual(res, True)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    def test_quant_lightning_indexer(self):
        # layout_query, b, t, s1, s2, act_seq_q, act_seq_k, sparse_mode
        test_case_list = [
            ("BSND", 24, None, 4, 512, None, None, 0),
        ]
        for case in test_case_list:
            self.quant_lightning_indexer_result(*case)


if __name__ == "__main__":
    run_tests()

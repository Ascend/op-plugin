import unittest
import torch
import numpy as np
import torch.nn as nn
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestCustomLightningIndexer(TestCase):
    def _get_data_from_pa_cache(self, key, block_table, act_s2):
        block_num, block_size, n2, d = key.shape
        if n2 != 1:
            raise ValueError("n2 only support 1")
        need_blcok_num = (act_s2 + block_size - 1) // block_size
        act_s2_align = need_blcok_num * block_size
        out = torch.zeros((act_s2_align, d), dtype=key.dtype, device=key.device)
        for i in range(need_blcok_num):
            out[i*block_size:(i+1)*block_size, :] = key[block_table[i], ...].reshape(block_size, d)

        return out[:act_s2, :]


    def _lightning_indexer(self, query, key, weights, actual_seq_lengths_query, actual_seq_lengths_key, block_table,
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
            now_q = query.reshape(-1, n1, d)[process_q_len:process_q_len+act_s1, :, :].transpose(0, 1).to(torch.float32)
            now_weights = weights.reshape(-1, n1, 1)[process_q_len:process_q_len+act_s1, :, :] \
                        .transpose(0, 1).to(torch.float32)
            process_q_len += act_s1
            now_block_table = block_table[batch_id, :]
            now_k = self._get_data_from_pa_cache(key, now_block_table, act_s2).transpose(0, 1).to(torch.float32)
            # n1,s1,d @ d,s2 -> n1,s1,s2
            relu_out = torch.maximum(torch.matmul(now_q, now_k), torch.tensor(0))
            weight_out = relu_out * now_weights
            # n1,s1,s2 -> s1,s2
            reduce_out = torch.sum(weight_out, dim=0)
            tmp_s1 = reduce_out.shape[0]
            tmp_s2 = reduce_out.shape[1]
            if sparse_mode == 3:
                for i in range(tmp_s1):
                    reduce_out[-1-i, tmp_s2-i:] = float('-inf')
            sorted_value, sorted_indices = torch.sort(reduce_out, dim=1, descending=True)
            return_s2 = min(sparse_count, tmp_s2)
            out[process_q_len - act_s1:process_q_len, 0, :return_s2] = sorted_indices.to(torch.int32)[:, :return_s2]

        out = out.reshape(out_shape)
        return out

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    def test_bsnd_lightning_indexer_eager(self):
        b = 1
        s1 = 1
        s2 = 8192
        n1 = 64
        n2 = 1
        d = 128
        block_size = 256
        t = 8192
        layout_query = 'BSND'

        np.random.seed(0)
        query = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, d))).to(torch.bfloat16)
        key = torch.tensor(np.random.uniform(-10, 10, (b*(s2//block_size), block_size, n2, d))).to(torch.bfloat16)
        weights = torch.tensor(np.random.uniform(-1, 1, (b, s1, n1))).to(torch.bfloat16)
        actual_seq_lengths_query = torch.tensor(np.random.uniform(s1, s1, (b))).to(torch.int32)
        actual_seq_lengths_key = torch.tensor(np.random.uniform(s2, s2, (b))).to(torch.int32)
        block_table = torch.tensor([range(b*s2//block_size)], dtype=torch.int32).reshape(b, -1)
        layout_key = 'PA_BSND'
        sparse_count = 2048
        sparse_mode = 3
        cpu_out = self._lightning_indexer(query, key, weights, actual_seq_lengths_query, actual_seq_lengths_key, block_table,
                                    layout_query, sparse_count, sparse_mode)

        query = query.npu()
        key = key.npu()
        weights = weights.npu()
        actual_seq_lengths_query = actual_seq_lengths_query.npu()
        actual_seq_lengths_key = actual_seq_lengths_key.npu()
        block_table = block_table.npu()

        npu_out, npu_out1 = torch_npu.npu_lightning_indexer(
            query, key, weights, actual_seq_lengths_query=actual_seq_lengths_query, 
                actual_seq_lengths_key=actual_seq_lengths_key, block_table=block_table, layout_query=layout_query, 
                layout_key=layout_key, sparse_count=sparse_count, sparse_mode=sparse_mode)

        # compare result
        npu_out = npu_out.reshape(-1, sparse_count).cpu()
        cpu_out = cpu_out.reshape(-1, sparse_count).cpu()
        res = npu_out.equal(cpu_out)
        self.assertRtolEqual(res, True)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    def test_tnd_lightning_indexer_eager(self):
        b = 3
        t = 5
        s2 = 8192
        n1 = 64
        n2 = 1
        d = 128
        block_size = 256
        layout_query = 'TND'

        np.random.seed(3)
        query = torch.tensor(np.random.uniform(-10, 10, (t, n1, d))).to(torch.bfloat16)
        key = torch.tensor(np.random.uniform(-10, 10, (b*(s2//block_size), block_size, n2, d))).to(torch.bfloat16)
        weights = torch.tensor(np.random.uniform(-1, 1, (t, n1))).to(torch.bfloat16)
        actual_seq_lengths_query = torch.tensor([1, 3, 5]).to(torch.int32)
        actual_seq_lengths_key = torch.tensor(np.random.uniform(s2, s2, (b))).to(torch.int32)
        block_table = torch.tensor([range(b*s2//block_size)], dtype=torch.int32).reshape(b, -1)
        layout_key = 'PA_BSND'
        sparse_count = 2048
        sparse_mode = 3
        cpu_out = self._lightning_indexer(query, key, weights, actual_seq_lengths_query, actual_seq_lengths_key, block_table,
                                    layout_query, sparse_count, sparse_mode)

        query = query.npu()
        key = key.npu()
        weights = weights.npu()
        actual_seq_lengths_query = actual_seq_lengths_query.npu()
        actual_seq_lengths_key = actual_seq_lengths_key.npu()
        block_table = block_table.npu()

        npu_out, npu_out1 = torch_npu.npu_lightning_indexer(
            query, key, weights, actual_seq_lengths_query=actual_seq_lengths_query, 
                actual_seq_lengths_key=actual_seq_lengths_key, block_table=block_table, layout_query=layout_query, 
                layout_key=layout_key, sparse_count=sparse_count, sparse_mode=sparse_mode)

        # compare result
        npu_out = npu_out.reshape(-1, sparse_count).cpu()
        cpu_out = cpu_out.reshape(-1, sparse_count).cpu()
        res = npu_out.equal(cpu_out)
        self.assertRtolEqual(res, True)


if __name__ == "__main__":
    run_tests()
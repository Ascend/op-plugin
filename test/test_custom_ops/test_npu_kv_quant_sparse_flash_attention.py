import math
import random
import unittest
import torch
import numpy as np
import torch.nn as nn
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestKvQuantSparseFlashAttention(TestCase):
    def pa_to_bsnd(self, pa_in, block_table, actual_seq_lengths):
        block_num, block_size, n, d = pa_in.shape
        b = len(actual_seq_lengths)
        out = torch.zeros((b, block_num * block_size // b, 1, d)).to(pa_in.dtype)
        for i in range(b):
            for j in range(actual_seq_lengths[i] // block_size):
                out[i, j * block_size: (j + 1) * block_size, 0, :] = \
                    pa_in[block_table[i][j], :, 0, :].reshape(block_size, d)
        return out


    def gather_kv(self, k_tensor, v_tensor, sparse_indices, sparse_block_size, sparse_count,
                  batch, n2_idx, s1_idx, cur_actual_seq_lengths_kv):
        s2_sparse = list()
        for sparse_id in sparse_indices:
            if sparse_id == -1: 
                break
            begin_idx = sparse_id * sparse_block_size
            end_idx = begin_idx + sparse_block_size \
                    if begin_idx + sparse_block_size <= cur_actual_seq_lengths_kv else cur_actual_seq_lengths_kv
            s2_sparse.extend(np.arange(begin_idx, end_idx))

        k_sparse, v_sparse = k_tensor[batch, n2_idx, s2_sparse, :], v_tensor[batch, n2_idx, s2_sparse, :]

        return k_sparse, v_sparse, torch.tensor(s2_sparse)

    def softmax(self, x):
        x = x.astype(np.float32)
        x_max = x.max(axis=-1, keepdims=True)
        x_sub = x - x_max
        y = np.exp(x_sub)
        x_sum = y.sum(axis=-1, keepdims=True)
        ans = y / x_sum
        return ans


    def cpu_kv_quant_sparse_flash_attention(self,
        query, key, value, sparse_indices, key_dequant_scale, value_dequant_scale,
        scale_value, sparse_block_size,
        actual_seq_lengths_query, actual_seq_lengths_kv,
        layout_query='BSND', layout_kv='PA_BSND', sparse_mode=3, block_table=None,
        attention_mode=0, quant_scale_repo_mode=0, tile_size=0, key_quant_mode=0,
        value_quant_mode=0, rope_head_dim=0):
        query_type = query.dtype
        query_rope = query[..., 512:]
        query = query[..., :512]
        key = self.pa_to_bsnd(key, block_table, actual_seq_lengths_kv)
        key_rope = key[..., 512: 512 + 64 * 2].view(query_type)
        key_quant_scale = key[..., 512 + 64 * 2:].view(torch.float32)
        key = key[..., :512].to(torch.float32)

        key_quant_scale = np.repeat(key_quant_scale, repeats=tile_size, axis=-1)
        key = (key * key_quant_scale).to(query_type)
        value = key

        batch_size = actual_seq_lengths_query.shape[0]
        num_heads = query.shape[2]
        num_kv_heads = key.shape[2]
        sparse_count = sparse_indices.shape[-1]
        g = num_heads // num_kv_heads

        q_bnsd_tensor = torch.transpose(torch.cat((query, query_rope), axis=-1), 1, 2)
        k_bnsd_tensor = torch.transpose(torch.cat((key, key_rope), axis=-1), 1, 2)
        v_bnsd_tensor = torch.transpose(value, 1, 2)
        sparse_indices_tensor = torch.transpose(sparse_indices, 1, 2)
        out_shape_bnsd = list(q_bnsd_tensor.shape)
        out_shape_bnsd[-1] = out_shape_bnsd[-1] - query_rope.shape[-1]
        y = torch.zeros(out_shape_bnsd, dtype=query_type)

        for batch in range(batch_size):
            cur_acutal_seq_lengths_q = actual_seq_lengths_query[batch]
            cur_actual_seq_lengths_kv = actual_seq_lengths_kv[batch]
            for n2_idx in range(num_kv_heads):
                for s1_idx in range(cur_acutal_seq_lengths_q):
                    q_curr = q_bnsd_tensor[batch, n2_idx * g: (n2_idx + 1) * g, s1_idx, :]
                    cur_sparse_indices = sparse_indices_tensor[batch, n2_idx, s1_idx, :]
                    k_sparse, v_sparse, s2_index = self.gather_kv(k_bnsd_tensor, v_bnsd_tensor, cur_sparse_indices, sparse_block_size,
                                                  sparse_count, batch, n2_idx, s1_idx, cur_actual_seq_lengths_kv)
                    mm1_res = torch.matmul(q_curr.to(torch.float32), k_sparse.to(torch.float32).T)
                    scale_res = mm1_res * scale_value
                    if sparse_mode == 3:
                        threshold = cur_actual_seq_lengths_kv - cur_acutal_seq_lengths_q + s1_idx + 1
                        mask_index = s2_index >= threshold
                        scale_res[:, mask_index] = -1e12
                    softmax_res = self.softmax(scale_res.numpy())
                    softmax_res = torch.tensor(softmax_res).to(query_type)
                    mm2_res = torch.matmul(softmax_res.to(torch.float32), v_sparse.to(torch.float32))
                    y[batch, n2_idx * g: (n2_idx + 1) * g, s1_idx, :] = mm2_res
        return torch.transpose(y, 1, 2)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    def test_sfa_eager(self, device="npu"):
        query_type = torch.bfloat16
        scale_value = 0.041666666666666664
        sparse_block_size = 1
        sparse_block_count = 2048
        b = 4
        s1 = 1
        s2 = 8192
        n1 = 128
        n2 = 1
        dn = 512
        dr = 64
        tile_size = 128
        block_size = 256
        layout_query = 'BSND'
        s2_act = 4096

        query = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, dn))).to(query_type)
        key = torch.tensor(np.random.uniform(-5, 10, (b * (s2 // block_size), block_size, n2, dn))).to(torch.int8)
        value = key.clone()
        idxs = random.sample(range(s2_act - s1 + 1), sparse_block_count)
        sparse_indices = torch.tensor([idxs for _ in range(b * s1 * n2)]).reshape(b, s1, n2, sparse_block_count). \
            to(torch.int32)
        query_rope = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, dr))).to(query_type)
        key_rope = torch.tensor(np.random.uniform(-10, 10, (b * (s2 // block_size), block_size, n2, dr))).to(query_type)
        act_seq_q = torch.tensor([s1] * b).to(torch.int32)
        act_seq_kv = torch.tensor([s2_act] * b).to(torch.int32)
        antiquant_scale = torch.tensor(np.random.uniform(-100, 100, (b * (s2 // block_size), block_size, n2,
            dn // tile_size))).to(torch.float32)
        key = torch.cat((key, key_rope.view(torch.int8), antiquant_scale.view(torch.int8)), axis=3)
        query = torch.cat((query, query_rope), axis=3)
        block_table = torch.tensor([range(b * s2 // block_size)], dtype=torch.int32).reshape(b, -1)

        # compare result
        cpu_out = self.cpu_kv_quant_sparse_flash_attention(
            query, key, value, sparse_indices,
            key_dequant_scale=antiquant_scale, value_dequant_scale=antiquant_scale,
            scale_value=scale_value, sparse_block_size=sparse_block_size,
            actual_seq_lengths_query=act_seq_q, actual_seq_lengths_kv=act_seq_kv,
            layout_query='BSND', layout_kv='PA_BSND', sparse_mode=3, block_table=block_table,
            attention_mode=2, quant_scale_repo_mode=1, tile_size=tile_size, key_quant_mode=2,
            value_quant_mode=2, rope_head_dim=64)

        query = query.npu()
        key = key.npu()
        value = value.npu()
        sparse_indices = sparse_indices.npu()
        query_rope = query_rope.npu()
        key_rope = key_rope.npu()
        act_seq_q = act_seq_q.npu()
        act_seq_kv = act_seq_kv.npu()
        block_table = block_table.npu()
        antiquant_scale = antiquant_scale.npu()

        npu_out = torch_npu.npu_kv_quant_sparse_flash_attention(
            query, key, value, sparse_indices, 
            scale_value=scale_value, sparse_block_size=sparse_block_size,
            actual_seq_lengths_query=act_seq_q, actual_seq_lengths_kv=act_seq_kv,
            layout_query='BSND', layout_kv='PA_BSND', sparse_mode=3, block_table=block_table,
            attention_mode=2, quant_scale_repo_mode=1, tile_size=tile_size, key_quant_mode=2,
            value_quant_mode=2, rope_head_dim=64)

        npu_out = npu_out.cpu().to(torch.float32).numpy()
        cpu_out = cpu_out.to(torch.float32).numpy()

        res = np.isclose(npu_out, cpu_out, rtol=0.005, atol=0.0001, equal_nan=False)
        true_ratio = np.mean(res)
        if true_ratio < 0.99:
            print("npu output:\n", npu_out, npu_out.shape)
            print("cpu output:\n", cpu_out, cpu_out.shape)
            print("correct ratio of cpu vs npu is:", true_ratio * 100, "%")
        self.assertTrue(true_ratio > 0.99, "precision compare fail")

    @SupportedDevices(['Ascend950'])
    def test_npu_kv_quant_sparse_flash_attention_tnd(self, device="npu"):
        query_type = torch.bfloat16
        scale_value = 0.041666666666666664
        sparse_block_size = 1
        sparse_block_count = 2048
        t = 1
        b = 1
        s1 = 1
        s2 = 3904
        n1 = 48
        n2 = 1
        dn = 512
        dr = 64
        tile_size = 128
        block_size = 128
        layout_query = 'TND'
        s2_act = 3904

        query = torch.tensor(np.random.uniform(-10, 100, (t, n1, dn))).to(query_type)
        key = torch.tensor(np.random.uniform(-100, 100, (b * (s2 // block_size), block_size, n2, dn))).to(torch.uint8)
        value = key.clone()
        idxs = random.sample(range(s2_act - s1 + 1), sparse_block_count)
        sparse_indices = torch.tensor([idxs for _ in range(b * s1 * n2)]).reshape(t, n2, sparse_block_count). \
            to(torch.int32)
        query_rope = torch.tensor(np.random.uniform(-10, 10, (t, n1, dr))).to(query_type)
        key_rope = torch.tensor(np.random.uniform(-10, 10, (b * (s2 // block_size), block_size, n2, dr))).to(query_type)
        act_seq_q = torch.tensor([s1] * b).to(torch.int32)
        act_seq_kv = torch.tensor([s2_act] * b).to(torch.int32)
        antiquant_scale = torch.tensor(np.random.uniform(-100, 100, (b * (s2 // block_size), block_size, n2,
            dn // tile_size))).to(torch.float32)
        key = torch.cat((key, key_rope.view(torch.uint8), antiquant_scale.view(torch.uint8)), axis=3)
        query = torch.cat((query, query_rope), axis=3)
        block_table = torch.tensor([range(b * s2 // block_size)], dtype=torch.int32).reshape(b, -1)

        # compare result
        cpu_out = self.cpu_kv_quant_sparse_flash_attention(
            query, key, value, sparse_indices,
            key_dequant_scale=antiquant_scale, value_dequant_scale=antiquant_scale,
            scale_value=scale_value, sparse_block_size=sparse_block_size,
            actual_seq_lengths_query=act_seq_q, actual_seq_lengths_kv=act_seq_kv,
            layout_query='TND', layout_kv='PA_BSND', sparse_mode=3, block_table=block_table,
            attention_mode=2, quant_scale_repo_mode=1, tile_size=tile_size, key_quant_mode=2,
            value_quant_mode=2, rope_head_dim=64)

        query = query.npu()
        key = key.npu()
        value = value.npu()
        sparse_indices = sparse_indices.npu()
        query_rope = query_rope.npu()
        key_rope = key_rope.npu()
        act_seq_q = act_seq_q.npu()
        act_seq_kv = act_seq_kv.npu()
        block_table = block_table.npu()
        antiquant_scale = antiquant_scale.npu()

        npu_out = torch_npu.npu_kv_quant_sparse_flash_attention(
            query, key, value, sparse_indices, 
            scale_value=scale_value, sparse_block_size=sparse_block_size,
            actual_seq_lengths_query=act_seq_q, actual_seq_lengths_kv=act_seq_kv,
            layout_query='TND', layout_kv='PA_BSND', sparse_mode=3, block_table=block_table,
            attention_mode=2, quant_scale_repo_mode=1, tile_size=tile_size, key_quant_mode=2,
            value_quant_mode=2, rope_head_dim=64)

        npu_out = npu_out.cpu().to(torch.float32).numpy()
        cpu_out = cpu_out.to(torch.float32).numpy()

        res = np.isclose(npu_out, cpu_out, rtol=0.005, atol=0.0001, equal_nan=False)
        true_ratio = np.mean(res)
        if true_ratio < 0.99:
            print("npu output:\n", npu_out, npu_out.shape)
            print("cpu output:\n", cpu_out, cpu_out.shape)
            print("correct ratio of cpu vs npu is:", true_ratio * 100, "%")
        self.assertTrue(true_ratio > 0.99, "precision compare fail")

if __name__ == "__main__":
    run_tests()
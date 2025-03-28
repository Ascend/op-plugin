import random
import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


NUM_HEADS = 4
NUM_TOKENS = 4
KV_HEADS = 1
BLOCK_SIZE = 16
HEAD_SIZE_QK = 576
HEAD_SIZE_VO = 512
NUM_BLOCKS = 19513
K_SEQLEN = 4
DTYPE = torch.float16
KV_RANGE = 1.0


class TestPagedAttention(TestCase):
    def group_mm_torch(self, heads, group_num, A, B):
        group_head = heads // group_num
        results = []
        for i in range(group_num):
            group_A = A[i * group_head: (i + 1) * group_head].float()
            group_B = B[i: i + 1].float()
            results.append(torch.matmul(group_A, group_B))
        return torch.cat(results, dim=0)

    def softmax(self, sim):
        sim = sim.float()
        row_max = torch.max(sim, dim=-1, keepdim=True)[0]
        sim = torch.exp(sim - row_max)
        return sim / torch.sum(sim, dim=-1, keepdim=True)

    def ref_masked_attention(self, query, key, value, scale, alibi_bias=None):
        """参考实现的注意力计算"""
        # 转置张量维度
        query = query.permute(1, 0, 2)  # [num_heads, seq_len, head_size]
        key = key.permute(1, 2, 0)      # [kv_heads, head_size, seq_len]
        
        # 分组矩阵乘法计算相似度
        sim_high = self.group_mm_torch(query.size(0), key.size(0), query, key)
        sim_high = sim_high * scale
        
        # 添加ALiBi偏置（如果存在）
        if alibi_bias is not None:
            sim_high += alibi_bias.float()
        
        # Softmax归一化
        p_high = self.softmax(sim_high)
        
        # 计算注意力输出
        value = value.permute(1, 0, 2)  # [kv_heads, seq_len, head_size]
        out = self.group_mm_torch(p_high.size(0), value.size(0), p_high, value)
        return out.permute(1, 0, 2)     # [seq_len, num_heads, head_size]

    def ref_single_query_cached_kv_attention(self, output, query, key_cache, value_cache, 
                                        block_tables, context_lens):
        """参考实现的单查询缓存KV注意力"""
        scale = 1.0 / (HEAD_SIZE_QK ** 0.5)
        
        for i, context_len in enumerate(context_lens):
            if context_len == 0:
                continue
            
            # 从缓存中收集KV数据
            keys = [key_cache[block_tables[i][j // BLOCK_SIZE], j % BLOCK_SIZE] for j in range(context_len)]
            values = [value_cache[block_tables[i][j // BLOCK_SIZE], j % BLOCK_SIZE] for j in range(context_len)]
            
            # 计算注意力
            attn_output = self.ref_masked_attention(
                query[i].unsqueeze(0),  # 增加batch维度
                torch.stack(keys),
                torch.stack(values),
                scale=scale
            )
            output[i] = attn_output.squeeze(0)


    def init_data(self):
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        query = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_SIZE_QK, dtype=DTYPE)
        key_cache = torch.randn(NUM_BLOCKS, BLOCK_SIZE, KV_HEADS, HEAD_SIZE_QK, dtype=DTYPE)
        value_cache = key_cache[..., :HEAD_SIZE_VO].clone()
        
        # 生成块表
        max_context_len = K_SEQLEN
        max_blocks_per_seq = (max_context_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        block_tables = [
            [random.randint(0, NUM_BLOCKS - 1) for _ in range(max_blocks_per_seq)]
            for _ in range(NUM_TOKENS)
        ]
        context_lens = [K_SEQLEN] * NUM_TOKENS
        
        return query, key_cache, value_cache, block_tables, context_lens

    def run_reference(self, query, key_cache, value_cache, block_tables, context_lens):
        ref_output = torch.zeros(NUM_TOKENS, NUM_HEADS, HEAD_SIZE_VO, dtype=DTYPE)
        self.ref_single_query_cached_kv_attention(
            ref_output, query, key_cache, value_cache,
            block_tables, context_lens
        )
        return ref_output

    def run_npu_optimized(self, query, key_cache, block_tables, context_lens):
        device = "npu:0"
        query_npu = query.to(device)
        key_cache_npu = key_cache.to(device)
        block_tables_npu = torch.tensor(block_tables, dtype=torch.int32, device=device)
        context_lens_npu = torch.tensor(context_lens, dtype=torch.int32)
        
        output_npu = torch.zeros(NUM_TOKENS, NUM_HEADS, HEAD_SIZE_VO, dtype=DTYPE, device=device)
        
        torch_npu._npu_paged_attention_mla(
            query_npu,
            key_cache_npu,
            KV_HEADS,
            NUM_HEADS,
            1.0 / (HEAD_SIZE_QK ** 0.5),
            block_tables_npu,
            context_lens_npu,
            HEAD_SIZE_VO,
            output_npu
        )
        return output_npu.cpu()

    @SupportedDevices(["Ascend910B"])
    def test_paged_attention_mla(self):
        query, key_cache, value_cache, block_tables, context_lens = self.init_data()
        
        ref_output = self.run_reference(query, key_cache, value_cache, block_tables, context_lens)
        npu_output = self.run_npu_optimized(query, key_cache, block_tables, context_lens)
        
        self.assertRtolEqual(npu_output, ref_output)


if __name__ == "__main__":
    run_tests()

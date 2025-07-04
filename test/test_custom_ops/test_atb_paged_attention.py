import random
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

MAX_SEQ_LEN = 1024
num_blocks = 64
num_tokens = 2
block_size = 128
kv_heads = 16
head_size = 288
num_heads = 32
dtype = np.float16
head_size_v = 96  


class TestPagedAttention(TestCase):
    def group_matmul(self, head, kv_head, A, B):
        group_num = head // kv_head
        score = []
        for i in range(kv_head):
            group_A = A[i * group_num: (i + 1) * group_num]
            group_B = B[i : i + 1]
            score.append(np.matmul(group_A, group_B))
        return np.concatenate(score, axis=0)

    def ref_masked_attention(self, query, key, value, scale):
        """参考注意力计算"""
        # 维度调整 [num_heads, seq_len, head_size]
        query = query * scale
        query = query.transpose(1, 0, 2)
        key = key.transpose(1, 2, 0)
        
        # QK^T计算
        sim = self.group_matmul(query.shape[0], key.shape[0], query, key)
        
        # Softmax归一化
        sim = sim - np.max(sim, axis=-1, keepdims=True)
        exp_sim = np.exp(sim.astype(np.float32))
        p = exp_sim / np.sum(exp_sim, axis=-1, keepdims=True)
        p = p.astype(dtype)
        
        # Value加权
        value = value.transpose(1, 0, 2)
        out = self.group_matmul(p.shape[0], key.shape[0], p, value)
        return out.transpose(1, 0, 2)

    def ref_attention_impl(self, query, key_cache, value_cache, block_tables, context_lens):
        """参考实现入口"""
        scale = 1.0 / np.sqrt(head_size)
        output = np.zeros((num_tokens, num_heads, head_size_v), dtype=dtype)
        
        for i in range(num_tokens):
            # 从缓存中收集当前序列的KV
            seq_blocks = block_tables[i]
            context_len = context_lens[i]
            
            keys = []
            values = []
            for pos in range(context_len):
                block_id = seq_blocks[pos // block_size]
                offset = pos % block_size
                keys.append(key_cache[block_id, offset].reshape(kv_heads, -1))
                values.append(value_cache[block_id, offset].reshape(kv_heads, -1))
            
            # 执行注意力计算
            out = self.ref_masked_attention(
                query[i:i + 1], 
                np.stack(keys), 
                np.stack(values),
                scale
            )
            output[i] = out.reshape(num_heads, -1)
        return output

    def prepare_inputs(self):
        """生成测试输入数据"""
        # Query输入
        query = np.random.uniform(-1, 1, (num_tokens, num_heads, head_size)).astype(dtype)
        
        # KV缓存初始化
        key_cache = np.random.uniform(-1, 1, (num_blocks, block_size, kv_heads, head_size)).astype(dtype)
        value_cache = np.random.uniform(-1, 1, (num_blocks, block_size, kv_heads, head_size_v)).astype(dtype)
        
        # 序列信息
        context_lens = np.full(num_tokens, MAX_SEQ_LEN, dtype=np.int32)
        max_blocks_per_seq = (MAX_SEQ_LEN + block_size - 1) // block_size
        block_tables = np.array([
            [random.randint(0, num_blocks - 1) for _ in range(max_blocks_per_seq)]
            for _ in range(num_tokens)
        ], dtype=np.int32)
        
        return query, key_cache, value_cache, block_tables, context_lens

    @SupportedDevices(["Ascend910B"])
    def test_paged_attention(self):

        query_np, key_cache_np, value_cache_np, block_tables_np, context_lens_np = self.prepare_inputs()
        ref_output = self.ref_attention_impl(query_np, key_cache_np, value_cache_np, block_tables_np, context_lens_np)
        
        query_t = torch.from_numpy(query_np).npu()
        key_cache_t = torch.from_numpy(key_cache_np).npu()
        value_cache_t = torch.from_numpy(value_cache_np).npu()
        block_tables_t = torch.from_numpy(block_tables_np).npu()
        context_lens_t = torch.from_numpy(context_lens_np)
        output_t = torch.zeros_like(query_t[:, :, :head_size_v]).npu()
        
        torch_npu._npu_paged_attention(
            query_t, key_cache_t, value_cache_t,
            kv_heads, num_heads, 1.0 / np.sqrt(head_size),
            block_tables_t, context_lens_t, output_t
        )
        
        ref_output = torch.from_numpy(ref_output)
        self.assertRtolEqual(output_t, ref_output)

    @SupportedDevices(["Ascend910B"])
    def test_paged_attention_cputensor(self):

        query_np, key_cache_np, value_cache_np, block_tables_np, context_lens_np = self.prepare_inputs()
        ref_output = self.ref_attention_impl(query_np, key_cache_np, value_cache_np, block_tables_np, context_lens_np)
        
        query_t = torch.from_numpy(query_np).npu()
        key_cache_t = torch.from_numpy(key_cache_np).npu()
        value_cache_t = torch.from_numpy(value_cache_np).npu()
        block_tables_t = torch.from_numpy(block_tables_np).npu()
        context_lens_t = torch.from_numpy(context_lens_np).npu()
        output_t = torch.zeros_like(query_t[:, :, :head_size_v]).npu()
        
        torch_npu._npu_paged_attention(
            query_t, key_cache_t, value_cache_t,
            kv_heads, num_heads, 1.0 / np.sqrt(head_size),
            block_tables_t, context_lens_t.cpu(), output_t
        )
        
        ref_output = torch.from_numpy(ref_output)
        self.assertRtolEqual(output_t, ref_output)

if __name__ == "__main__":
    run_tests()

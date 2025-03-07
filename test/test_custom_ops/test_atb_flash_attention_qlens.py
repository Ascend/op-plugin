import unittest
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


# 封装 generate_data 函数的输入参数
@dataclass
class GenerateDataParams:
    num_prompts: int = 2
    num_heads: int = 32
    kv_heads: int = 32
    head_size: int = 128
    block_size: int = 128
    num_blocks: int = 64


# 封装 ref_fa_with_prefix_cache 函数的输入参数
@dataclass
class RefFAWithPrefixCacheParams:
    output: torch.Tensor
    query: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    query_lens: torch.Tensor
    mask: torch.Tensor


# 封装 generate_data 函数的返回值
@dataclass
class GenerateDataResult:
    query: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    query_lens: torch.Tensor
    mask: torch.Tensor
    ref_output: torch.Tensor


class TestFA(TestCase):
    SCALE_TYPE = 0
    MAX_SEQ_LEN = 1024

    rand_list = np.random.uniform(0.0, 2.0, size=100).astype(np.float32)
    rand_list = [x if x > 1.0 else 1.0 for x in rand_list]

    def group_matmul(self, head, kv_head, A, B):
        group_num = head // kv_head
        score = None
        for i in range(kv_head):
            group_score = torch.matmul(
                A[i * group_num: (i + 1) * group_num, :, :],
                B[i: (i + 1), :, :],
            )
            if score is None:
                score = group_score
            else:
                score = torch.cat((score, group_score), dim=0)
        return score

    def ref_masked_attention(self, query, key, value, scale: float, mask=None):
        query_len = query.shape[0]
        seq_len = key.shape[0]
        query = query * scale
        query = query.permute(1, 0, 2)
        key = key.permute(1, 2, 0)
        score = self.group_matmul(query.shape[0], key.shape[0], query, key)

        if mask is not None:
            score = score + mask[:, seq_len - query_len:seq_len, :seq_len]

        row_max = torch.max(score, dim=-1, keepdim=True).values
        score -= row_max
        score = score.to(torch.float32)
        score = torch.exp(score)
        row_sum = torch.sum(score, axis=-1, keepdims=True)
        p = score / row_sum
        p = p.to(torch.bfloat16)

        value = value.permute(1, 0, 2)
        out = self.group_matmul(query.shape[0], key.shape[0], p, value)
        out = out.permute(1, 0, 2)
        return out

    def ref_fa_with_prefix_cache(self, params: RefFAWithPrefixCacheParams):
        num_prompts = params.query_lens.shape[0]
        num_heads = params.query.shape[1]
        kv_heads = params.value_cache.shape[2]
        head_size = params.key_cache.shape[3]
        head_size_v = params.value_cache.shape[3]
        block_size = params.value_cache.shape[1]

        curr = 0
        for i in range(num_prompts):
            seq_len = int(params.seq_lens[i])
            query_len = int(params.query_lens[i])
            querys = params.query[curr:curr + query_len]
            block_table = params.block_tables[i]

            keys = []
            values = []
            for j in range(seq_len):
                block_number = int(block_table[j // block_size])
                block_offset = j % block_size

                k = params.key_cache[block_number, block_offset, :, :]
                k = k.reshape(kv_heads, head_size)
                keys.append(k)

                v = params.value_cache[block_number, block_offset, :, :]
                v = v.reshape(kv_heads, head_size_v)
                values.append(v)
            keys = torch.stack(keys, dim=0)
            values = torch.stack(values, dim=0)

            if self.SCALE_TYPE == 1:
                scale = 1.0 * self.rand_list[i] / (head_size ** 0.5)
            else:
                scale = 1.0 / (head_size ** 0.5)
            out = self.ref_masked_attention(querys, keys, values, scale, params.mask)
            out = out.reshape(query_len, num_heads, head_size_v)
            params.output[curr:curr + query_len] = out
            curr += query_len

    def generate_data(self, params: GenerateDataParams):
        seq_lens = [1024] * params.num_prompts
        context_lens = random.choices([128 * n for n in range(1, 8)], k=params.num_prompts)
        query_lens = [seq_len - context_len for seq_len, context_len in zip(seq_lens, context_lens)]

        num_tokens = sum(query_lens)
        head_size_v = params.head_size

        query = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(num_tokens, params.num_heads, params.head_size))).to(torch.bfloat16)
        key_cache = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(params.num_blocks, params.block_size, params.kv_heads, params.head_size))).to(torch.bfloat16)
        value_cache = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(params.num_blocks, params.block_size, params.kv_heads, head_size_v))).to(torch.bfloat16)
        max_seq_len = max(seq_lens)
        max_num_blocks_per_seq = (max_seq_len + params.block_size - 1) // params.block_size
        block_tables = []
        for _ in range(params.num_prompts):
            block_table = [
                random.randint(0, params.num_blocks - 1) for _ in range(max_num_blocks_per_seq)
            ]
            block_tables.append(block_table)

        seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
        query_lens = torch.tensor(query_lens, dtype=torch.int32)
        block_tables = torch.tensor(block_tables, dtype=torch.int32)
        mask = torch.ones(size=(1, max_seq_len, max_seq_len), dtype=torch.bfloat16)
        mask = torch.triu(mask, 1)
        mask *= -10000.0

        ref_output = torch.zeros((num_tokens, params.num_heads, head_size_v), dtype=torch.bfloat16)
        cache_params = RefFAWithPrefixCacheParams(
            output=ref_output,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
            query_lens=query_lens,
            mask=mask
        )
        self.ref_fa_with_prefix_cache(cache_params)

        return GenerateDataResult(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
            query_lens=query_lens,
            mask=mask,
            ref_output=ref_output
        )

    @unittest.skip("skip case")
    @SupportedDevices(['Ascend910B'])
    def test_flash_attention_qlens(self):
        num_heads = 32
        num_kv_heads = 16
        head_dim = 128
        num_prompts = 2
        block_num = 64
        block_size = 128

        data_params = GenerateDataParams(
            num_prompts=num_prompts,
            num_heads=num_heads,
            kv_heads=num_kv_heads,
            head_size=head_dim,
            block_size=block_size,
            num_blocks=block_num
        )
        result = self.generate_data(data_params)

        query = result.query.npu()
        key_cache = result.key_cache.npu()
        value_cache = result.value_cache.npu()
        block_tables = result.block_tables.npu()
        gt_output = result.ref_output.npu()
        output = torch.empty_like(gt_output)

        seq_lens = result.seq_lens
        query_lens = result.query_lens

        scale = head_dim ** -0.5

        mask_compress = torch.ones(size=(128, 128), dtype=torch.bfloat16)
        mask_compress = torch.triu(mask_compress, 1)
        mask_compress = mask_compress.npu()

        torch_npu._npu_flash_attention_qlens(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_tables,
            mask=mask_compress,
            seq_len=query_lens,
            context_lens=seq_lens,
            num_kv_heads=num_kv_heads,
            num_heads=num_heads,
            scale_value=scale,
            out=output)
        self.assertRtolEqual(output, gt_output)

if __name__ == '__main__':
    run_tests()

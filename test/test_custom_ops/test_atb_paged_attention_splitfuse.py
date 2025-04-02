import random
from dataclasses import dataclass
import numpy as np
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


# 封装 generate_data 函数的输入参数
@dataclass
class GenerateDataParams:
    num_seqs: int = 2
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
    max_seq_len: int


class TestFA(TestCase):
    SCALE_TYPE = 0
    MAX_SEQ_LEN = 1024

    rand_list = np.random.uniform(0.0, 2.0, size=100).astype(np.float32)
    rand_list = [x if x > 1.0 else 1.0 for x in rand_list]

    def make_attention_mask(self, max_seq_len, seq_lens, query_lens):
        # for paged attention
        atten_mask = np.zeros([0, max_seq_len])
        for i, context_length in enumerate(seq_lens):
            q_len = query_lens[i]
            ones_len = context_length - q_len
            ones = np.ones((q_len, ones_len), dtype=np.float16)
            bias_cache = np.tril(np.ones((q_len, max_seq_len - ones_len), dtype=np.float16))
            bias_cache = np.concatenate((ones, bias_cache), axis=1)
            mask_value = -10000
            bias_cache[bias_cache == 0] = mask_value
            bias_cache[bias_cache == 1] = 0

            atten_mask = np.concatenate([atten_mask, bias_cache], axis=0)
        atten_mask = torch.from_numpy(atten_mask)
        return atten_mask

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
        # Q * K.T
        query_len = query.shape[0]
        seq_len = key.shape[0]
        query = query * scale
        query = query.permute(1, 0, 2)
        key = key.permute(1, 2, 0)
        score = self.group_matmul(query.shape[0], key.shape[0], query, key)

        if mask is not None:
            score = score + mask[:, seq_len - query_len:seq_len, :seq_len]
        # softmax
        row_max = torch.max(score, dim=-1, keepdim=True).values
        score -= row_max
        score = score.to(torch.float32)
        score = torch.exp(score)
        row_sum = torch.sum(score, axis=-1, keepdims=True)
        p = score / row_sum
        p = p.to(torch.bfloat16)
        # P * V
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
        num_seqs = params.num_seqs
        num_heads = params.num_heads
        kv_heads = params.kv_heads
        head_size = params.head_size
        block_size = params.block_size
        num_blocks = params.num_blocks

        assert num_seqs > 1
        num_prefill_seqs = np.random.choice(np.arange(1, num_seqs))
        num_decode_seqs = num_seqs - num_prefill_seqs
        # seq_lens
        seq_prefill_lens = [1024] * num_prefill_seqs
        seq_decode_lens = [1024] * num_decode_seqs
        seq_lens = seq_prefill_lens + seq_decode_lens
        # context_lens
        context_prefill_lens = random.choices([128 * n for n in range(1, 8)], k=num_prefill_seqs)
        context_decode_lens = [1] * num_decode_seqs
        context_lens = context_prefill_lens + context_decode_lens
        # query_lens
        query_prefill_lens = [seq_len - context_len for seq_len, context_len in zip(seq_prefill_lens, context_prefill_lens)]
        query_decode_lens = [1] * num_decode_seqs
        query_lens = query_prefill_lens + query_decode_lens

        num_tokens = sum(query_lens)

        head_size_v = head_size
        query = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(num_tokens, num_heads, head_size))).to(torch.bfloat16)
        key_cache = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(num_blocks, block_size, kv_heads, head_size))).to(torch.bfloat16)
        value_cache = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(num_blocks, block_size, kv_heads, head_size_v))).to(torch.bfloat16)
        max_seq_len = max(seq_lens)

        max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
        block_tables = []
        for _ in range(num_seqs):
            block_table = [
                random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_seq)
            ]
            block_tables.append(block_table)

        seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
        query_lens = torch.tensor(query_lens, dtype=torch.int32)
        block_tables = torch.tensor(block_tables, dtype=torch.int32)

        mask = torch.ones(size=(1, max_seq_len, max_seq_len), dtype=torch.bfloat16)
        mask = torch.triu(mask, 1)
        mask *= -10000.0

        ref_output = torch.zeros((num_tokens, num_heads, head_size_v), dtype=torch.bfloat16)
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
            ref_output=ref_output,
            max_seq_len=max_seq_len
        )

    @SupportedDevices(['Ascend910B'])
    def test_paged_attention_splitfuse(self):
        num_heads = 32
        num_kv_heads = 16
        head_dim = 128
        num_seqs = 4
        block_num = 64
        block_size = 128

        data_params = GenerateDataParams(
            num_seqs=num_seqs,
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

        scale = head_dim ** -0.5

        atten_mask = self.make_attention_mask(result.max_seq_len, result.seq_lens.cpu().tolist(), result.query_lens.cpu().tolist())
        atten_mask = atten_mask.to(torch.bfloat16).npu()

        torch_npu._npu_paged_attention_splitfuse(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_tables,
            context_lens=result.seq_lens,
            mask=atten_mask,
            seq_len=result.query_lens,
            num_kv_heads=num_kv_heads,
            num_heads=num_heads,
            scale_value=scale,
            out=output)

        self.assertRtolEqual(output, gt_output)


if __name__ == '__main__':
    run_tests()

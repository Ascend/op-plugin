#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

import random
from dataclasses import dataclass
import numpy as np
import torch
import torch_npu
import math

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

def shape_nd_to_nz(shape, dtype='float16'):
    assert len(shape) >= 2
    batch = shape[:-2]   # 最后两维nd->nz
    a, b = shape[-2], shape[-1]
    a0, b0 = 16, 16
    return list(batch) + [math.ceil(b / b0), math.ceil(a / a0), a0, b0]

def gen_axes_for_transpose(offset, base):
    return [x for x in range(offset)] + [x + offset for x in base]

def convert_nd_to_nz(x):
    array_trans = gen_axes_for_transpose(len(x.shape) - 2, [2, 0, 1, 3]) # (m1, m0, n1, n0) -> (n1, m1, m0, n0)
    x_shape = shape_nd_to_nz(x.shape, dtype=x.dtype)
    *_, n1, m1, m0, n0 = x_shape
    return x.reshape(x_shape[:-4] + [m1, m0, n1, n0]).transpose(*array_trans) # x原始需要对齐，才能reshape


@dataclass
class GenerateDataParams:
    num_seqs: int = 2
    num_heads: int = 32
    kv_heads: int = 32
    head_size: int = 128
    block_size: int = 128
    num_blocks: int = 64


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


class TestPagedAttentionSplitfuseV2(TestCase):
    MAX_SEQ_LEN = 1024

    def make_attention_mask(self, max_seq_len, seq_lens, query_lens):
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
        query_len = query.shape[0]
        seq_len = key.shape[0]
        query = query * scale
        query = query.permute(1, 0, 2)
        key = key.permute(1, 2, 0)
        score = self.group_matmul(query.shape[0], key.shape[0], query, key)

        if mask is not None:
            score = score + mask[seq_len - query_len:seq_len, :seq_len]
        row_max = torch.max(score, dim=-1, keepdim=True).values
        score -= row_max
        score = score.to(torch.float32)
        score = torch.exp(score)
        row_sum = torch.sum(score, axis=-1, keepdims=True)
        p = score / row_sum
        p = p.to(torch.float16)
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
        seq_prefill_lens = [1024] * num_prefill_seqs
        seq_decode_lens = [1024] * num_decode_seqs
        seq_lens = seq_prefill_lens + seq_decode_lens
        context_prefill_lens = random.choices([128 * n for n in range(1, 8)], k=num_prefill_seqs)
        context_decode_lens = [1] * num_decode_seqs
        context_lens = context_prefill_lens + context_decode_lens
        query_prefill_lens = [seq_len - context_len for seq_len, context_len in zip(seq_prefill_lens, context_prefill_lens)]
        query_decode_lens = [1] * num_decode_seqs
        query_lens = query_prefill_lens + query_decode_lens

        num_tokens = sum(query_lens)

        head_size_v = head_size
        query = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(num_tokens, num_heads, head_size))).to(torch.float16)
        key_cache = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(num_blocks, block_size, kv_heads, head_size))).to(torch.float16)
        value_cache = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(num_blocks, block_size, kv_heads, head_size_v))).to(torch.float16)
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

        # float16 ND 格式的因果 mask，0=可见，-10000=被遮住
        # MASK_TYPE_NORM_COMPRESS 要求 mask 尺寸必须是 2048×2048（ATB 固定要求）
        mask = torch.ones(size=(2048, 2048), dtype=torch.float16)
        mask = torch.triu(mask, 1)
        mask *= -10000.0

        ref_output = torch.zeros((num_tokens, num_heads, head_size_v), dtype=torch.float16)
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
        # golden 计算使用 ND 格式（4D）
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

    @SupportedDevices(['Ascend310P'])
    def test_paged_attention_splitfuse_v2(self):
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
        # key_cache/value_cache：从 ND 格式转换为 fractal_nz 格式（lastDim=16）
        # [64, 128, 16, 128] → [64, 128, 128, 16]

        key_cache = result.key_cache.numpy().reshape(block_num, block_size, -1)
        key_cache_nz = convert_nd_to_nz(key_cache)
        key_cache_nz = key_cache_nz.reshape(block_num, -1, block_size, 16).astype(np.float16)
        key_cache_nz = np.ascontiguousarray(key_cache_nz)
        key_cache = torch.from_numpy(key_cache_nz).npu()

        value_cache = result.value_cache.numpy().reshape(block_num, block_size, -1)
        value_cache_nz = convert_nd_to_nz(value_cache)
        value_cache_nz = value_cache_nz.reshape(block_num, -1, block_size, 16).astype(np.float16)
        value_cache_nz = np.ascontiguousarray(value_cache_nz)
        value_cache = torch.from_numpy(value_cache_nz).npu()
        block_tables = result.block_tables.npu()
        gt_output = result.ref_output.npu()
        output = torch.empty_like(gt_output)

        scale = head_dim ** -0.5

        torch_npu._npu_paged_attention_splitfuse_v2(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_tables,
            context_lens=result.seq_lens.npu(),
            mask=result.mask.npu(),
            seq_len=result.query_lens,
            num_kv_heads=num_kv_heads,
            num_heads=num_heads,
            scale_value=scale,
            mask_type=5,
            out=output)

        self.assertRtolEqual(output, gt_output)

    @SupportedDevices(['Ascend310P'])
    def test_paged_attention_splitfuse_v2_default_mask_type(self):
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
        # key_cache/value_cache：从 ND 格式转换为 fractal_nz 格式（lastDim=16）
        # [64, 128, 16, 128] → [64, 128, 128, 16]

        key_cache = result.key_cache.numpy().reshape(block_num, block_size, -1)
        key_cache_nz = convert_nd_to_nz(key_cache)
        key_cache_nz = key_cache_nz.reshape(block_num, -1, block_size, 16).astype(np.float16)
        key_cache_nz = np.ascontiguousarray(key_cache_nz)
        key_cache = torch.from_numpy(key_cache_nz).npu()

        value_cache = result.value_cache.numpy().reshape(block_num, block_size, -1)
        value_cache_nz = convert_nd_to_nz(value_cache)
        value_cache_nz = value_cache_nz.reshape(block_num, -1, block_size, 16).astype(np.float16)
        value_cache_nz = np.ascontiguousarray(value_cache_nz)
        value_cache = torch.from_numpy(value_cache_nz).npu()
        block_tables = result.block_tables.npu()
        gt_output = result.ref_output.npu()
        output = torch.empty_like(gt_output)

        scale = head_dim ** -0.5

        torch_npu._npu_paged_attention_splitfuse_v2(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_tables,
            context_lens=result.seq_lens.npu(),
            mask=result.mask.npu(),
            seq_len=result.query_lens,
            num_kv_heads=num_kv_heads,
            num_heads=num_heads,
            scale_value=scale,
            out=output)

        self.assertRtolEqual(output, gt_output)


if __name__ == '__main__':
    run_tests()

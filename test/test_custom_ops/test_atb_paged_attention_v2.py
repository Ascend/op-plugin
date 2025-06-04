import random

import numpy as np
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

np.random.seed(1)
random.seed(1)
MAX_SEQ_LEN = 1024
kv_head_num = 32


class PagedInputData:
    def __init__(self, query, key_cache, value_cache, block_tables, context_lens, mask):
        self.query = query
        self.key_cache = key_cache
        self.value_cache = value_cache
        self.block_tables = block_tables
        self.context_lens = context_lens
        self.mask = mask


class TestPagedAttentionAlibi(TestCase):
    def group_mm_torch(self, heads, group_num, A, B):
        group_head = heads // group_num
        score = None
        for i in range(group_num):
            group_score = torch.matmul(A[i * group_head: (i + 1) * group_head, :, :].to(torch.float32),
                                    B[i:(i + 1), :, :].to(torch.float32))
            if score is None:
                score = group_score
            else:
                score = torch.cat((score, group_score), 0)
        return score

    # pylint:disable = huawei-too-many-arguments
    def ref_masked_attention(
            self,
            query,
            key,
            value,
            scale: float,
            alibi_bias,
            mask_data_type=torch.bfloat16
    ):
        # Q * K.T
        query = query
        query = torch.permute(query, (1, 0, 2))
        key = torch.permute(key, (1, 2, 0))  # 0 1 2
        sim = self.group_mm_torch(query.shape[0], key.shape[0], query, key).to(mask_data_type)
        sim = sim.to(torch.float32) * scale
        sim = sim + alibi_bias.to(torch.float32)
        sim = sim.numpy()
        # softmax
        row_max = np.max(sim, axis=-1, keepdims=True)
        sim -= row_max
        sim = np.exp(sim)
        row_sum = np.sum(sim, axis=-1, keepdims=True)
        p = sim / row_sum
        p = torch.from_numpy(p).to(mask_data_type)
        # P * V
        value = torch.permute(value, (1, 0, 2))

        out = self.group_mm_torch(query.shape[0], key.shape[0], p, value)
        out = torch.permute(out, (1, 0, 2))
        return out

    # pylint:disable = huawei-too-many-arguments
    def ref_single_query_cached_kv_attention(
            self,
            output,
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            mask,
            mask_dim=4,
            mask_data_type=torch.bfloat16
    ) -> None:
        num_heads = query.shape[1]
        kv_heads = value_cache.shape[2]
        head_size = key_cache.shape[3]
        head_size_v = value_cache.shape[3]
        block_size = value_cache.shape[1]

        num_input_tokens = query.shape[0]
        for i in range(num_input_tokens):
            q = query[i].view(1, num_heads, head_size)
            block_table = block_tables[i]
            context_len = int(context_lens[i])

            keys = []
            values = []
            for j in range(context_len):
                block_number = int(block_table[j // block_size])
                block_offset = j % block_size

                k = key_cache[block_number, block_offset, :, :]
                k = k.reshape(kv_heads, head_size)
                keys.append(k)

                v = value_cache[block_number, block_offset, :, :]
                v = v.reshape(kv_heads, head_size_v)
                values.append(v)
            keys = torch.stack(keys, axis=0)
            values = torch.stack(values, axis=0)
            scale = np.float32(1.0 / (head_size ** 0.5))
            if mask_dim == 4:
                out = self.ref_masked_attention(q, keys, values, scale, mask[i, :, :, :context_len], mask_data_type)
                out = out.reshape(num_heads, head_size_v)
            elif mask_dim == 3:
                out = self.ref_masked_attention(q, keys, values, scale, mask[i, :, :context_len], mask_data_type)
                out = out.reshape(num_heads, head_size_v)
            output[i] = out

    # pylint:disable = huawei-too-many-arguments
    def calc_data(self, num_tokens, num_heads, kv_heads, head_size, block_size, num_blocks, k_seqlen, dtype, mask_dim=4, mask_data_type=torch.bfloat16):
        head_size_v = np.random.randint(1, head_size)
        query = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(num_tokens, num_heads, head_size))).to(mask_data_type)
        key_cache = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(num_blocks, block_size, kv_heads, head_size))).to(mask_data_type)
        value_cache = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(num_blocks, block_size, kv_heads, head_size_v))).to(mask_data_type)
        context_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_tokens)]
        context_lens = [k_seqlen] * num_tokens
        max_context_len = max(context_lens)
        max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
        block_tables = []
        for _ in range(num_tokens):
            block_table = [random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_seq)]
            block_tables.append(block_table)

        # alibi mask
        if mask_dim == 4:
            alibi_slopes = np.random.random(num_heads).astype(np.float16)
            mask = np.zeros((num_tokens, num_heads, 1, max_context_len), dtype=np.float16)
            for i, context_len in enumerate(context_lens):
                position_ids = np.arange(context_len).astype(np.int32)
                alibi_bias = (position_ids - context_len + 1).astype(np.float16)
                alibi_bias = alibi_slopes.reshape(-1, 1, 1) * alibi_bias.reshape(1, 1, -1)
                mask[i, :, :, :context_len] = alibi_bias
            mask = torch.from_numpy(mask).to(mask_data_type)
        # normal mask
        elif mask_dim == 3:
            mask = np.zeros((num_tokens, 1, max_context_len), dtype=np.float16)
            for i in range(num_tokens):
                mask[i, :, :i] = -10000
            mask = torch.from_numpy(mask).to(mask_data_type)
        else:
            assert (False)
        ref_output = torch.zeros((num_tokens, num_heads, head_size_v)).to(mask_data_type)
        self.ref_single_query_cached_kv_attention(
            ref_output,
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            mask,
            mask_dim,
            mask_data_type)
        q = query
        key_cache = key_cache
        value_cache = value_cache
        block_tables = np.array(block_tables).astype(np.int32)
        contex_lens = np.array(context_lens).astype(np.int32)
        alib_mask = mask
        golden_out = ref_output
        # pylint:disable=too-many-return-values
        return q, key_cache, value_cache, block_tables, contex_lens, alib_mask, golden_out

    def compare_output_data(self, out, golden, ratios):
        error_count = 0
        strict_error_count = 0
        fp16_min_normal = 1.0 / (1 << 14)
        golden = golden.to(torch.float32)
        out = out.to(torch.float32)
        total_elements = out.shape[0] * out.shape[1] * out.shape[2]
        diff = torch.abs(golden - out)
        max_diff = diff.max().item()
        limit_error = torch.maximum(torch.abs(golden * ratios[0]), torch.tensor(ratios[1]))
        strict_limit_error = torch.maximum(torch.abs(golden * ratios[2]), torch.tensor(ratios[3]))
        error_count = torch.gt(diff, limit_error).sum().item()
        strict_error_count = torch.gt(diff, strict_limit_error).sum().item()
        print(f"maxDiff {max_diff}")
        print("1/1000 Accuracy is %f", 1 - float(error_count) / total_elements)
        print("5/1000 Accuracy is %f", 1 - float(strict_error_count) / total_elements)
        # 旧精度标准：双千分之五
        if self.data_type == torch.bfloat16 or self.is_int8_flag:
            print("accuracy is correct: %r", (float(strict_error_count) / total_elements) <= ratios[2])
        else:
            print("accuracy is correct: %r", (float(strict_error_count) / total_elements) <= ratios[0])
        # 新精度标准 参考精度标准v0.3浮点计算单标杆
        # 计算次数 两个matmul + 一个softmax
        calc_times = out.shape[2] * self.max_context_len + 4
        if self.data_type == torch.bfloat16:
            if calc_times < 2048:
                error = 2**(-7)
            else:
                error = 2**(-6)
            error_threshold = torch.clamp(torch.abs(golden), min=1) * error
            return (diff <= error_threshold).all()
        else:
            if calc_times < 2048:
                error = 2**(-8)
            else:
                error = 2**(-7)
            error_threshold = torch.clamp(torch.abs(golden), min=1) * error
            return (diff <= error_threshold).all()

    @SupportedDevices(['Ascend910B'])
    def test_paged_attention_v2_bf16(self):
        self.num_tokens = 1
        self.num_heads = 32
        self.kv_heads = 32
        self.block_size = 128
        self.head_size = 288
        self.num_blocks = 64
        self.k_seqlen = 128
        self.tor = 1.0 / (self.head_size ** 0.5)
        self.dtype = "float16"
        self.mask_dim = 4
        self.data_type = torch.bfloat16
        self.max_context_len = self.k_seqlen
        self.q, self.key_cache, self.value_cache, self.block_tables, self.contex_lens, self.alib_mask, self.golden_out = self.calc_data(
            self.num_tokens,
            self.num_heads,
            self.kv_heads,
            self.head_size,
            self.block_size,
            self.num_blocks,
            self.k_seqlen,
            self.dtype,
            self.mask_dim,
            self.data_type)
        self.data = self.q, self.key_cache, self.value_cache, torch.from_numpy(self.block_tables), torch.from_numpy(
            self.contex_lens), self.alib_mask, self.golden_out
        self.in_tensors = [tensor.npu() for tensor in self.data]
        self.query = self.in_tensors[0]
        self.keyCache = self.in_tensors[1]
        self.valueCache = self.in_tensors[2]
        self.blockTables = self.in_tensors[3]
        self.contextLens = self.in_tensors[4].cpu() 
        self.mask = self.in_tensors[5]
        self.attnOut = torch.empty_like(self.golden_out).npu()
        torch_npu.atb._npu_paged_attention_v2(self.query, self.keyCache, self.blockTables, self.contextLens, value_cache=self.valueCache, mask=self.mask, num_kv_heads=self.kv_heads, num_heads=self.num_heads, scale_value=self.tor, mask_type=2, out=self.attnOut)
        ratios = [0.001, 0.001, 0.005, 0.005]
        self.compare_output_data(self.attnOut.cpu(), self.golden_out, ratios)

    @SupportedDevices(['Ascend910B'])
    def test_paged_attention_v2(self):
        self.num_tokens = 1
        self.num_heads = 32
        self.kv_heads = 32
        self.block_size = 128
        self.head_size = 288
        self.num_blocks = 64
        self.k_seqlen = 128
        self.tor = 1.0 / (self.head_size ** 0.5)
        self.dtype = "float16"
        self.mask_dim = 4
        self.data_type = torch.float16
        self.is_int8_flag = False
        self.max_context_len = self.k_seqlen
        self.q, self.key_cache, self.value_cache, self.block_tables, self.contex_lens, self.alib_mask, self.golden_out = self.calc_data(
            self.num_tokens, self.num_heads, self.kv_heads, self.head_size, self.block_size, self.num_blocks,
            self.k_seqlen, self.dtype, self.mask_dim, self.data_type)
        self.data = self.q, self.key_cache, self.value_cache, torch.from_numpy(self.block_tables), torch.from_numpy(
            self.contex_lens), self.alib_mask, self.golden_out
        self.in_tensors = [tensor.npu() for tensor in self.data]
        self.query = self.in_tensors[0]
        self.keyCache = self.in_tensors[1]
        self.valueCache = self.in_tensors[2]
        self.blockTables = self.in_tensors[3]
        self.contextLens = self.in_tensors[4].cpu() 
        self.mask = self.in_tensors[5]
        self.attnOut = torch.empty_like(self.golden_out).npu()
        torch_npu.atb._npu_paged_attention_v2(self.query, self.keyCache, self.blockTables, self.contextLens, value_cache=self.valueCache, mask=self.mask, num_kv_heads=self.kv_heads, num_heads=self.num_heads, scale_value=self.tor, mask_type=2, out=self.attnOut)
        self.assertRtolEqual(self.golden_out, self.attnOut)
        ratios = [0.001, 0.001, 0.005, 0.005]
        self.compare_output_data(self.attnOut.cpu(), self.golden_out, ratios)

if __name__ == '__main__':
    run_tests()

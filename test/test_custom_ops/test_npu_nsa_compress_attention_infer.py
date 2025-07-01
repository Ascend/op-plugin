# Copyright (c) 2020, Huawei Technologies.All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import unittest
from dataclasses import dataclass
import numpy as np
import torch
from einops import rearrange
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices

TOPK_CHECK_OUT_SUFFIX = 2


def golden_add_gen(input_shape, compress_block_size, compress_stride, select_block_size, x=None):
    S1, S2 = input_shape
    pcmp = torch.randn(input_shape, dtype=torch.float)
    if x is not None:
        pcmp = x
    pcmp_transpose = pcmp.transpose(1, 0)
    pslc = torch.zeros([S2, S1], dtype=torch.float)
    for j in range(1, S2):
        for m in range(int(select_block_size / compress_stride)):
            for n in range(int(compress_block_size / compress_stride)):
                idx = int(select_block_size / compress_stride * j) - m - n
                if idx >= S2:
                    continue
                if idx < 0:
                    continue
                pslc[j - 1, :] += pcmp_transpose[idx, :]
    return pslc


def golden_reduce_sum(pslc, s2, g, s2_new):
    pslc[0, :] = np.inf
    pslc[s2_new - 2:s2_new, :] = np.inf
    p = pslc.reshape(s2, -1, g)
    res = p.sum(-1)
    return res.transpose(1, 0)[:, :s2_new]


def sort_and_select(input_tensor, k):
    # 获取输入tensor的形状 (行数, 列数)
    rows, cols = input_tensor.shape

    # 结果矩阵，用来存储每行处理后的结果
    result = np.zeros((rows, k), dtype=np.int32)  # 每行保留k个元素，指定 dtype 为 int32

    # 每行多存几个结果，用于topk校验
    topk_check = np.zeros((rows, k + TOPK_CHECK_OUT_SUFFIX), dtype=np.int32)

    for i in range(rows):
        # 获取当前行
        row = input_tensor[i, :]

        # 保留第一个和最后两个元素
        first = row[0]
        last_two = row[-2:]

        # 排序其余元素，去除第一个和最后两个元素
        middle_elements = row[1:-2]
        sorted_middle = np.argsort(-middle_elements, kind='mergesort')  # 返回排序后的索引（降序）

        # 获取前k-3个最大的数的索引
        top_k_indices = sorted_middle[:k - 3]

        # 每行多存几个结果，用于topk校验
        topk_check_indices = sorted_middle[:k - 3 + TOPK_CHECK_OUT_SUFFIX]

        # 构建新的行索引
        result_row_indices = np.concatenate(([0], [cols - 2, cols - 1], top_k_indices + 1))  # 修改这里，选择最后两个元素
        # 每行多存几个结果，用于topk校验
        result_check_row_indices = np.concatenate(([0], [cols - 2, cols - 1], topk_check_indices + 1))

        # 将结果存储到 result 中
        result[i, :] = result_row_indices.astype(np.int32)  # 转换为 int32 类型
        # 每行多存几个结果，用于topk校验
        topk_check[i, :] = result_check_row_indices.astype(np.int32)

    return result, topk_check


class TestPagedMLAttention():
    @dataclass
    class AttentionInputs:
        query: any
        key_cache: any
        value_cache: any
        block_tables: any
        q_seqlen_list: any
        k_seqlen_list: any
        global_mask: any
        mask_type: any

    @dataclass
    class GenDataParams:
        q_seqlen_list: list
        k_seqlen_list: list
        num_heads: int
        kv_heads: int
        head_size: int
        head_size_rope: int
        num_blocks: int
        block_size: int
        mask_type: int
        dtype: any

    @classmethod
    def check_attr(cls, batch: int, q_seqlen: int, kv_seqlen: int, num_blocks: int, block_size: int):
        if batch * kv_seqlen > num_blocks * block_size:
            raise RuntimeError("[ERROR] the number of K and V tokens is too big to fit in the paged cache.")

        if block_size > 128:
            raise RuntimeError("[ERROR] blockSize > 128 is not supported.")

        if q_seqlen > 4:
            raise RuntimeError("[ERROR] q_seqlen > 4 is not supported.")

    @classmethod
    # pylint:disable = huawei-too-many-arguments
    def check_attr1(cls, batch: int, q_seqlen: int, max_kv_seqlen: int, min_kv_seqlen: int, compress_block_size: int, compress_stride: int, select_block_size: int, num_blocks: int, block_size: int, topk: int):
        if batch * max_kv_seqlen > num_blocks * block_size:
            logging("[ERROR] the number of K and V tokens is too big to fit in the paged cache.")
            return False

        if block_size > 128:
            logging("[ERROR] blockSize > 128 is not supported.")
            return False

        if q_seqlen != 1:
            logging("[ERROR] q_seqlen != 1.")
            return False
        selectKvSeqlenMax = (max_kv_seqlen - 1) * compress_stride + compress_block_size
        selectBlockNumMax = (selectKvSeqlenMax + select_block_size - 1) / select_block_size
        if selectBlockNumMax > 4096:
            logging(f"[ERROR] selectBlockNum must be within (0, 4096]. selectBlockNumMax = {selectBlockNumMax}")
            return False
        selectKvSeqlenMin = (min_kv_seqlen - 1) * compress_stride + compress_block_size
        selectBlockNumMin = (selectKvSeqlenMin + select_block_size - 1) // select_block_size
        if selectBlockNumMin < topk + TOPK_CHECK_OUT_SUFFIX:
            return False
        return True

    @classmethod
    def group_matmul(cls, head, kv_head, A, B):
        group_num = head // kv_head
        score = None
        for i in range(kv_head):
            group_score = torch.matmul(A[i * group_num: (i + 1) * group_num, :, :].to(torch.float32),
                                    B[i:(i + 1), :, :].to(torch.float32))
            if score is None:
                score = group_score
            else:
                score = torch.cat((score, group_score), 0)
        return score

    @classmethod
    def softmax_numpy(cls, sim):
        sim = sim.cpu().numpy()
        row_max = np.max(sim, axis=-1, keepdims=True)
        sim_sub = sim - row_max
        sim_sub = np.exp(sim_sub)
        row_sum = np.sum(sim_sub, axis=-1, keepdims=True)
        soft_res = sim_sub / row_sum
        return soft_res

    # pylint:disable = huawei-too-many-arguments
    def ref_masked_attention(self,
                             query,
                             key,
                             value,
                             scale: float,
                             mask,
                             index, compress_block_size, compress_stride, select_block_size, top_k, g
                             ):
        # Q * K.T
        query = query

        query = torch.permute(query, (1, 0, 2))
        key = torch.permute(key, (1, 2, 0))

        sim_high = self.group_matmul(query.shape[0], key.shape[0], query, key)

        sim_high = sim_high * scale
        if mask is not None:
            sim_high = sim_high + (
                    mask[:sim_high.shape[-2], :sim_high.shape[-1]] * self.post_mask_factor
            ).astype(np.float32)

        # softmax
        p_high = self.softmax_numpy(sim_high)
        p = torch.from_numpy(p_high).to(query.dtype)
        p_high = torch.from_numpy(p_high).to(torch.float32)

        value = torch.permute(value, (1, 0, 2))
        out_high = self.group_matmul(query.shape[0], key.shape[0], p_high, value)
        out = self.group_matmul(query.shape[0], key.shape[0], p, value)

        out_high = torch.permute(out_high, (1, 0, 2))
        out = torch.permute(out, (1, 0, 2))
        out = out.to(query.dtype)

        ## 计算importance score
        x = torch.permute(p_high, (1, 0, 2))
        x = x.squeeze(0)
        s1 = x.shape[0]
        s2 = x.shape[1]
        pslc_add = golden_add_gen([s1, s2], compress_block_size, compress_stride, select_block_size, x)
        s2_new = int(np.ceil(((s2 - 1) * compress_stride + compress_block_size) / select_block_size))  # 杨东更改
        pslc = golden_reduce_sum(pslc_add, s2, g, s2_new)

        topk_output, topk_check_output = sort_and_select(pslc.numpy(), top_k)
        # pylint:disable=too-many-return-values
        return out, out_high, p_high, p, sim_high, pslc, topk_output, s2_new, topk_check_output

    # pylint:disable = huawei-too-many-arguments
    def ref_single_query_cached_kv_attention(self, attention_inputs: AttentionInputs, output, true_out, topk_out,
                                             pslc_out, topk_check_out, scale, compress_block_size, compress_stride, select_block_size, top_k, g):
        num_heads = attention_inputs.query.shape[1]
        kv_heads = attention_inputs.value_cache.shape[2]
        head_size_qk = attention_inputs.key_cache.shape[3]
        head_size_vo = attention_inputs.value_cache.shape[3]
        block_size = attention_inputs.value_cache.shape[1]

        batch = len(attention_inputs.q_seqlen_list)
        cu_seqlen = 0
        p_high_out = torch.ones((batch, num_heads, 1, max(attention_inputs.k_seqlen_list)), dtype=torch.float32)
        p_out = torch.ones((batch, num_heads, 1, max(attention_inputs.k_seqlen_list)), dtype=attention_inputs.query.dtype)
        for i in range(batch):
            q_seqlen = int(attention_inputs.q_seqlen_list[i])
            k_seqlen = int(attention_inputs.k_seqlen_list[i])
            q = attention_inputs.query[cu_seqlen:(cu_seqlen + q_seqlen), :, :]
            block_table = attention_inputs.block_tables[i]
            keys = []
            values = []
            for j in range(k_seqlen):  # j 每个k token拼接
                block_number = int(block_table[j // block_size])
                block_offset = j % block_size
                k = attention_inputs.key_cache[block_number, block_offset, :, :]
                k = k.reshape(kv_heads, head_size_qk)
                keys.append(k)

                v = attention_inputs.value_cache[block_number, block_offset, :, :]
                v = v.reshape(kv_heads, head_size_vo)
                values.append(v)
            keys = torch.stack(keys, axis=0)
            values = torch.stack(values, axis=0)
            if attention_inputs.mask_type == 1:
                mask = attention_inputs.global_mask[cu_seqlen:(cu_seqlen + q_seqlen), :]
            else:
                mask = None
            out, out_high, p_high, p, _, pslc, topk_output, s2_new, topk_check_output = self.ref_masked_attention(q, keys, values, scale,
                                                                                              mask, i, compress_block_size, compress_stride, select_block_size, top_k, g)
            p_high_out[i, :, :, :k_seqlen] = p_high
            p_out[i, :, :, :k_seqlen] = p
            out = out.reshape(-1, num_heads, head_size_vo)
            out_high = out_high.reshape(-1, num_heads, head_size_vo)
            pslc = pslc.reshape(-1, kv_heads, s2_new)
            output[cu_seqlen: cu_seqlen + q_seqlen, :, :] = out
            true_out[cu_seqlen: cu_seqlen + q_seqlen, :, :] = out_high
            pslc_out[cu_seqlen: cu_seqlen + q_seqlen, :, :s2_new] = pslc
            topk_out[cu_seqlen: cu_seqlen + q_seqlen, :, :] = topk_output
            topk_check_out[cu_seqlen: cu_seqlen + q_seqlen, :, :] = topk_check_output
            cu_seqlen += attention_inputs.q_seqlen_list[i]

        return scale

    # pylint:disable = huawei-too-many-arguments
    def calc_data(self, gen_data_params: GenDataParams, query, key_cache, value_cache, block_tables, kv_seqlen_list,
                num_head, kv_heads, select_block_size, top_k, compress_block_size, compress_stride, scale, block_size, input_dtype):
        head_size_qk = gen_data_params.head_size + gen_data_params.head_size_rope
        head_size_vo = gen_data_params.head_size
        num_tokens = np.array(gen_data_params.q_seqlen_list).sum()
        batch_size = len(gen_data_params.q_seqlen_list)

        mask = None

        shape_out = (num_tokens, gen_data_params.num_heads, head_size_vo)
        ref_output = torch.zeros(shape_out, dtype=gen_data_params.dtype)
        true_out = torch.zeros(shape_out, dtype=torch.float32)
        s2 = max(kv_seqlen_list)
        s2_new_after_score = int(np.ceil(((s2 - 1) * compress_stride + compress_block_size) / select_block_size))  # s2是k_seq_len
        pslc_out_shape_out = (num_tokens, kv_heads, s2_new_after_score)
        pslc_out = np.zeros(pslc_out_shape_out, dtype=np.float32)

        # tok_k shape
        topk_shape_out = (num_tokens, gen_data_params.kv_heads, top_k)
        topk_out = np.zeros(topk_shape_out, dtype=np.float32)
        # 用于辅助topk校验
        topk_check_shape_out = (num_tokens, gen_data_params.kv_heads, top_k + TOPK_CHECK_OUT_SUFFIX)
        topk_check_out = np.zeros(topk_check_shape_out, dtype=np.float32)

        attention_inputs = self.AttentionInputs(query, key_cache, value_cache, block_tables,
                                                gen_data_params.q_seqlen_list, gen_data_params.k_seqlen_list, mask,
                                                gen_data_params.mask_type)
        g = num_head // kv_heads
        scale = self.ref_single_query_cached_kv_attention(
            attention_inputs,
            ref_output,
            true_out,
            topk_out,
            pslc_out,
            topk_check_out, scale, compress_block_size, compress_stride, select_block_size, top_k, g
        )

        atten_mask = np.zeros((len(gen_data_params.k_seqlen_list), gen_data_params.k_seqlen_list[0]), dtype=np.float32)

        key_cache_reshape = key_cache.reshape(gen_data_params.num_blocks, gen_data_params.block_size, gen_data_params.kv_heads * head_size_qk)

        value_cache_reshape = value_cache.reshape(gen_data_params.num_blocks, gen_data_params.block_size, gen_data_params.kv_heads * head_size_vo)

        # pylint:disable=too-many-return-values
        return (query,
                key_cache_reshape,
                value_cache_reshape,
                torch.from_numpy(np.array(block_tables).astype(np.int32)),
                np.array(gen_data_params.k_seqlen_list).astype(np.int64),
                torch.from_numpy(atten_mask),
                ref_output,
                true_out,
                torch.from_numpy(topk_out.astype(np.int32)),
                torch.from_numpy(topk_check_out.astype(np.int32)),
                scale)


class TestNpuCompressAttentionInfer(TestCase):

    # pylint:disable = huawei-too-many-arguments
    def cpu_op_exec(self, query, key, value, block_table, kv_seqlen_list, num_head, kv_heads, select_block_size, top_k, compress_block_size, compress_stride, scale, block_size, layout):
        input_dtype = query.dtype  # torch.float16 / torch.bfloat16
        # 计算cpu结果
        batch = query.size(0)
        q_seqlen_list = [1] * batch
        num_blocks = key.size(0)
        key = key.reshape(num_blocks, block_size, kv_heads, -1)
        value = value.reshape(num_blocks, block_size, kv_heads, -1)
        embedding_size = value.size(-1)
        embedding_size_rope = key.size(-1) - value.size(-1)
        mask_type = 0
        max_kv_seqlen = max(kv_seqlen_list)
        min_kv_seqlen = min(kv_seqlen_list)

        testObj = TestPagedMLAttention()
        testObj.check_attr(batch, 1, max_kv_seqlen, num_blocks, block_size)
        testObj.check_attr1(batch, 1, max_kv_seqlen, min_kv_seqlen, compress_block_size, compress_stride, select_block_size, num_blocks, block_size, top_k)
        gen_data_params = testObj.GenDataParams(q_seqlen_list, kv_seqlen_list, num_head,
                                kv_heads, embedding_size, embedding_size_rope,
                                num_blocks, block_size, mask_type, input_dtype)

        _, _, _, _, _, _, golden_atten_out, high_percision_atten_out, golden_topk_out, topk_check_out, _ = testObj.calc_data(
            gen_data_params, query, key, value, block_table, kv_seqlen_list,
            num_head, kv_heads, select_block_size, top_k, compress_block_size, compress_stride, scale, block_size, input_dtype)

        return golden_atten_out, high_percision_atten_out, golden_topk_out, topk_check_out

    # pylint:disable = huawei-too-many-arguments
    def npu_op_exec(self, query, key, value, scale_value, head_num, key_value_head_num, select_block_size, select_block_count, page_block_size, compress_block_size, compress_stride, block_table, atten_mask, topk_mask, actual_seq_qlen, actual_cmp_seq_kvlen, actual_sel_seq_kvlen):
        output = torch_npu.npu_nsa_compress_attention_infer(query, key, value, scale_value, head_num, key_value_head_num, 
                                                            select_block_size, select_block_count, page_block_size, 
                                                            compress_block_size, compress_stride, block_table=block_table, 
                                                            atten_mask=None, topk_mask=None, actual_seq_qlen=None, 
                                                            actual_cmp_seq_kvlen=actual_cmp_seq_kvlen, actual_sel_seq_kvlen=None)
        return output

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_nsa_compress_attention_infer(self):
        query = torch.randn([1, 32, 65], dtype=torch.float16)
        key = torch.randn([25, 48, 65], dtype=torch.float16)
        value = torch.randn([25, 48, 18], dtype=torch.float16)
        scale_value = 0.01
        head_num = 32
        key_value_head_num = 1
        select_block_size = 32
        select_block_count = 397
        page_block_size = 48
        compress_block_size = 32
        compress_stride = 16
        block_table = torch.tensor([[23, 2, 20, 22, 4, 21, 7, 12, 3, 20, 20, 0, 15, 0, 4, 8, 10, 20, 21, 18, 18, 18, 11, 12, 20]]).int()
        actual_cmp_seq_kvlen = [1180]
        atten_mask = None
        topk_mask = None
        actual_seq_qlen = None
        actual_sel_seq_kvlen = None


        npuout = self.npu_op_exec(query.npu(), key.npu(), value.npu(), scale_value, head_num, key_value_head_num, 
                                  select_block_size, select_block_count, page_block_size, compress_block_size, 
                                  compress_stride, block_table.npu(), atten_mask, topk_mask, 
                                  actual_seq_qlen, actual_cmp_seq_kvlen, actual_sel_seq_kvlen)

        golden_atten_out, _, _, _ = self.cpu_op_exec(query, key, value, 
                                                                                                       block_table, actual_cmp_seq_kvlen, 
                                                                                                       head_num, key_value_head_num, 
                                                                                                       select_block_size, select_block_count, 
                                                                                                       compress_block_size, compress_stride, 
                                                                                                       scale_value, page_block_size, layout='TND')

        self.assertRtolEqual(golden_atten_out.numpy(), npuout[0].cpu().numpy(), 0.001, 0.001)

if __name__ == "__main__":
    run_tests()

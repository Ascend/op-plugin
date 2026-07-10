#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

"""
测试 ATB SelfAttention V3 算子（_npu_flash_attention_v3）

V3 与 V1 的核心区别：
  - V1：使用 MASK_TYPE_NORM（float16 格式 mask，-10000 表示遮住）
  - V3：使用 MASK_TYPE_NORM_COMPRESS（float16 格式压缩 mask，-10000 表示遮住，0=可见）
  - V3 新增 mask_type 参数，支持指定压缩 mask 类型
  - V3 mask 尺寸为 [1， 128， 2048， 16]，格式为NZ，4D

测试流程：
  1. 生成随机 Q/K/V 数据和 float16 因果 mask（[2048, 2048]）
  2. 用 numpy 手动计算 Self-Attention 作为 golden 参考输出
  3. 调用 NPU 上的 _npu_flash_attention_v3 算子
  4. 对比 NPU 输出与 golden 输出，验证精度（assertRtolEqual）

仅支持 Ascend 310P 设备
"""

import math
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestFAV3(TestCase):

    def gen_seq_len(self, batch, max_seq, variate_seq=False):
        """
        生成每个 batch 的序列长度

        Args:
            batch: batch 大小
            max_seq: 最大序列长度
            variate_seq: 是否生成可变长度序列

        Returns:
            seqlen: 实际序列长度数组，shape=[batch]
            seqlen_aligned: 对齐到 16 的序列长度（ATB 算子要求对齐）
            ntokens: 所有序列的总 token 数
        """
        if variate_seq:
            # 可变长度模式：生成不同长度的序列，对齐到 16 的倍数
            num = max_seq // 16
            seqlen_aligned_arange = np.arange(1, num) * 16  # [16, 32, 48, ...]
            if batch > num:
                # batch 数量超过预设长度，随机补充剩余
                seqlen_aligned_remain = np.random.randint(1, max_seq, size=(batch - num))
                seqlen_aligned_remain[:] = ((seqlen_aligned_remain[:] + 15) // 16) * 16
                seqlen_aligned = np.concatenate((seqlen_aligned_arange, seqlen_aligned_remain), 0)
            else:
                seqlen_aligned = seqlen_aligned_arange
            # 在对齐长度基础上随机减少 0~15，得到实际长度
            sp_list = np.random.randint(0, 15, size=(num - 1))
            seqlen = seqlen_aligned - sp_list
            # 取最后 batch 个序列
            seqlen = seqlen[-batch:]
            seqlen_aligned = seqlen_aligned[-batch:]
        else:
            # 固定长度模式：所有序列都是 max_seq
            max_seq_aligned = (max_seq + 15) // 16 * 16
            sp_list = np.ones((batch,)) * (max_seq_aligned - max_seq)
            sp_list = sp_list.astype(np.int32)
            seqlen = np.ones((batch,)) * max_seq
            seqlen = seqlen.astype(np.int32)
            seqlen_aligned = np.ones((batch,)) * max_seq_aligned
            seqlen_aligned = seqlen_aligned.astype(np.int32)
        ntokens = seqlen.sum()
        return seqlen, seqlen_aligned, ntokens

    def group_matmul(self, heads, group_num, A, B):
        """
        分组矩阵乘法，支持 GQA（Grouped Query Attention）

        当 heads != group_num 时（如 heads=32, group_num=8），
        多个 query head 共享同一个 kv head。

        Args:
            heads: query head 数量
            group_num: kv head 数量（分组数）
            A: 左矩阵，shape=[heads, seq_q, dim]
            B: 右矩阵，shape=[group_num, seq_kv, dim] 或 [group_num, dim, seq_kv]

        Returns:
            结果矩阵，shape=[heads, seq_q, seq_kv] 或 [heads, seq_q, dim]
        """
        group_head = heads // group_num  # 每组包含的 query head 数
        score = None
        for i in range(group_num):
            # 第 i 组：group_head 个 query head 与第 i 个 kv head 做矩阵乘法
            group_score = np.matmul(A[i * group_head: (i + 1) * group_head, :, :].astype(np.float32),
                                    B[i:(i + 1), :, :].astype(np.float32)).astype(np.float16)
            if score is None:
                score = group_score
            else:
                score = np.concatenate((score, group_score), 0)
        return score

    def calc_expect_func(self, batch, seqlen, heads, embed, group_num=32):
        """
        用 numpy 手动计算 Self-Attention，作为 golden 参考输出

        计算流程（逐 batch）：
          Q·K^T → 缩放 → 加 mask → Softmax → ·V → 输出

        Args:
            batch: batch 大小
            seqlen: 每个序列的长度
            heads: query head 数量
            embed: 每个 head 的维度（head_dim）
            group_num: kv head 数量（GQA 分组数）

        Returns:
            (q, k, v, mask, q_len, tor, heads, out_expect) 元组
        """
        is_mask = True         # 是否使用因果 mask
        variate_seq = False    # 是否使用可变序列长度
        is_decoder = False     # 是否是 decoder 模式（q 和 kv 长度不同）
        max_seq = 2048         # mask 矩阵的最大尺寸
        src_type = 'float16'   # 数据类型
        fp32 = True            # softmax 是否在 fp32 下计算

        # 根据是否 decoder 模式，生成 q 和 kv 的序列长度
        if is_decoder:
            q_seqlen, q_seqlen_aligned, q_ntokens = self.gen_seq_len(batch, 1, variate_seq)
            kv_seqlen, kv_seqlen_aligned, kv_ntokens = self.gen_seq_len(batch, seqlen, variate_seq)
        else:
            # 非 decoder 模式：q 和 kv 长度相同
            q_seqlen, q_seqlen_aligned, q_ntokens = self.gen_seq_len(batch, seqlen, variate_seq)
            kv_seqlen, kv_seqlen_aligned, kv_ntokens = q_seqlen, q_seqlen_aligned, q_ntokens

        max_s = np.max(q_seqlen)   # 最大序列长度，决定 mask 尺寸
        ntokens2 = (q_seqlen * kv_seqlen).sum()
        embed_v = embed            # value 的 head dim 与 key 相同

        # 生成随机 Q、K、V 数据，shape 为 [total_tokens, heads * embed]
        q = np.random.uniform(-1.0, 1.0, size=(q_ntokens, heads * embed)).astype(np.float16)
        k = np.random.uniform(-1.0, 1.0, size=(kv_ntokens, group_num * embed)).astype(np.float16)
        v = np.random.uniform(-1.0, 1.0, size=(kv_ntokens, group_num * embed_v)).astype(np.float16)

        # 生成 float16 因果 mask（上三角矩阵）
        # SelfAttention MASK_TYPE_NORM_COMPRESS 要求 2D float16，尺寸需要覆盖所有 token
        # mask 尺寸用 max_seq（2048）而不是 max_s（实际最大序列长度）
        # 0 = 可见（可以 attend），-10000 = 被遮住（不能 attend）
        mask = np.ones(shape=(max_seq, max_seq)).astype(np.float16)
        mask = np.triu(mask, 1)  # 上三角为 1，对角线及以下为 0
        mask = mask * np.float16(-10000.0)  # 上三角为 -10000（被遮住），对角线及以下为 0（可见）

        # 逐 batch 计算 Self-Attention
        q_offset = 0
        k_offset = 0
        v_offset = 0

        s = None    # 原始 attention score（未缩放）
        _p = None   # softmax 概率
        out = None  # 最终输出

        for idx in range(batch):
            q_s = q_seqlen[idx]     # 当前 batch 的 query 序列长度
            kv_s = kv_seqlen[idx]   # 当前 batch 的 kv 序列长度

            # 取出当前 batch 的 Q 切片，reshape 为 [heads, q_s, embed]
            q_slice = q[q_offset:q_offset + q_s][:]
            q_slice = q_slice.reshape(q_s, heads, embed)
            q_slice = np.transpose(q_slice, (1, 0, 2))  # [q_s, heads, embed] → [heads, q_s, embed]

            # 取出当前 batch 的 K 切片，reshape 为 [group_num, kv_s, embed]
            k_slice = k[k_offset:k_offset + kv_s][:]
            k_slice = k_slice.reshape(kv_s, group_num, embed)
            k_slice = np.transpose(k_slice, (1, 0, 2))   # [kv_s, group_num, embed] → [group_num, kv_s, embed]
            k_slice_t = np.transpose(k_slice, (0, 2, 1))  # [group_num, kv_s, embed] → [group_num, embed, kv_s]（转置用于 Q·K^T）

            # 取出当前 batch 的 V 切片，reshape 为 [group_num, kv_s, embed_v]
            v_slice = v[v_offset:v_offset + kv_s][:]
            v_slice = v_slice.reshape(kv_s, group_num, embed_v)
            v_slice = np.transpose(v_slice, (1, 0, 2))   # [kv_s, group_num, embed_v] → [group_num, kv_s, embed_v]

            # 第1步：计算 Q·K^T（分组矩阵乘法，支持 GQA）
            score = self.group_matmul(heads, group_num, q_slice, k_slice_t)
            # score shape: [heads, q_s, kv_s]

            if s is None:
                s = score.reshape([-1, ])
            else:
                s = np.concatenate((s, score.reshape([-1, ])), 0)

            # 第2步：缩放（除以 sqrt(head_dim)）
            tor = np.float16(1.0 / math.sqrt(1.0 * embed))
            score = score * tor

            # 第3步：应用因果 mask
            # float16 mask：0=可见，-10000=被遮住，直接加到 score 上
            if is_mask:
                mask_pb = mask[:q_s, :kv_s]               # 取出当前序列对应的 mask 子矩阵
                mask_pb = np.expand_dims(mask_pb, axis=0)  # 增加 head 维度
                mask_pb = np.repeat(mask_pb, repeats=heads, axis=0)  # 复制到所有 head
                score = score + mask_pb                    # 直接加上 mask 值

            # 第4步：Softmax（数值稳定版本：减去最大值后再 exp）
            score_max = np.max(score, axis=-1)
            score = score - score_max.reshape((heads, q_s, 1))
            score_exp = np.exp(score.astype(np.float32))

            if not fp32:
                # fp16 Softmax：除法在 fp16 下进行（精度较低）
                score_sum = np.sum(score_exp.astype(np.float16), axis=-1)
                if _p is None:
                    _p = score_exp.astype(np.float16).reshape([-1, ])
                else:
                    _p = np.concatenate((_p, score_exp.astype(np.float16).reshape([-1, ])), 0)
                p = score_exp.astype(np.float16) / score_sum.reshape((heads, q_s, 1)).astype(np.float16)
                out_sub = self.group_matmul(heads, group_num, p, v_slice)
            else:
                # fp32 Softmax：除法在 fp32 下计算后再转回 fp16（精度更高）
                score_sum = np.sum(score_exp, axis=-1)
                if _p is None:
                    _p = score_exp.astype(np.float16).reshape([-1, ])
                else:
                    _p = np.concatenate((_p, score_exp.astype(np.float16).reshape([-1, ])), 0)
                p = score_exp.astype(np.float16)
                out_sub = self.group_matmul(heads, group_num, p, v_slice)
                out_sub = out_sub / score_sum.reshape((heads, q_s, 1)).astype(np.float16)

            # 第5步：输出 reshape 为 [q_s, heads, embed_v]
            out_sub = out_sub.reshape(heads, q_s, embed_v)
            out_sub = np.transpose(out_sub, (1, 0, 2))  # [heads, q_s, embed_v] → [q_s, heads, embed_v]
            out_sub = np.ascontiguousarray(out_sub)

            # 拼接所有 batch 的输出
            if out is None:
                out = out_sub
            else:
                out = np.concatenate((out, out_sub), 0)

            # 移动偏移量到下一个 batch
            q_offset += q_s
            k_offset += kv_s
            v_offset += kv_s

        # 将数据整理为算子输入格式
        q = q.astype(src_type).reshape(-1, heads, embed)       # [total_tokens, heads, embed]
        k = k.astype(src_type).reshape(-1, group_num, embed)   # [total_tokens, kv_heads, embed]
        v = v.astype(src_type).reshape(-1, group_num, embed_v) # [total_tokens, kv_heads, embed_v]
        def round_up(x, align):
            if align == 0:
                return -1
            return (x + align - 1) // align * align


        def custom_pad(x, pad_dims):
            return torch.nn.functional.pad(x, pad_dims)


        def custom_reshape(x, target_shape):
            return x.reshape(target_shape)


        def custom_transpose(x, dim1, dim2):
            return x.transpose(dim1, dim2)


        def nd_to_nz_2d(in_tensor):
            aux_dims = [0, 0, 0, 0]
            aux_dims[0] = 1
            aux_dims[1] = round_up(in_tensor.size(0), 16)

            pad_dims = [0, 0, 0, 0]
            pad_dims[3] = round_up(in_tensor.size(0), 16) - in_tensor.size(0)

            aux_dims[2] = round_up(in_tensor.size(1), 16) // 16
            aux_dims[3] = 16
            pad_dims[1] = round_up(in_tensor.size(1), 16) - in_tensor.size(1)

            return custom_transpose(
                custom_reshape(custom_pad(in_tensor, pad_dims), aux_dims), 1,
                2).contiguous()

        mask_2048 = np.zeros((2048, 2048), dtype=np.float16)
        # 构造严格上三角（不包括对角线）的 mask，赋值为 -inf
        mask_2048[np.triu_indices(2048, k=1)] = 1
        mask_2048 *= -60000
        mask_2048 = mask_2048.astype('float16').reshape(2048, 2048)
        mask_nz = nd_to_nz_2d(torch.from_numpy(mask_2048)).npu()
        mask_nz = torch_npu.npu_format_cast(mask_nz, 29)

        q_len = q_seqlen.astype(np.int32)                       # 每个序列的长度
        out_expect = out.astype(src_type).reshape(-1, heads, embed_v)  # [total_tokens, heads, embed_v]

        # 返回：q, k, v, mask, seq_len, scale, heads, golden_output
        ret_data = q, k, v, mask_nz, q_len, tor, heads, out_expect
        return ret_data

    @SupportedDevices(['Ascend310P'])
    def test_flash_attention_v3(self):
        """
        测试 _npu_flash_attention_v3 算子（默认 mask_type）

        参数：
          - batch=16, seqlen=128, heads=32, embed=128, kv_heads=32
          - 默认 mask_type=3（MASK_TYPE_NORM_COMPRESS，int8 压缩 mask）

        验证方式：NPU 算子输出 vs numpy golden 输出，通过 assertRtolEqual 比较
        """
        kv_head = 32
        data = self.calc_expect_func(16, 128, 32, 128, group_num=kv_head)

        # 将 numpy 数组转为 torch tensor
        in_tensors = []
        for tensor in data:
            if isinstance(tensor, np.ndarray):
                in_tensors.append(torch.from_numpy(tensor))
            elif torch.is_tensor(tensor):
                in_tensors.append(tensor)
            else:
                in_tensors.append(torch.tensor(tensor))

        query = in_tensors[0].npu()    # [total_tokens, heads, embed]
        key = in_tensors[1].npu()      # [total_tokens, kv_heads, embed]
        value = in_tensors[2].npu()    # [total_tokens, kv_heads, embed_v]
        # 310P 的 NORM_COMPRESS 长 mask 需要保持 NPU FRACTAL_NZ 4D 格式。
        mask = in_tensors[3]
        seq_len = in_tensors[4].cpu()  # 序列长度（在 CPU 上）
        tor = data[5]              # 缩放因子 1/sqrt(head_dim)
        heads = data[6]            # head 数量
        group_num = kv_head        # kv head 数量
        cal_out = in_tensors[7]    # numpy golden 输出
        out = torch.empty_like(in_tensors[7]).npu()  # NPU 输出缓冲区

        # 调用 V3 算子（默认 mask_type，使用 NORM_COMPRESS）
        torch_npu._npu_flash_attention_v3(query, key, value, mask, seq_len, tor, heads, group_num, out=out)

        # 对比 NPU 输出与 golden 输出
        self.assertRtolEqual(cal_out, out)

    @SupportedDevices(['Ascend310P'])
    def test_flash_attention_v3_with_mask_type(self):
        """
        测试 _npu_flash_attention_v3 算子（显式指定 mask_type=3）

        与 test_flash_attention_v3 相同，但显式传入 mask_type=3
        验证 mask_type 参数的传递是否正确

        mask_type=3 对应 SelfAttentionParam 中的 MASK_TYPE_NORM_COMPRESS
        注意：SelfAttentionParam NORM_COMPRESS=3，PagedAttentionParam NORM_COMPRESS=5，两个枚举值不同
        """
        kv_head = 32
        data = self.calc_expect_func(16, 128, 32, 128, group_num=kv_head)

        in_tensors = []
        for tensor in data:
            if isinstance(tensor, np.ndarray):
                in_tensors.append(torch.from_numpy(tensor))
            elif torch.is_tensor(tensor):
                in_tensors.append(tensor)
            else:
                in_tensors.append(torch.tensor(tensor))

        query = in_tensors[0].npu()
        key = in_tensors[1].npu()
        value = in_tensors[2].npu()
        mask = in_tensors[3]
        seq_len = in_tensors[4].cpu()
        tor = data[5]
        heads = data[6]
        group_num = kv_head
        cal_out = in_tensors[7]
        out = torch.empty_like(in_tensors[7]).npu()

        # 调用 V3 算子，显式指定 mask_type=3（SelfAttentionParam::MASK_TYPE_NORM_COMPRESS）
        torch_npu._npu_flash_attention_v3(query, key, value, mask, seq_len, tor, heads, group_num,
                                          mask_type=3, out=out)

        self.assertRtolEqual(cal_out, out)


if __name__ == '__main__':
    run_tests()

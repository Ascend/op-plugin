import math
import unittest
import random
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestMLA(TestCase):
    def shape_nd_to_nz(self, shape, dtype='torch.float16'):
        assert len(shape) >= 2
        batch = shape[:-2]  # 最后两维nd->nz
        a, b = shape[-2], shape[-1]
        if dtype != torch.int8:
            a0, b0 = 16, 16
        else:
            a0, b0 = 16, 32
        return list(batch) + [math.ceil(b / b0), math.ceil(a / a0), a0, b0]

    def gen_axes_for_transpose(self, offset, base):
        return [x for x in range(offset)] + [x + offset for x in base]

    def convert_nd_to_nz(self, x):
        array_trans = self.gen_axes_for_transpose(
            len(x.shape) - 2, [2, 0, 1, 3])  # (m1, m0, n1, n0) -> (n1, m1, m0, n0)
        x_shape = self.shape_nd_to_nz(x.shape, dtype=x.dtype)
        *_, n1, m1, m0, n0 = x_shape
        # x原始需要对齐，才能reshape
        return x.reshape(x_shape[:-4] + [m1, m0, n1, n0]).permute(*array_trans)

    def compare_output_data(self, out, golden, ratios):
        error_count = 0
        strict_error_count = 0
        fp16_min_normal = 1.0 / (1 << 14)
        golden = golden.flatten().to(torch.float32)
        out = out.flatten().to(torch.float32)
        out_len = out.shape[0]
        diff = torch.abs(golden - out)
        max_diff = diff.max().item()
        limit_error = torch.maximum(
            torch.abs(golden * ratios[0]), torch.tensor(ratios[1]))
        strict_limit_error = torch.maximum(
            torch.abs(golden * ratios[2]), torch.tensor(ratios[3]))
        error_count = torch.gt(diff, limit_error).sum().item()
        strict_error_count = torch.gt(diff, strict_limit_error).sum().item()
        print(f"maxDiff {max_diff}")
        print("1/1000 Accuracy is %f", 1 - float(error_count) / out_len)
        print("5/1000 Accuracy is %f", 1 - float(strict_error_count) / out_len)
        if self.data_type == torch.bfloat16 or self.is_int8_flag:
            print("accuracy is correct in old standard: %r",
                  (float(strict_error_count) / out_len) <= ratios[2])
        else:
            print("accuracy is correct in old standard: %r",
                  (float(strict_error_count) / out_len) <= ratios[0])
        calc_times = self.head_size_qk * self.max_context_len + 4
        if self.data_type == torch.bfloat16:
            if calc_times < 2048:
                error = 2**(-7)
            else:
                error = 2**(-6)
            error_threshold = torch.clamp(torch.abs(golden), min=1) * error
            res = (diff <= error_threshold).all().item()
            print("accuracy is correct in new standard: %r", res)
            return res
        else:
            if calc_times < 2048:
                error = 2**(-8)
            else:
                error = 2**(-7)
            error_threshold = torch.clamp(torch.abs(golden), min=1) * error
            res = (diff <= error_threshold).all().item()
            print("accuracy is correct in new standard: %r", res)
            return res

    def get_alibi_slopes(self, n_heads):
        n = 2 ** math.floor(math.log2(n_heads))
        m0 = 2.0 ** (-8.0 / n)
        slopes = torch.pow(m0, torch.arange(1, n + 1))
        if n < n_heads:
            m1 = 2.0 ** (-4.0 / n)
            mm = torch.pow(m1, torch.arange(1, 1 + 2 * (n_heads - n), 2))
            slopes = torch.cat([slopes, mm])
        return slopes

    def group_mm_torch(self, heads, group_num, A, B, is_k):
        group_head = heads // group_num
        score_high = None
        for i in range(group_num):
            if self.is_int8_flag:
                int8_B = B[i: (i + 1), :, :, ]
                head_dim = int8_B.shape[2]
                int32_B = torch.matmul(torch.eye(int8_B.shape[1]).to(
                    torch.float32), int8_B.to(torch.float32)).to(torch.int32)
                if is_k:
                    if self.has_bias:
                        int32_B = int32_B + \
                            self.offset1[i * head_dim:(i + 1) * head_dim]
                    fp32_B = int32_B.to(
                        torch.float32) * self.de_scale1_fp32[i * head_dim:(i + 1) * head_dim]
                    fp32_B = torch.permute(fp32_B, (0, 2, 1))
                else:
                    if self.has_bias:
                        int32_B = int32_B + \
                            self.offset2[i * head_dim:(i + 1) * head_dim]
                    fp32_B = int32_B.to(
                        torch.float32) * self.de_scale2_fp32[i * head_dim:(i + 1) * head_dim]
                group_score_high = torch.matmul(A[i * group_head: (i + 1) * group_head, :, :].to(torch.float32),
                                                fp32_B)
            elif self.is_quant_flag:
                group_score_int32 = torch.matmul(A[i * group_head: (i + 1) * group_head, :, :].to(torch.int32),
                                                 B[i: (i + 1), :, :].to(torch.int32)).to(torch.int32)
                if is_k:
                    group_score_high = group_score_int32.to(torch.float32) * self.de_scale1_fp32[(
                        i * group_head): (i + 1) * group_head].reshape(group_head, 1, 1).to(torch.float32)
                else:
                    group_score_high = group_score_int32.to(torch.float32) * self.de_scalev[(
                        i * group_head): (i + 1) * group_head].reshape(group_head, 1, 1).to(torch.float32)
            else:
                group_score_high = torch.matmul(A[i * group_head: (i + 1) * group_head, :, :].to(torch.float32),
                                                B[i: (i + 1), :, :].to(torch.float32))
            if score_high is None:
                score_high = group_score_high
            else:
                score_high = torch.cat((score_high, group_score_high), 0)
        return score_high

    def process_deq_scale(self, deq_scale) -> np.ndarray:
        new_deq_scale = np.frombuffer(deq_scale.tobytes(), dtype=np.uint32)
        return new_deq_scale.astype(np.uint64)

    def softmax(self, sim):
        row_max = torch.max(sim, axis=-1, keepdims=True)[0]
        sim_sub = sim - row_max
        sim_sub = torch.exp(sim_sub)
        row_sum = torch.sum(sim_sub, axis=-1, keepdims=True)
        soft_res = sim_sub / row_sum
        return soft_res

    def softmax_numpy(self, sim):
        sim = sim.cpu().numpy()
        row_max = np.max(sim, axis=-1, keepdims=True)
        sim_sub = sim - row_max
        sim_sub = np.exp(sim_sub)
        row_sum = np.sum(sim_sub, axis=-1, keepdims=True)
        soft_res = sim_sub / row_sum
        return soft_res

    def softmax_quant_numpy(self, sim, is_first):
        lm = np.max(sim, axis=-1, keepdims=True)
        if is_first:
            hm = lm
            self.dm = 0
        else:
            hm = np.maximum(self.gm, lm)
            self.dm = self.gm - hm
        self.gm = hm
        sim_sub = sim - hm
        sim_sub = np.exp(sim_sub)
        row_sum = np.sum(sim_sub, axis=-1, keepdims=True)
        row_maxp = np.max(sim_sub, axis=-1, keepdims=True)
        scale = row_maxp.astype("float32") / 127.0
        sim_int8 = sim_sub / scale
        soft_res = sim_int8.astype("float16")
        soft_res = np.rint(soft_res).astype("int8")
        de_scalev = self.de_scale2_fp32 * row_maxp[:, 0, 0] / 127
        return soft_res, row_sum, de_scalev, hm, self.dm

    def softmax_quant_numpy_online(self, sim, heads, kv_head, value):
        group_head = heads // kv_head
        score_high = None
        kv_seqlen = value.shape[1]
        cur_kv_seqlen = kv_seqlen
        n_loop = (cur_kv_seqlen + self.block_size_calc -
                  1) // self.block_size_calc
        qk_n = self.block_size_calc
        self.tmp_l_list = []
        self.tmp_o_list = []
        for cur_nIndx in range(self.kvsplit):
            kv_seqlen_align = (kv_seqlen + self.block_size -
                               1) // self.block_size * self.block_size
            start_kv = cur_nIndx * self.kv_split_per_core
            cur_kv_seqlen = self.kv_split_per_core
            kv_loop = (kv_seqlen_align + self.kv_split_per_core -
                       1) // self.kv_split_per_core
            if cur_nIndx >= kv_loop:
                continue
            if cur_nIndx == (kv_loop - 1):
                cur_kv_seqlen = kv_seqlen - cur_nIndx * self.kv_split_per_core
            n_loop = (cur_kv_seqlen + self.block_size_calc -
                      1) // self.block_size_calc
            qk_n = self.block_size_calc
            end_kv = start_kv
            for n_idx in range(n_loop):
                is_first = (n_idx == 0)
                if n_idx == n_loop - 1:
                    qk_n = cur_kv_seqlen - n_idx * self.block_size_calc
                end_kv = end_kv + qk_n
                sim_block = sim[:, :, start_kv: end_kv]
                p_block, ll, de_scalev, hm, dm = self.softmax_quant_numpy(sim_block, is_first)
                self.de_scalev = de_scalev
                value_block = value[:, start_kv: end_kv, :]
                lo = self.group_mm_torch(
                    heads, kv_head, torch.from_numpy(p_block), value_block, 0)
                lo = lo.cpu().numpy()
                if n_idx == 0:
                    self.gl = ll
                    self.go = lo
                else:
                    dm = np.exp(dm)
                    self.gl = self.gl * dm
                    self.gl = self.gl + ll
                    self.go = self.go * dm
                    self.go = self.go + lo
                start_kv = start_kv + qk_n
            self.go = self.go / self.gl
            self.tmp_o_list.append(self.go.reshape(
                [1, self.num_heads, 1, value.shape[2]]))
            ls = np.log(self.gl) + self.gm
            self.tmp_l_list.append(ls.reshape([1, self.num_heads]))
        if self.kvsplit > 1:
            l_list = np.concatenate(self.tmp_l_list, 0)
            o_list = np.concatenate(self.tmp_o_list, 0)
            l_list = np.transpose(l_list, (1, 0))
            lse_max = np.max(l_list, axis=1, keepdims=True)
            l_tmp = np.exp(l_list - lse_max)
            lse_sum = np.sum(l_tmp, axis=1, keepdims=True)
            lse_logsum = np.log(lse_sum) + lse_max
            scale = np.exp(l_list - lse_logsum)
            o_list = o_list * scale.transpose(1, 0)[:, :, np.newaxis, np.newaxis]
            self.go = np.sum(o_list, axis=0, keepdims=True)
            self.go = np.squeeze(self.go, axis=0)
        return torch.from_numpy(self.go)

    def ref_masked_attention(self,
                             query,  # (1, num_heads, head_size)
                             key,  # (context_len, kv_heads, head_size)
                             value,
                             scale: float,
                             alibi_bias,
                             mask_data_type=torch.bfloat16,
                             query_rope=None,
                             key_rope=None
                             ):
        # Q * K.T
        query = query
        query = torch.permute(query, (1, 0, 2))
        if self.is_quant_flag:
            query_rope = torch.permute(query_rope, (1, 0, 2))
            key_rope = torch.permute(key_rope, (1, 2, 0))
        if not self.is_int8_flag:
            key = torch.permute(key, (1, 2, 0))  # 0 1 2
        else:
            key = torch.permute(key, (1, 0, 2))
        sim_high = self.group_mm_torch(
            query.shape[0], key.shape[0], query, key, 1)
        if self.is_quant_flag:
            self.is_quant_flag = False
            sim_high_rope = self.group_mm_torch(
                query_rope.shape[0], key_rope.shape[0], query_rope, key_rope, 0)
            self.is_quant_flag = True
            sim_high = sim_high + sim_high_rope
        sim_out = sim_high.to(torch.float32)
        sim_high = sim_high.to(torch.float32) * scale
        if alibi_bias is not None:
            sim_high = sim_high + alibi_bias.to(torch.float32)
        if self.is_quant_flag:
            self.gm = np.full([query.shape[0], 1, 1],
                              np.finfo(np.float32).min)
            p_high, row_sum, de_scalev, _, _ = self.softmax_quant_numpy(
                sim_high.numpy(), 1)
            self.de_scalev = de_scalev
            value = torch.permute(value, (1, 0, 2))
            out_high = self.group_mm_torch(
                query.shape[0], key.shape[0], torch.from_numpy(p_high), value, 0)
            out_high = out_high / row_sum
            out_high = torch.permute(out_high, (1, 0, 2))
            s_qk = sim_high.numpy()
            out = self.softmax_quant_numpy_online(
                s_qk, query.shape[0], key.shape[0], value)
        else:
            # softmax
            p_high = self.softmax_numpy(sim_high)
            p = torch.from_numpy(p_high).to(mask_data_type)
            p_high = torch.from_numpy(p_high)

            # P * V
            value = torch.permute(value, (1, 0, 2))
            out = self.group_mm_torch(
                query.shape[0], key.shape[0], p, value, 0)
            out_high = self.group_mm_torch(
                query.shape[0], key.shape[0], p_high, value, 0)
            out = torch.permute(out, (1, 0, 2))
            out_high = torch.permute(out_high, (1, 0, 2))
        sim_out = torch.permute(sim_out, (1, 0, 2))
        return out, out_high, sim_out

    def ref_single_query_cached_kv_attention(self,
                                             sim,
                                             output,
                                             true_out,
                                             query,
                                             key_cache,
                                             value_cache,
                                             block_tables,
                                             context_lens,
                                             mask,
                                             mask_dim=4,
                                             mask_data_type=torch.bfloat16,
                                             query_rope=None,
                                             key_cache_rope=None
                                             ) -> None:
        mask_index_coff = 1
        if self.compressHead:
            query = query.view(self.num_tokens * self.kv_heads,
                               self.num_heads // self.kv_heads, self.head_size_qk)
            output = output.view(self.num_tokens * self.kv_heads,
                                 self.num_heads // self.kv_heads, self.head_size_vo)
            true_out = true_out.view(
                self.num_tokens * self.kv_heads, self.num_heads // self.kv_heads, self.head_size_vo)
            if mask_dim == 4:
                mask_shape = mask.shape
                mask = mask.view(
                    mask_shape[0] * self.kv_heads, self.num_heads // self.kv_heads, 1, self.max_context_len)
            else:
                mask_index_coff = self.kv_heads
        num_heads = query.shape[1]
        kv_heads = value_cache.shape[2]
        head_size_qk = key_cache.shape[3]
        head_size_vo = value_cache.shape[3]
        block_size = value_cache.shape[1]

        num_input_tokens = query.shape[0]
        index = 0
        q_rope = None
        for i in range(len(context_lens)):
            block_table = block_tables[i]
            context_len = int(context_lens[i])
            if context_len == 0:
                continue

            q = query[index].view(1, num_heads, head_size_qk)
            if self.is_quant_flag:
                q_rope = query_rope[index].view(1, num_heads, 64)
            keys = []
            values = []
            keys_rope = []
            for j in range(context_len):
                block_number = int(block_table[j // block_size])
                block_offset = j % block_size

                k = key_cache[block_number, block_offset, :, :]
                k = k.reshape(kv_heads, head_size_qk)
                keys.append(k)
                if self.is_quant_flag:
                    k_rope = key_cache_rope[block_number, block_offset, :, :]
                    k_rope = k_rope.reshape(kv_heads, 64)
                    keys_rope.append(k_rope)

                v = value_cache[block_number, block_offset, :, :]
                v = v.reshape(kv_heads, head_size_vo)
                values.append(v)
            keys = torch.stack(keys, axis=0)
            if self.is_quant_flag:
                keys_rope = torch.stack(keys_rope, axis=0)
            values = torch.stack(values, axis=0)
            scale = np.float32(1.0 / (head_size_qk ** 0.5))
            if mask_dim == 4:
                out, out_high, sim_out = self.ref_masked_attention(
                    q, keys, values, scale, mask[i, :, :, :context_len], mask_data_type, q_rope, keys_rope)
                out = out.reshape(num_heads, head_size_vo)
            elif mask_dim == 3:
                out, out_high, sim_out = self.ref_masked_attention(
                    q, keys, values, scale, mask[i // mask_index_coff, :, :context_len], mask_data_type, q_rope, keys_rope)
                out = out.reshape(num_heads, head_size_vo)
            else:
                out, out_high, sim_out = self.ref_masked_attention(
                    q, keys, values, scale, mask, mask_data_type, q_rope, keys_rope)
                out = out.reshape(num_heads, head_size_vo)
            out_high = out_high.reshape(num_heads, head_size_vo)
            sim_out = sim_out.reshape(1, num_heads * context_len)
            output[index] = out.to(mask_data_type)
            true_out[index] = out_high
            sim[index] = sim_out
            index = index + 1

    def CalcuHeadSplitNd(self, embeddingSize, embeddingSizeV):
        if self.former_head <= 64:
            embedQKSplit = 512 if embeddingSize > 512 else embeddingSize
            embedVOSplit = 512 if embeddingSizeV > 512 else embeddingSizeV
        else:
            embedQKSplit = 512 if embeddingSize > 512 else embeddingSize
            embedVOSplit = 256 if embeddingSizeV > 256 else embeddingSizeV
        return embedQKSplit, embedVOSplit

    def get_blockszie_calc(self, max_context_len, block_size, embeddingSize, embeddingSizeV):
        embedQKSplit = self.embedQKSplit
        embedVOSplit = self.embedVOSplit
        BLOCK_LIMIT = 128 * 128
        KV_SEQLEN_SLICE = 128
        KV_SEQLEN_SLICE_256 = 256
        KV_SEQLEN_SLICE_512 = 512
        BLOCK_LIMIT_NO_PPONG = 128 * 256
        BLOCK_LIMIT_NO_PPONG_UINT8 = 128 * 256 * 2
        block_size_calc = block_size
        headdimMax = np.maximum(embedQKSplit, embedVOSplit)
        if self.is_quant_flag:
            tBlockAlign = 32
        else:
            tBlockAlign = 16
        l0Limit = tBlockAlign * BLOCK_LIMIT_NO_PPONG_UINT8 / 32
        if block_size <= KV_SEQLEN_SLICE / 2 and \
                block_size * 2 * embedQKSplit <= BLOCK_LIMIT and \
                block_size * 2 * embedVOSplit <= BLOCK_LIMIT:
            block_size_calc = block_size * 2
        if not self.is_int8_flag and \
                max_context_len >= KV_SEQLEN_SLICE_256 and \
                self.kv_split_per_core >= KV_SEQLEN_SLICE_256 and \
                self.former_head <= BLOCK_LIMIT / KV_SEQLEN_SLICE_256 and \
                KV_SEQLEN_SLICE_256 * embedQKSplit <= l0Limit and \
                KV_SEQLEN_SLICE_256 * embedVOSplit <= l0Limit and \
                (block_size == KV_SEQLEN_SLICE_256 // 4 or block_size == KV_SEQLEN_SLICE_256 // 2):
            block_size_calc = 256
        if self.is_quant_flag and \
            max_context_len >= KV_SEQLEN_SLICE_512 and \
            self.kv_split_per_core >= KV_SEQLEN_SLICE_512 and \
            KV_SEQLEN_SLICE_512 * embedQKSplit <= BLOCK_LIMIT_NO_PPONG * 2 and \
            KV_SEQLEN_SLICE_512 * embedVOSplit <= BLOCK_LIMIT_NO_PPONG * 2 and \
            (block_size == KV_SEQLEN_SLICE_256 // 4 or block_size == KV_SEQLEN_SLICE_256 // 2) and \
                self.head_num_move < 4:
            block_size_calc = KV_SEQLEN_SLICE_512
        return block_size_calc

    def calc_data(self, num_tokens, num_heads, kv_heads, head_size_qk, head_size_vo, block_size, num_blocks, k_seqlen,
                  dtype, mask_dim=0, mask_data_type=torch.bfloat16,
                  dynamic_batch=False, dynamic_seqlen=None, is_int8_flag=False, has_bias=False,
                  compressHead=False, is_kv_combined=True, is_nz_in=False, is_quant_flag=False):
        self.num_heads = num_heads
        self.kv_heads = kv_heads
        self.num_tokens = num_tokens
        self.compressHead = compressHead
        self.head_size_qk = head_size_qk
        self.head_size_vo = head_size_vo
        self.is_quant_flag = is_quant_flag

        q_min_range = -1.0
        q_max_range = 1.0
        kv_range = 1.0
        kv_type = dtype
        key_cache_rope = None
        query_rope = None
        if self.is_quant_flag:
            q_min_range = -5
            q_max_range = 5
            kv_min_range = -5
            kv_max_range = 5
            dtype = torch.int8
            kv_type = torch.int8
            query = torch.from_numpy(np.random.uniform(q_min_range, q_max_range, size=(
                num_tokens, num_heads, head_size_vo))).to(dtype)
            query_rope = torch.from_numpy(
                np.random.uniform(-kv_range, kv_range, size=(num_tokens, num_heads, 64))).to(mask_data_type)
        else:
            query = torch.from_numpy(np.random.uniform(q_min_range, q_max_range, size=(
                num_tokens, num_heads, head_size_qk))).to(dtype)
        if is_int8_flag:
            kv_range = 4.0
            kv_type = torch.int8

        if not compressHead:
            if self.is_quant_flag:
                key_cache = torch.from_numpy(np.random.uniform(kv_min_range, kv_max_range, size=(
                    num_blocks, block_size, kv_heads, head_size_vo))).to(kv_type)
                key_cache_rope = torch.from_numpy(np.random.uniform(-kv_range, kv_range, size=(
                    num_blocks, block_size, kv_heads, 64))).to(torch.float16)
            else:
                key_cache = torch.from_numpy(np.random.uniform(-kv_range, kv_range, size=(
                    num_blocks, block_size, kv_heads, head_size_qk))).to(kv_type)
            if not is_kv_combined:
                value_cache = torch.from_numpy(np.random.uniform(-kv_range, kv_range, size=(
                    num_blocks, block_size, kv_heads, head_size_vo))).to(kv_type)
            else:
                value_cache = key_cache[:, :, :, :head_size_vo]
        else:
            key_cache = torch.from_numpy(np.random.uniform(-kv_range, kv_range, size=(
                num_blocks * kv_heads, block_size, 1, head_size_qk))).to(kv_type)
            if not is_kv_combined:
                value_cache = torch.from_numpy(np.random.uniform(-kv_range, kv_range, size=(
                    num_blocks * kv_heads, block_size, 1, head_size_vo))).to(kv_type)
            else:
                value_cache = key_cache[:, :, :, :head_size_vo]
        self.data_type = dtype

        if dynamic_batch:
            context_lens = dynamic_seqlen
        else:
            context_lens = [k_seqlen] * num_tokens
        max_context_len = max(context_lens)
        self.max_context_len = max_context_len
        batch = len(context_lens)

        # alibi mask
        if mask_dim == 4:
            mask = np.zeros(
                (batch, num_heads, 1, self.max_context_len), dtype=np.float32)
            alibi_slopes = self.get_alibi_slopes(num_heads)
            for i, context_len in enumerate(context_lens):
                if context_len == 0:
                    continue
                position_ids = np.arange(context_len).astype(np.int32)
                alibi_bias = (position_ids - context_len +
                              1).astype(np.float32)
                alibi_bias = alibi_slopes.reshape(-1,
                                                  1, 1) * alibi_bias.reshape(1, 1, -1)
                mask[i, :, :, :context_len] = alibi_bias
            mask = torch.from_numpy(mask).to(mask_data_type)
        # normal mask
        elif mask_dim == 3:
            mask = np.zeros((batch, 1, max_context_len), dtype=np.float16)
            for i in range(batch):
                mask[i, :, :i] = -10000
            mask = torch.from_numpy(mask).to(mask_data_type)
        else:  # no mask
            mask = None

        if compressHead:
            context_lens = [val for val in context_lens for _ in range(kv_heads)]
        batch = len(context_lens)
        max_num_blocks_per_seq = (
            max_context_len + block_size - 1) // block_size
        block_tables = []   # （num_tokens, max_num_blocks_per_seq）
        for i in range(batch):
            block_table = [i * max_num_blocks_per_seq + _ for _ in range(max_num_blocks_per_seq)]
            block_tables.append(block_table)

        self.is_int8_flag = is_int8_flag
        if is_int8_flag:
            de_scale1_fp32 = np.random.randint(-1, 2,
                                               size=(kv_heads * head_size)).astype(np.float32)
            de_scale1_int64 = self.process_deq_scale(de_scale1_fp32)

            de_scale2_fp32 = np.random.randint(-1, 2,
                                               size=(kv_heads * head_size)).astype(np.float32)
            de_scale2_int64 = self.process_deq_scale(de_scale2_fp32)

            offset1 = np.random.randint(-20, 20,
                                        size=(kv_heads * head_size)).astype(np.int32)

            offset2 = np.random.randint(-20, 20,
                                        size=(kv_heads * head_size)).astype(np.int32)

            self.de_scale1_int64 = torch.tensor(
                list(de_scale1_int64), dtype=torch.int64)
            self.de_scale2_int64 = torch.tensor(
                list(de_scale2_int64), dtype=torch.int64)
            self.de_scale1_fp32 = torch.from_numpy(de_scale1_fp32)
            self.de_scale2_fp32 = torch.from_numpy(de_scale2_fp32)
            self.offset1 = torch.from_numpy(offset1)
            self.offset2 = torch.from_numpy(offset2)
            self.has_bias = has_bias

        self.de_scale1_fp32 = torch.from_numpy(np.random.uniform(-5 / 127, 5 / 127, size=(num_heads)).astype(np.float32)).to(torch.float32)
        self.de_scale2_fp32 = torch.from_numpy(np.random.uniform(-5 / 127, 5 / 127, size=(num_heads)).astype(np.float32)).to(torch.float32)
        if self.is_quant_flag:

            self.scale = torch.from_numpy(np.random.uniform(
                0, 127, size=(num_heads)).astype(np.float32)).to(torch.float32)
            self.kvsplit = 1
            self.kv_split_per_core = max_context_len
            self.head_num_move = 1
            self.former_head = self.num_heads
            self.embedQKSplit, self.embedVOSplit = self.CalcuHeadSplitNd(
                head_size_qk, head_size_vo)
            self.block_size_calc = self.get_blockszie_calc(
                max_context_len, block_size, head_size_qk, head_size_vo)
            self.block_size = block_size

        shape_out = (num_tokens, num_heads, head_size_vo)
        ref_output = torch.zeros(shape_out, dtype=mask_data_type)
        true_out = torch.zeros(shape_out, dtype=torch.float32)
        sim = torch.zeros((num_tokens, num_heads * k_seqlen),
                          dtype=torch.float32)
        self.ref_single_query_cached_kv_attention(
            sim,
            ref_output,
            true_out,
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            mask,
            mask_dim,
            mask_data_type,
            query_rope,
            key_cache_rope
        )
        if self.is_quant_flag:
            self.q_split1, self.q_split2 = query, query_rope
            key_cache_split1 = key_cache.reshape(num_blocks, block_size, -1)
            key_cache_split2 = key_cache_rope.reshape(
                num_blocks, block_size, -1)
            key_cache_split1_nz = self.convert_nd_to_nz(key_cache_split1)
            key_cache_split2_nz = self.convert_nd_to_nz(key_cache_split2)
            self.key_cache_split1 = key_cache_split1_nz.reshape(
                num_blocks, -1, block_size, 32).to(torch.int8)
            self.key_cache_split2 = key_cache_split2_nz.reshape(
                num_blocks, -1, block_size, 16).to(mask_data_type)
        elif not is_nz_in:
            self.q_split1, self.q_split2 = torch.split(query, [512, 64], dim=2)
            self.key_cache_split1, self.key_cache_split2 = torch.split(
                key_cache.to(mask_data_type), [512, 64], dim=3)
        else:
            self.q_split1, self.q_split2 = torch.split(query, [512, 64], dim=2)
            key_cache_split1, key_cache_split2 = torch.split(
                key_cache, [512, 64], dim=3)
            key_cache_split1 = key_cache_split1.reshape(
                num_blocks, block_size, -1)
            key_cache_split2 = key_cache_split2.reshape(
                num_blocks, block_size, -1)
            key_cache_split1_nz = self.convert_nd_to_nz(key_cache_split1)
            key_cache_split2_nz = self.convert_nd_to_nz(key_cache_split2)
            self.key_cache_split1 = key_cache_split1_nz.to(
                mask_data_type).reshape(num_blocks, -1, block_size, 16)
            self.key_cache_split2 = key_cache_split2_nz.to(
                mask_data_type).reshape(num_blocks, -1, block_size, 16)
        self.block_tables = np.array(block_tables).astype(np.int32)
        self.contex_lens = np.array(context_lens).astype(np.int32)
        self.alib_mask = mask
        self.golden_out = ref_output
        self.true_out = true_out

    def golden_calc(self, in_tensors):
        golden_out = torch.tensor(self.golden_out)
        return [golden_out]

    def golden_compare(self, out_tensors, golden_tensors):
        result_old = self.compare_output_data(
            out_tensors.npu(), golden_tensors.npu(), [0.001, 0.001, 0.005, 0.005])
        return result_old

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_mla_split_quant_nz(self):
        num_tokens = 32
        num_heads = 32
        kv_heads = 1
        block_size = 128
        head_size_qk = 576
        head_size_vo = 512
        num_blocks = 64
        k_seqlen = 256
        tor = 1.0 / (head_size_qk ** 0.5)
        mask_dim = 0
        dtype = torch.float16
        is_kv_combined = True
        is_nz_in = True
        is_quant_flag = True
        self.calc_data(num_tokens, num_heads, kv_heads, head_size_qk, head_size_vo, block_size, num_blocks, k_seqlen, dtype, mask_dim, dtype,
                       is_kv_combined=is_kv_combined, is_nz_in=is_nz_in, is_quant_flag=is_quant_flag)
        block_table = torch.tensor(self.block_tables).int().npu()
        contex_lens = torch.tensor(self.contex_lens).int().cpu()
        ctkv = torch_npu.npu_format_cast(
            self.key_cache_split1.contiguous().npu(), 29)
        krope = torch_npu.npu_format_cast(
            self.key_cache_split2.contiguous().npu(), 29)
        qk_descale = self.de_scale1_fp32.npu()
        pv_descale = self.de_scale2_fp32.npu()
        attenOut = torch_npu.atb.npu_multi_head_latent_attention(self.q_split1.npu(), self.q_split2.npu(),
                                                                 ctkv, krope, block_table, contex_lens, num_heads, tor, kv_heads, qk_descale=qk_descale,
                                                                 pv_descale=pv_descale,
                                                                 cache_mode='int8_nzcache')
        self.assertRtolEqual(attenOut, self.golden_out)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_mla_split_quant_nz_head128(self):
        num_tokens = 32
        num_heads = 128
        kv_heads = 1
        block_size = 128
        head_size_qk = 576
        head_size_vo = 512
        num_blocks = 64
        k_seqlen = 256
        tor = 1.0 / (head_size_qk ** 0.5)
        mask_dim = 0
        dtype = torch.float16
        is_kv_combined = True
        is_nz_in = True
        is_quant_flag = True
        self.calc_data(num_tokens, num_heads, kv_heads, head_size_qk, head_size_vo, block_size, num_blocks, k_seqlen, dtype, mask_dim, dtype,
                       is_kv_combined=is_kv_combined, is_nz_in=is_nz_in, is_quant_flag=is_quant_flag)

        block_table = torch.tensor(self.block_tables).int().npu()
        contex_lens = torch.tensor(self.contex_lens).int().cpu()
        ctkv = torch_npu.npu_format_cast(
            self.key_cache_split1.contiguous().npu(), 29)
        krope = torch_npu.npu_format_cast(
            self.key_cache_split2.contiguous().npu(), 29)
        qk_descale = self.de_scale1_fp32.npu()
        pv_descale = self.de_scale2_fp32.npu()
        attenOut = torch_npu.atb.npu_multi_head_latent_attention(self.q_split1.npu(), self.q_split2.npu(),
                                                                 ctkv, krope, block_table, contex_lens, num_heads, tor, kv_heads, qk_descale=qk_descale,
                                                                 pv_descale=pv_descale,
                                                                 cache_mode='int8_nzcache')
        self.assertRtolEqual(attenOut, self.golden_out)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_mla_split_quant_nz_bf16(self):
        num_tokens = 32
        num_heads = 32
        kv_heads = 1
        block_size = 128
        head_size_qk = 576
        head_size_vo = 512
        num_blocks = 64
        k_seqlen = 256
        tor = 1.0 / (head_size_qk ** 0.5)
        mask_dim = 0
        dtype = torch.bfloat16
        is_kv_combined = True
        is_nz_in = True
        is_quant_flag = True
        self.calc_data(num_tokens, num_heads, kv_heads, head_size_qk, head_size_vo, block_size, num_blocks, k_seqlen, dtype, mask_dim, dtype,
                       is_kv_combined=is_kv_combined, is_nz_in=is_nz_in, is_quant_flag=is_quant_flag)
        block_table = torch.tensor(self.block_tables).int().npu()
        contex_lens = torch.tensor(self.contex_lens).int().cpu()
        ctkv = torch_npu.npu_format_cast(
            self.key_cache_split1.contiguous().npu(), 29)
        krope = torch_npu.npu_format_cast(
            self.key_cache_split2.contiguous().npu(), 29)
        qk_descale = self.de_scale1_fp32.npu()
        pv_descale = self.de_scale2_fp32.npu()
        attenOut = torch_npu.atb.npu_multi_head_latent_attention(self.q_split1.npu(), self.q_split2.npu(),
                                                                 ctkv, krope, block_table, contex_lens, num_heads, tor, kv_heads, qk_descale=qk_descale,
                                                                 pv_descale=pv_descale,
                                                                 cache_mode='int8_nzcache')
        self.assertRtolEqual(attenOut, self.golden_out)


if __name__ == "__main__":
    run_tests()

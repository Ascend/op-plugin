from enum import Enum
import math
import random

import numpy as np
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class ScaleType(Enum):
    SCALE_TOR = 0
    SCALE_LOGN = 1
    SCALE_LOGN_FP32 = 2

np.random.seed(123)
MASK_TYPE_NO_MASK = 0
MASK_TYPE_NO_HEAD = 1
MASK_TYPE_NO_BATCH = 2
MASK_TYPE_ALIBI_WITH_BATCH = 3
MASK_TYPE_ALIBI_NO_BATCH = 4
MASK_TYPE_NO_HEAD_DECODER = 5
MASK_TYPE_SWA = 6
MASK_TYPE_SWA_DECODER = 7
MASK_TYPE_ALIBI_WITH_PREFIX_BATCH = 8
MASK_TYPE_NO_BATCH_WITH_PREFIX = 9
MASK_TYPE_ALIBI_NO_BATCH_WITH_PREFIX = 10
CAL_TYPE_PREFIX_ENCODER = 4
MASK_TYPE_ALIBI_COMPRESS = 4
MASK_TYPE_ALIBI_COMPRESS_SQRT = 5
UNPAD_FLASH_ATTENTION_ENCODER_PREFIX_CACHE_ND = 2012


class TestFlashAttentionPrefixEncoder(TestCase):
    def close_pack(self, in_data, seq_len):
        kv = in_data.numpy()
        dim1len = np.size(kv, -2)
        if max(seq_len) > dim1len:
            return None
        kv = kv.reshape(np.prod(kv.shape[0:-1]), kv.shape[-1])
        c_offset = 0
        s_offset = 0
        for i, _ in enumerate(seq_len):
            kv[c_offset:c_offset + seq_len[i]
               ][:] = kv[s_offset:s_offset + seq_len[i]][:]
            c_offset += seq_len[i]
            s_offset += dim1len
        return torch.from_numpy(kv[0:sum(seq_len)][:])

    # pylint:disable = huawei-too-many-arguments
    def set_data_params(self, dynamic_batch=False, batch_state=None, window_size=0, cache_type=0,
                        is_mask=True, is_decoder=False, is_alibi=False, alibi_dim=4,
                        batch=1, kv_head=1, heads=1, embeddim=128, embeddimv=0, max_seq=2048,
                        kv_seqLen=None, is_clamp=0, clamp_min=0,
                        clamp_max=0, data_type=torch.float16, op_type=0, mask_type=0,
                        no_cache=False, long_seq=False, is_triu_mask=False, is_multi_layer=False,
                        is_sqrt=False, left_align=False, scaleType=ScaleType.SCALE_TOR.value, fav3=False,
                        tor=1, bnsd=False, is_compress=False, q_seqlens=None, num_blocks=None,
                        block_size=None):
        if kv_seqLen is None:
            kv_seqLen = [] 
        self.dynamic_batch = dynamic_batch
        self.batch_state = batch_state
        self.is_mask = is_mask
        self.is_decoder = is_decoder
        self.is_alibi = is_alibi
        self.alibi_dim = alibi_dim
        self.batch = batch
        self.kv_head = kv_head
        self.heads = heads
        self.embeddim = embeddim
        self.embeddimv = embeddimv
        self.max_seq = max_seq
        self.kv_seqLen = kv_seqLen
        self.dynamic_batch = dynamic_batch
        self.is_clamp = is_clamp
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.data_type = data_type
        self.no_cache = no_cache
        self.long_seq = long_seq
        self.mask_type = mask_type
        self.is_triu_mask = is_triu_mask
        self.is_multi_layer = is_multi_layer
        self.is_sqrt = is_sqrt
        self.left_align = left_align
        self.fav3 = fav3
        self.scaleType = scaleType
        self.tor = tor
        self.is_int8_flag = False
        self.online = False
        self.bnsd = bnsd
        self.window_size = window_size
        self.is_compress = is_compress
        self.cache_type = cache_type
        self.q_seqlens = q_seqlens if q_seqlens is not None else kv_seqLen

        if self.embeddimv == 0:
            self.embeddimv = self.embeddim
        if is_decoder:
            self.q_seqlen, self.q_ntokens = self.gen_seq_len(batch, [
                                                             1] * batch)
        else:
            self.q_seqlen, self.q_ntokens = self.gen_seq_len(
                batch, self.q_seqlens)
        self.kv_seqlen, self.kv_ntokens = self.gen_seq_len(batch, kv_seqLen)
        if is_multi_layer:
            self.layer_id = torch.from_numpy(
                np.array([1], dtype=np.int32)).to(torch.int32)
        else:
            self.layer_id = torch.from_numpy(
                np.array([0], dtype=np.int32)).to(torch.int32)
        self.q_max_seq = np.max(self.q_seqlen)
        self.kv_max_seq = np.max(self.kv_seqlen)
        q = torch.from_numpy(
            np.random.uniform(-1.0, 1.0, size=(self.q_ntokens, heads * self.embeddim)))
        self.q = q.to(data_type)
        if num_blocks is None:
            self.k = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(
                self.layer_id[0] + 1, batch, self.max_seq, kv_head * self.embeddim))).to(data_type)
            self.v = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(
                self.layer_id[0] + 1, batch, self.max_seq, kv_head * self.embeddimv))).to(data_type)
        else:
            self.k_cache = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(num_blocks, block_size, kv_head, embeddim))).to(data_type)
            self.v_cache = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(num_blocks, block_size, kv_head, embeddim))).to(data_type)
            batch = len(kv_seqLen)
            max_context_len = max(kv_seqLen)
            max_num_blocks_per_seq = (
                max_context_len + block_size - 1) // block_size
            block_tables = []
            offset = 0
            for i in range(batch):
                num_blocks_cur_seq = (kv_seqLen[i] + block_size - 1) // block_size
                block_table = [
                    random.randint(0, num_blocks - 1)
                    if j < num_blocks_cur_seq
                    else 0
                    for j in range(max_num_blocks_per_seq)
                ]
                offset += num_blocks_cur_seq
                block_tables.append(block_table)
            self.block_tables = torch.from_numpy(np.array(block_tables)).to(torch.int32)
            self.block_tables = torch.from_numpy(
                np.array(block_tables)).to(torch.int32)
            self.k = torch.stack([self.k_cache[self.block_tables[torch.tensor(i, dtype=torch.long)].to(
                torch.long)].reshape(-1, kv_head * self.embeddim)[:max_context_len, :] for i in range(batch)])
            self.v = torch.stack([self.v_cache[self.block_tables[torch.tensor(i, dtype=torch.long)].to(
                torch.long)].reshape(-1, kv_head * self.embeddim)[:max_context_len, :] for i in range(batch)])
            self.k = self.k.reshape(1, batch, max_context_len, kv_head * self.embeddim)
            self.v = self.v.reshape(1, batch, max_context_len, kv_head * self.embeddim)
        if self.fav3:
            self.is_int8_flag = True
            self.q_scale, self.q_offset, self.q_int8 = self.quant_per_head(
                self.q, heads, embeddim, (self.q_ntokens, heads * self.embeddim))
            self.k_scale, self.k_offset, self.k_int8 = self.quant_per_head(
                self.k, kv_head, embeddim, (self.layer_id[0] + 1, batch, self.max_seq, kv_head * self.embeddim))
            self.v_scale, self.v_offset, self.v_int8 = self.quant_per_head(
                self.v, kv_head, embeddim, (self.layer_id[0] + 1, batch, self.max_seq, kv_head * self.embeddim))
            self.k_scale = (self.k_scale.view(kv_head, 1) * torch.ones([kv_head, heads // kv_head])).view(-1)
            self.k_offset = (self.k_offset.view(kv_head, 1) * torch.ones([kv_head, heads // kv_head])).view(-1)
            self.v_scale = (self.v_scale.view(kv_head, 1) * torch.ones([kv_head, heads // kv_head])).view(-1)
            self.v_offset = (self.v_offset.view(kv_head, 1) * torch.ones([kv_head, heads // kv_head])).view(-1)
            self.offline_scale = torch.from_numpy(np.random.uniform(1 / 127, 3 / 127, size=(heads))).to(torch.float32)

            self.q_int8 = torch.from_numpy(
                np.random.uniform(-5.0, 5.0, size=(self.q_ntokens, heads * self.embeddim))).to(torch.int8)
            self.k_int8 = torch.from_numpy(np.random.uniform(-5.0, 5.0, size=(
                self.layer_id[0] + 1, batch, self.max_seq, kv_head * self.embeddim))).to(torch.int8)
            self.v_int8 = torch.from_numpy(np.random.uniform(-5.0, 5.0, size=(
                self.layer_id[0] + 1, batch, self.max_seq, kv_head * self.embeddimv))).to(torch.int8)

        self.gen_mask(batch, heads, data_type, mask_type,
                      window_size, is_compress, cache_type)

    def quant_per_head(self, data, heads, embeddim, shape):
        temp = data.view(-1, heads, self.embeddim)
        scale = torch.stack([self.fav3_quant(
            temp[:, i, :], data_min=-1, data_max=1, symmetric=True)[0] for i in range(heads)])
        offset = torch.stack([self.fav3_quant(
            temp[:, i, :], data_min=-1, data_max=1, symmetric=True)[1] for i in range(heads)])
        int8_data = torch.zeros_like(temp)
        for i in range(heads):
            int8_data[:, i, :] = (
                (temp[:, i, :] / scale[i]).round_() + offset[i])
        int8_data = int8_data.view(shape).to(torch.int8)
        return scale, offset, int8_data

    def fav3_quant(self, data, data_min=0, data_max=0, symmetric=False, bit=8):
        n = 2 ** (bit - 1)
        if symmetric:
            quant_min, quant_max = -(n - 1), (n - 1)
        else:
            quant_min, quant_max = -n, (n - 1)
        span = quant_max - quant_min
        if data_min == data_max:
            data_max = data.max().item()
            data_min = data.min().item()
        if symmetric:
            scale = max(data_max, -data_min) / (float(span) / 2)
            offset = 0
        else:
            scale = (data_max - data_min) / float(span)
            offset = (data_min * quant_min + data_max *
                      quant_max) / (data_min - data_max)
        return torch.tensor(float(scale), dtype=torch.float), torch.tensor(int(offset), dtype=torch.float)

    def get_alibi_slopes(self, n_heads):
        n = 2 ** math.floor(math.log2(n_heads))
        m0 = 2.0 ** (-8.0 / n)
        slopes = torch.pow(m0, torch.arange(1, n + 1))
        if n < n_heads:
            m1 = 2.0 ** (-4.0 / n)
            mm = torch.pow(m1, torch.arange(1, 1 + 2 * (n_heads - n), 2))
            slopes = torch.cat([slopes, mm])
        return slopes

    def get_alibi_bias(self, n_heads, max_seqlen):
        if not self.left_align:
            self.bias = torch.arange(max_seqlen)
            self.bias = self.bias[None, :] - self.bias[:, None]
            if (self.is_sqrt):
                self.bias = torch.sqrt(
                    torch.abs(self.bias)) * torch.sign(self.bias)
            bias = torch.empty(
                n_heads,
                max_seqlen,
                max_seqlen
            )[:, :max_seqlen, :max_seqlen].copy_(self.bias)
            self.alibi_slopes = self.get_alibi_slopes(n_heads)
        else:
            self.bias = torch.arange(max_seqlen, dtype=torch.float32).unsqueeze(
                0).unsqueeze(0).expand(n_heads, max_seqlen, -1)
            self.alibi_slopes = torch.Tensor(self.get_interleave(n_heads))
            bias = self.bias
        bias = bias * self.alibi_slopes[:, None, None]
        return bias

    def get_interleave(self, n, alibi_bias_max=8.0):
        def get_interleave_power_of_2(n, alibi_bias_max):
            if n == 0:
                return 0
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]
        if math.log2(n).is_integer():
            return get_interleave_power_of_2(n, alibi_bias_max)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return get_interleave_power_of_2(closest_power_of_2, alibi_bias_max) + \
                self.get_interleave(
                    2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

    def gen_swa_cmp(self, max_seq, window_size):
        swa_mask = np.ones(shape=(1, 512, 512)) * self.pre_mask_coff
        pp_n = 128 if self.embeddim <= 128 else 64
        if window_size <= pp_n * 3:
            true_size = window_size
        else:
            if window_size % pp_n == 0:
                true_size = pp_n * 3
            else:
                true_size = pp_n * 2 + window_size % pp_n
        triu_mask = np.triu(swa_mask, 1)
        tril_mask = np.tril(swa_mask, -true_size)
        swa_mask = triu_mask + tril_mask
        swa_mask = torch.from_numpy(swa_mask).to(torch.float32)
        return swa_mask

    def gen_swa_mask(self, max_seq, window_size, pre_mask_coff, cache_type=0):
        swa_mask = np.ones(shape=self.mask_info[0]) * pre_mask_coff
        if window_size < max_seq and self.is_decoder:
            if cache_type == 1:
                for idx, _ in enumerate(self.kv_seqLen):
                    swa_mask[idx, :, :window_size] = 0
            else:
                for idx, _ in enumerate(self.kv_seqLen):
                    swa_mask[idx, :, kvseqlen - window_size: kvseqlen] = 0
        elif window_size < max_seq or self.is_compress:
            triu_mask = np.triu(swa_mask, 1)
            tril_mask = np.tril(swa_mask, -window_size)
            swa_mask = triu_mask + tril_mask
        else:
            swa_mask = np.triu(swa_mask, 1)
        return swa_mask

    # pylint:disable = huawei-too-many-arguments
    def gen_mask(self, batch, heads, data_type, mask_type, window_size, is_compress, cache_type=0):
        q_max_seq = self.max_seq
        kv_max_seq = self.max_seq
        mask_type_dict = {
            # 四维的alibi mask
            MASK_TYPE_ALIBI_WITH_BATCH: ((batch, heads, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :, :q_s, :kv_s]))),
            MASK_TYPE_ALIBI_WITH_PREFIX_BATCH: ((batch, heads, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :, kv_s - q_s:kv_s, :kv_s]))),
            # 三维的alibi mask
            MASK_TYPE_ALIBI_NO_BATCH: ((heads, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:, :q_s, :kv_s]))),
            MASK_TYPE_ALIBI_NO_BATCH_WITH_PREFIX: ((heads, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:, kv_s - q_s:kv_s, :kv_s]))),
            MASK_TYPE_NO_HEAD: ((batch, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :q_s, :kv_s]))),
            MASK_TYPE_NO_HEAD_DECODER: ((batch, 1, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :q_s, :kv_s]))),
            MASK_TYPE_NO_BATCH: ((1, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:, :q_s, :kv_s]))),
            MASK_TYPE_NO_BATCH_WITH_PREFIX: ((1, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:, kv_s - q_s:kv_s, :kv_s]))),
            MASK_TYPE_SWA: ((1, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:, :q_s, :kv_s]))),
            MASK_TYPE_SWA_DECODER: ((batch, 1, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :q_s, :kv_s]))),
            # 不加mask
            MASK_TYPE_NO_MASK: ((1, q_max_seq, kv_max_seq),
                                (lambda mask, idx, q_s, kv_s: 0))
        }
        if data_type == torch.float16:
            post_mask_coff = 1
            pre_mask_coff = -10000.0
        elif data_type == torch.bfloat16 and self.is_alibi:
            post_mask_coff = 1
            pre_mask_coff = -float("inf")
        elif data_type == torch.float32 and self.is_alibi:
            post_mask_coff = 1
            pre_mask_coff = 1
        else:
            post_mask_coff = -3e38
            pre_mask_coff = 1
        if data_type == torch.float16:
            if self.window_size > 0:
                select_zero = False
            elif self.is_alibi or self.long_seq:
                select_zero = False
            else:
                select_zero = True
        elif data_type == torch.bfloat16:
            if self.window_size > 0:
                select_zero = False
            elif self.is_alibi:
                select_zero = False
            elif self.dynamic_batch or self.is_decoder:
                select_zero = True
            else:
                select_zero = False
        else:
            if self.is_alibi or self.is_decoder:
                select_zero = True
            else:
                select_zero = False
        if self.is_triu_mask:
            select_zero = False
        self.mask_info = mask_type_dict.get(mask_type, ((1, self.max_seq, self.max_seq), lambda mask, idx, q_s, kv_s: 0))
        mask = np.ones(shape=self.mask_info[0]) * pre_mask_coff
        mask = np.triu(mask, 1)
        zero_indice = random.choices(range(self.max_seq), k=300)
        if self.window_size > 0:
            mask = self.gen_swa_mask(
                self.max_seq, window_size, pre_mask_coff, cache_type)
        if self.is_alibi:
            self.alibi_bias = self.get_alibi_bias(heads, self.max_seq)
            mask += self.alibi_bias.numpy()
        if select_zero:
            mask.flat[zero_indice] = 0
        self.mask = torch.from_numpy(mask).to(torch.float32)
        self.post_mask_coff = post_mask_coff
        self.pre_mask_coff = pre_mask_coff

    def quantize_tensor_symmetric(self, x, prev_max_abs_vals=None, num_bits=8):
        if x.dtype != torch.float:
            x = x.to(torch.float)
        quant_min = -2 ** (num_bits - 1)
        quant_max = 2 ** (num_bits - 1) - 1

        current_max_abs_vals = x.abs().max(dim=1).values
        if prev_max_abs_vals is not None:
            max_abs_vals = torch.max(prev_max_abs_vals, current_max_abs_vals)
        else:
            max_abs_vals = current_max_abs_vals
        scales = max_abs_vals / (quant_max)
        x_q = torch.clamp(torch.round(
            x / scales.unsqueeze(1)), quant_min, quant_max)
        x_q = torch.round(x_q)
        x_q = x_q.to(torch.int8)
        return x_q, scales, max_abs_vals

    def dequantize_tensor(self, x_q, scales, value):
        x_deq = x_q.to(torch.float32)
        scales = scales.unsqueeze(1)
        x_deq = x_deq * value
        x_deq = x_deq * scales
        return x_deq

    # pylint:disable = huawei-too-many-arguments
    def online_softmax(self, s_qk, q_s, v_slice, heads, kv_head, embed, online, dtype):
        ans = None
        group_num = heads // kv_head
        for head_idx in range(heads):
            s_head_idx = s_qk[head_idx]
            O = torch.zeros((q_s, embed)).to(dtype)
            Br = q_s
            Bc = 128
            self.row_block_size = Br
            self.col_block_size = Bc
            d = embed
            V_mat = v_slice[head_idx // group_num]
            Tr = q_s // Br
            Tc = q_s // Bc

            d = embed
            Tr = q_s // Br
            Tc = q_s // Bc

            start_row_idx = 0
            start_col_idx = 0

            for i in range(Tr):
                Oi = torch.zeros((Br, d)).to(dtype)
                li = torch.zeros((Br, 1)).to(dtype)
                mi = torch.full((Br, 1), -torch.inf).to(dtype)
                pp_max_num = None
                for j in range(Tc):
                    Sij = s_head_idx[i * Br: (i + 1) * Br, start_col_idx +
                                     j * Bc: start_col_idx + (j + 1) * Bc].to(dtype)
                    Vj = V_mat[start_col_idx + j *
                               Bc: start_col_idx + (j + 1) * Bc, :]
                    mi_new = torch.max(
                        torch.column_stack([mi, torch.max(Sij, dim=1).values[:, None]]), dim=1
                    ).values[:, None].to(dtype)
                    Pij_hat = torch.exp((Sij - mi_new).to(torch.float32))
                    Pij_hat = Pij_hat.to(dtype)
                    li = torch.exp((mi - mi_new).to(torch.float32)).to(dtype) * \
                        li + torch.sum(Pij_hat, dim=1)[:, None]
                    if self.is_int8_flag:
                        if online:
                            x_q, scales, pp_max_num = self.quantize_tensor_symmetric(
                                Pij_hat, pp_max_num)
                            if pp_max_num is None:
                                pp_max_num = pp_max_num
                            pv = x_q.to(torch.int32) @ Vj.to(torch.int32)
                            Oi = Oi * torch.exp((mi - mi_new).to(torch.float32)).to(
                                dtype) + self.dequantize_tensor(pv, scales, self.v_scale[head_idx]).to(dtype)
                        else:
                            x_q = Pij_hat / self.offline_scale[head_idx]
                            x_q = torch.round(x_q.to(torch.float32))
                            pv = x_q.to(torch.int32) @ Vj.to(torch.int32)
                            pv = pv.to(torch.float32)
                            value = self.v_scale[head_idx] * \
                                self.offline_scale[head_idx]
                            Oi = Oi * \
                                torch.exp((mi - mi_new).to(torch.float32)
                                          ).to(dtype) + (pv * value).to(dtype)
                    else:
                        Oi = Oi * \
                            torch.exp((mi - mi_new).to(torch.float32)
                                      ).to(dtype) + Pij_hat @ Vj.to(dtype)
                    mi = mi_new

                if (q_s % Bc != 0):
                    Bc = q_s % Bc
                    start_row_idx = (
                        q_s // self.row_block_size) * self.row_block_size
                    start_col_idx = (
                        q_s // self.col_block_size) * self.col_block_size

                    Sij = s_head_idx[i * Br: (i + 1) * Br,
                                     start_col_idx: start_col_idx + Bc].to(dtype)
                    Vj = V_mat[start_col_idx: start_col_idx + Bc, :]
                    mi_new = torch.max(
                        torch.column_stack([mi, torch.max(Sij, dim=1).values[:, None]]), dim=1
                    ).values[:, None].to(dtype)
                    Pij_hat = torch.exp((Sij - mi_new).to(torch.float32))
                    Pij_hat = Pij_hat.to(dtype)
                    li = torch.exp((mi - mi_new).to(torch.float32)).to(dtype) * \
                        li + torch.sum(Pij_hat, dim=1)[:, None]
                    if self.is_int8_flag:
                        if online:
                            x_q, scales, pp_max_num = self.quantize_tensor_symmetric(
                                Pij_hat, pp_max_num)
                            if pp_max_num is None:
                                pp_max_num = pp_max_num
                            pv = x_q.to(torch.int32) @ Vj.to(torch.int32)
                            Oi = Oi * torch.exp((mi - mi_new).to(torch.float32)).to(
                                dtype) + self.dequantize_tensor(pv, scales, self.v_scale[head_idx]).to(dtype)
                        else:
                            x_q = Pij_hat / self.offline_scale[head_idx]
                            x_q = torch.round(x_q.to(torch.float32))
                            pv = x_q.to(torch.int32) @ Vj.to(torch.int32)
                            pv = pv.to(torch.float32)
                            value = self.v_scale[head_idx] * \
                                self.offline_scale[head_idx]
                            Oi = Oi * \
                                torch.exp((mi - mi_new).to(torch.float32)
                                          ).to(dtype) + (pv * value).to(dtype)
                    else:
                        Oi = Oi * \
                            torch.exp((mi - mi_new).to(torch.float32)
                                      ).to(dtype) + Pij_hat @ Vj.to(dtype)
                Oi = Oi / li
                O[i * Br: (i + 1) * Br, :] = Oi
            if ans is None:
                ans = O
            else:
                ans = torch.cat((ans, O), 1)
        return ans

    def gen_out_tensor(self, online=False):
        q_offset = 0
        k_offset = 0
        v_offset = 0
        batch = self.batch
        dynamic_batch = self.dynamic_batch
        batch_state = self.batch_state
        heads = self.heads
        is_decoder = self.is_decoder
        embed = self.embeddim
        embedv = self.embeddimv
        max_seq = self.max_seq
        q_seqlen = self.q_seqlen
        kv_seqlen = self.kv_seqLen
        kv_head = self.kv_head
        mask = self.mask
        is_mask = self.is_mask
        q = self.q
        k = self.k
        v = self.v
        if self.fav3:
            q = self.q_int8
            k = self.k_int8
            v = self.v_int8
        q_ntokens = self.q_ntokens
        kv_ntokens = self.kv_ntokens
        layer_id = self.layer_id[0]
        s = None
        _p = None
        out = None
        ans_concat = None
        ans_concat_true = None
        out_true = None

        self.encoder_logN = torch.tensor(
            [2.0] * self.max_seq).to(torch.float32)
        self.encoder_logN.uniform_(1, 2)
        self.decoder_logN = torch.tensor([2.0] * batch).to(torch.float32)
        self.decoder_logN.uniform_(1, 2)
        for idx in range(batch):
            if dynamic_batch and batch_state[idx] == 0 and not is_decoder:
                continue
            if dynamic_batch and batch_state[idx] == 0:
                output = torch.zeros([heads, q_s, embedv])
                output = torch.permute(output, (1, 0, 2))
                if out is None:
                    out = output
                    if not self.fav3:
                        out_true = output
                else:
                    out = torch.cat((out, output), 0)
                    if not self.fav3:
                        out_true = torch.cat((out_true, output), 0)
                q_offset += q_s
                k_offset += max_seq
                v_offset += max_seq
                continue
            q_s = q_seqlen[idx]
            kv_s = kv_seqlen[idx]
            q_slice = q[q_offset:q_offset + q_s][:]
            q_slice = q_slice.view(q_s, heads, embed)
            q_slice = torch.permute(q_slice, (1, 0, 2))
            k_slice = k[layer_id][idx][:kv_s][:]
            k_slice = k_slice.view(kv_s, kv_head, embed)
            k_slice_t = torch.permute(k_slice, (1, 2, 0))
            v_slice = v[layer_id][idx][:kv_s][:]
            v_slice = v_slice.view(kv_s, kv_head, embedv)
            v_slice = torch.permute(v_slice, (1, 0, 2))

            if self.fav3:
                score = self.group_mm_torch(
                    heads, kv_head, q_slice, k_slice_t, torch.int32)
            else:
                score = self.group_mm_torch(heads, kv_head, q_slice, k_slice_t)
            if self.fav3:
                score = score.to(torch.float32)
                score = score * self.q_scale.view(heads, 1, 1)
                score = score.to(torch.float16)

            if s is None:
                s = score.view([-1, ])
            else:
                s = torch.cat((s, score.view([-1, ])), 0)

            if self.scaleType == ScaleType.SCALE_LOGN_FP32.value:
                if is_decoder:
                    score *= self.decoder_logN[idx]
                else:
                    score *= self.encoder_logN[None, :q_s, None]

            if self.fav3:
                score = score * torch.tensor(self.tor, dtype=torch.float16)
            else:
                score *= self.tor
            if self.is_clamp == 1:
                clamp_min_brc = np.ones((score.shape)) * self.clamp_min
                clamp_max_brc = np.ones((score.shape)) * self.clamp_max
                score = np.float16(np.maximum(score, clamp_min_brc))
                score = torch.from_numpy(np.float16(
                    np.minimum(score, clamp_max_brc)))
            if is_mask:
                mask = self.mask_info[1](self.mask, idx, q_s, kv_s)
                score = score + \
                    self.mask_info[1](self.mask, idx, q_s,
                                      kv_s) * self.post_mask_coff

            s_qk = score
            s_qk_true = score.to(torch.float32)
            score = score.numpy().astype(np.float32)
            if self.is_int8_flag:
                ans = self.online_softmax(
                    s_qk, q_s, v_slice, heads, kv_head, embed, online, torch.float16)
                if ans_concat is None:
                    ans_concat = ans
                else:
                    ans_concat = torch.cat((ans_concat, ans), 0)

                ans_true = self.online_softmax(
                    s_qk_true, q_s, v_slice, heads, kv_head, embed, online, torch.float32)
                if ans_concat_true is None:
                    ans_concat_true = ans_true
                else:
                    ans_concat_true = torch.cat((ans_concat_true, ans_true), 0)

            score_max = np.max(score, axis=-1)
            score = score - score_max.reshape((heads, q_s, 1))
            score_exp = np.exp(score)
            score_sum = np.sum(score_exp, axis=-1)

            if _p is None:
                _p = score_exp.astype(np.float32).reshape([-1, ])
            else:
                _p = np.concatenate(
                    (_p, score_exp.astype(np.float32).reshape([-1, ])), 0)
            if self.fav3:
                p = score_exp
                p = p * 127
                p = torch.from_numpy(p).to(torch.int8)
            else:
                p_true = (score_exp / score_sum.reshape((heads, q_s, 1)))
                p_true = torch.from_numpy(p_true)
                p = p_true.to(torch.bfloat16)
                o_true = self.group_mm_torch(heads, kv_head, p_true, v_slice)

            output = self.group_mm_torch(heads, kv_head, p, v_slice)
            if self.fav3:
                output = output.to(torch.float)
                v_scale = self.v_scale
                v_scale = v_scale.view(heads, 1, 1)
                output = output * v_scale
                output = output / 127
                output = output / score_sum.reshape((heads, q_s, 1))
            else:
                o_true = o_true.view(heads, q_s, embedv)
                o_true = torch.permute(o_true, (1, 0, 2)).contiguous()
            output = output.view(heads, q_s, embedv)
            output = torch.permute(output, (1, 0, 2)).contiguous()
            if out is None:
                out = output
                if not self.fav3:
                    out_true = o_true
            else:
                out = torch.cat((out, output), 0)
                if not self.fav3:
                    out_true = torch.cat((out_true, o_true), 0)

            q_offset += q_s
            k_offset += max_seq
            v_offset += max_seq

        if self.is_int8_flag:
            ans_concat = ans_concat.view(q_ntokens, heads * embedv)
            ans_concat_true = ans_concat_true.view(q_ntokens, heads * embedv)
            self.golden_out = ans_concat
            self.golden_out_true = ans_concat_true
        else:
            out = out.view(q_ntokens, heads * embedv)
            self.golden_out = out.to(self.data_type)
            out_true = out_true.view(q_ntokens, heads * embedv)
            self.golden_out_true = out_true.to(torch.float32)
        if self.long_seq:
            self.max_seq = 128
            self.gen_mask(self.batch, self.heads, self.data_type,
                          self.mask_type, 0, False, 0)

    def gen_out_tensor_bnsd(self):
        q_offset = 0
        k_offset = 0
        v_offset = 0
        batch = self.batch
        dynamic_batch = self.dynamic_batch
        batch_state = self.batch_state
        heads = self.heads
        is_decoder = self.is_decoder
        embed = self.embeddim
        embedv = self.embeddimv
        max_seq = self.max_seq
        q_seqlen = self.q_seqlen
        kv_seqlen = self.kv_seqLen
        kv_head = self.kv_head
        mask = self.mask
        is_mask = self.is_mask
        q = self.q
        k = self.k
        v = self.v
        q_ntokens = self.q_ntokens
        kv_ntokens = self.kv_ntokens
        layer_id = self.layer_id[0]
        s = None
        _p = None
        out = None
        obsnd = torch.zeros(batch, max_seq, heads, embedv)
        out_true_bnsd = torch.zeros(batch, max_seq, heads, embedv)
        kbsnd = k.view(layer_id + 1, batch, max_seq, kv_head, embed)
        vbsnd = v.view(layer_id + 1, batch, max_seq, kv_head, embedv)
        qbsnd = torch.zeros(batch, max_seq, heads, embed)
        self.encoder_logN = torch.tensor(
            [2.0] * self.max_seq).to(torch.float32)
        self.encoder_logN.uniform_(1, 2)
        self.decoder_logN = torch.tensor([2.0] * batch).to(torch.float32)
        self.decoder_logN.uniform_(1, 2)
        for idx in range(batch):
            if dynamic_batch and batch_state[idx] == 0 and not is_decoder:
                continue
            if dynamic_batch and batch_state[idx] == 0:
                output = torch.zeros([heads, q_s, embedv])
                output = torch.permute(output, (1, 0, 2))
                if out is None:
                    out = output
                else:
                    out = torch.cat((out, output), 0)
                q_offset += q_s
                k_offset += max_seq
                v_offset += max_seq
                continue
            q_s = q_seqlen[idx]
            kv_s = kv_seqlen[idx]
            q_slice = q[q_offset:q_offset + q_s][:]
            q_slice = q_slice.view(q_s, heads, embed)
            for q_s_idx in range(q_s):
                qbsnd[idx][q_s_idx] = q_slice[q_s_idx][:]
            q_slice = torch.permute(q_slice, (1, 0, 2))
            k_slice = k[layer_id][idx][:kv_s][:]
            k_slice = k_slice.view(kv_s, kv_head, embed)
            k_slice_t = torch.permute(k_slice, (1, 2, 0))
            v_slice = v[layer_id][idx][:kv_s][:]
            v_slice = v_slice.view(kv_s, kv_head, embedv)
            v_slice = torch.permute(v_slice, (1, 0, 2))

            score = self.group_mm_torch(heads, kv_head, q_slice, k_slice_t)
            if s is None:
                s = score.view([-1, ])
            else:
                s = torch.cat((s, score.view([-1, ])), 0)
            score = score * self.tor
            if self.scaleType == ScaleType.SCALE_LOGN_FP32.value:
                if is_decoder:
                    score *= self.decoder_logN[idx]
                else:
                    score *= self.encoder_logN[None, :q_s, None]
            if self.is_clamp == 1:
                clamp_min_brc = np.ones((score.shape)) * self.clamp_min
                clamp_max_brc = np.ones((score.shape)) * self.clamp_max
                score = np.float16(np.maximum(score, clamp_min_brc))
                score = torch.from_numpy(np.float16(
                    np.minimum(score, clamp_max_brc)))
            if is_mask:
                score = score + \
                    self.mask_info[1](self.mask, idx, q_s,
                                      kv_s) * self.post_mask_coff
            score = score.numpy().astype(np.float32)
            score_max = np.max(score, axis=-1)
            score = score - score_max.reshape((heads, q_s, 1))
            score_exp = np.exp(score)
            score_sum = np.sum(score_exp, axis=-1)

            if _p is None:
                _p = score_exp.astype(np.float32).reshape([-1, ])
            else:
                _p = np.concatenate(
                    (_p, score_exp.astype(np.float32).reshape([-1, ])), 0)

            p_true = (score_exp / score_sum.reshape((heads, q_s, 1)))
            p_true = torch.from_numpy(p_true)
            o_true = self.group_mm_torch(heads, kv_head, p_true, v_slice)
            o_true = o_true.view(heads, q_s, embedv)
            o_true = torch.permute(o_true, (1, 0, 2)).contiguous()

            p = p_true.to(torch.bfloat16)
            output = self.group_mm_torch(heads, kv_head, p, v_slice)
            output = output.view(heads, q_s, embedv)
            output = torch.permute(output, (1, 0, 2)).contiguous()
            if out is None:
                out = output
                out_true = o_true
            else:
                out = torch.cat((out, output), 0)
                out_true = torch.cat((out_true, o_true), 0)

            for i in range(0, q_s):
                obsnd[idx][i] = output[i]
                out_true_bnsd[idx] = out_true[i]
            q_offset += q_s
            k_offset += max_seq
            v_offset += max_seq
        obnsd = torch.permute(obsnd, (0, 2, 1, 3))
        out_true_bnsd = torch.permute(out_true_bnsd, (0, 2, 1, 3))
        self.qbnsd = torch.permute(qbsnd, (0, 2, 1, 3)).to(self.data_type)
        self.kbnsd = torch.permute(kbsnd, (0, 1, 3, 2, 4)).to(self.data_type)
        self.vbnsd = torch.permute(vbsnd, (0, 1, 3, 2, 4)).to(self.data_type)
        out = out.view(q_ntokens, heads * embedv)
        out_true = out_true.view(q_ntokens, heads * embedv)
        if (self.is_decoder == 1):
            self.golden_out = out
            self.golden_out_true = out_true.to(torch.float32)
        else:
            self.golden_out = obnsd.to(self.data_type)
            self.golden_out_true = out_true_bnsd.to(torch.float32)

        if self.no_cache:
            self.k = self.close_pack(
                self.k.to(torch.float32), kv_seqlen).to(self.data_type)
            self.v = self.close_pack(
                self.v.to(torch.float32), kv_seqlen).to(self.data_type)
        if self.long_seq:
            self.max_seq = 128
            self.gen_mask(self.batch, self.heads,
                          self.data_type, self.mask_type)

    def gen_seq_len(self, batch, seq_len):
        ntokens = sum(seq_len)
        return seq_len, ntokens

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

        limit_error = torch.maximum(
            torch.abs(golden * ratios[0]), torch.tensor(ratios[1]))
        strict_limit_error = torch.maximum(
            torch.abs(golden * ratios[2]), torch.tensor(ratios[3]))
        error_count = torch.gt(diff, limit_error).sum().item()
        strict_error_count = torch.gt(diff, strict_limit_error).sum().item()
        print(f"maxDiff {max_diff}")
        print("1/1000 Accuracy is %f", 1 - float(error_count) / out_len)
        print("5/1000 Accuracy is %f", 1 - float(strict_error_count) / out_len)
        if self.data_type == torch.bfloat16:
            print("accuracy is correct in old standard: %r", (float(strict_error_count) / out_len) <= ratios[2])
        else:
            print("accuracy is correct in old standard: %r", (float(strict_error_count) / out_len) <= ratios[0])

        calc_times = self.heads * self.max_seq + 4 
        if self.data_type == torch.bfloat16:
            if calc_times < 2048:
                error = 2**(-7)
            else:
                error = 2**(-6)
            error_threshold = torch.clamp(torch.abs(golden), min=1) * error
            res = (diff <= error_threshold).all().item()
            return res
        elif self.data_type == torch.float16:
            if calc_times < 2048:
                error = 2**(-8)
            else:
                error = 2**(-7)
            error_threshold = torch.clamp(torch.abs(golden), min=1) * error
            res = (diff <= error_threshold).all().item()
            return res
        else:
            if calc_times < 2048:
                error = 2**(-11)
            elif calc_times >= 2048 and calc_times < 16384:
                error = 2**(-10)
            else:
                error = 2**(-14)
            error_threshold = torch.clamp(torch.abs(golden), min=1) * error
            res = (diff <= error_threshold).all().item()
            return res

    def group_mm_torch(self, heads, group_num, A, B, dtype=torch.float32):
        group_head = heads // group_num
        score = None
        for i in range(group_num):
            group_score = torch.matmul(
                A[i * group_head: (i + 1) * group_head, :, :].to(dtype), B[i:(i + 1), :, :].to(dtype))
            if score is None:
                score = group_score
            else:
                score = torch.cat((score, group_score), 0)
        return score

    def golden_calc(self, in_tensors):
        golden_out = self.golden_out.clone().detach().requires_grad_(True).half().npu()
        return [golden_out]

    def golden_compare(self, out_tensors, golden_tensors):
        return self.compare_output_data(out_tensors[0].half(), golden_tensors[0].half(), [0.001, 0.001, 0.005, 0.005])

    @SupportedDevices(['Ascend910B'])
    def test_flash_attention_case_fa_encoder_withcache_bf16_alibi_sqrt_single_batch_without_tail_block(self):
        batch = 1
        kv_head = 1
        isdecoder = 0
        heads = 12
        embeddim = 128
        max_seq = 4096
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        q_seqlens = [128]
        kv_seqLen = [512]
        dynamic_batch = False
        block_size = 128
        num_blocks = 1024
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch=dynamic_batch,
                             is_decoder=isdecoder, batch=batch, kv_head=kv_head, heads=heads,
                             embeddim=embeddim, max_seq=max_seq, kv_seqLen=kv_seqLen,
                             data_type=data_type, is_alibi=True,
                             op_type=UNPAD_FLASH_ATTENTION_ENCODER_PREFIX_CACHE_ND, mask_type=MASK_TYPE_ALIBI_WITH_PREFIX_BATCH,
                             no_cache=True, is_sqrt=True, tor=tor, q_seqlens=q_seqlens,
                             num_blocks=num_blocks, block_size=block_size)
        self.gen_out_tensor()
        self.alibi_slopes *= -1
        mask = np.ones((256, 256)) * float("inf")
        mask = np.triu(mask, 1)
        self.mask = self.bias[:256, :256] * -1 + mask
        self.mask = self.mask.to(torch.bfloat16)
        q_seqlen = np.array(q_seqlens)
        q_seqlen = torch.from_numpy(q_seqlen).to(torch.int32).cpu()
        kv_seqLen = np.array(kv_seqLen)
        kv_seqLen = torch.from_numpy(kv_seqLen).to(torch.int32).cpu()
        self.q = self.q.view(sum(q_seqlens), heads, embeddim)
        self.k_cache = self.k_cache.view(num_blocks, block_size, kv_head * embeddim)
        self.v_cache = self.v_cache.view(num_blocks, block_size, kv_head * embeddim)
        output = torch.empty_like(self.q).npu()
        torch_npu.atb._npu_flash_attention_prefix_v2(self.q.npu(), self.k_cache.npu(), self.v_cache.npu(), self.block_tables.npu(), self.mask.to(data_type).npu(), q_seqlen, kv_seqLen, slopes=self.alibi_slopes.to(torch.float32).npu(), kernel_type=1, mask_type=5, num_kv_heads=1, num_heads=12, scale_value=tor, out=output)
        ratios = [0.001, 0.001, 0.005, 0.005]
        res = self.compare_output_data(output.cpu(), self.golden_out.cpu(), ratios)
        self.assertEqual(res, True)

if __name__ == "__main__":
    run_tests()

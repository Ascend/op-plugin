# Copyright (c) 2025, Huawei Technologies.All rights reserved.
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
import math
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices


class TestNpuCompress(TestCase):

    def softmax_numpy(self, sim):
        sim = sim.to(torch.float32).cpu().numpy()

        row_max = np.max(sim, axis=-1, keepdims=True)
        sim_sub = sim - row_max
        sim_sub = np.exp(sim_sub)
        row_sum = np.sum(sim_sub, axis=-1, keepdims=True)
        soft_res = sim_sub / row_sum
        return soft_res

    def softmax(self, x):
        x = x.astype(np.float32)
        x_max = x.max(axis=-1, keepdims=True)
        x_sub = x - x_max
        y = np.exp(x_sub)
        x_sum = y.sum(axis=-1, keepdims=True)
        ans = y
        return ans, x_sum, x_max

    def _t_broadcastKV_sigle(self, numHeads, numKeyValueHeads, kv_tensor, input_dtype):
        factor = numHeads // numKeyValueHeads
        kv_shape = kv_tensor.shape
        B = kv_shape[0]
        S = kv_shape[2]
        D = kv_shape[3]
        kv_res = np.zeros([B, numHeads, S, D], dtype=kv_tensor.dtype)
        for i in range(numHeads):
            j = i // factor
            kv_res[:, i:i + 1, :, :] = kv_tensor[:, j:j + 1, :, :]
        return kv_res, kv_res.shape

    def calcu_attention(self, q_part, k_part, v_part, scale_value, q_dtype):
        qkBmmRes = np.matmul(q_part, k_part.transpose(0, 1, 3, 2), dtype=np.float32)
        qkEleRes = qkBmmRes * scale_value

        softmax_res, softmax_sum, _ = self.softmax(qkEleRes)
        bmm2 = np.matmul(softmax_res.astype(np.float32), v_part.astype(np.float32),
                            dtype=np.float32)
        bmm2Res = bmm2 / softmax_sum

        return bmm2Res.reshape([1, bmm2Res.shape[1] * bmm2Res.shape[2], 1, bmm2Res.shape[-1]])

    # pylint:disable = huawei-too-many-arguments
    def generate_topk_indices_and_block_table(self, query, num_heads, num_key_value_heads, select_block_size, select_block_count, block_size, actual_seq_kvlen):

        batch = query.shape[0]
        numKeyValueHeads = num_key_value_heads
        numHeads = num_heads

        topk_indices_shape = [batch, numKeyValueHeads, select_block_count]
        topk_indices = np.random.uniform(-1, -1, topk_indices_shape).astype(np.int32)

        max_k_seqlen = np.max(actual_seq_kvlen)
        max_num_blocks_per_seq = math.ceil(max_k_seqlen / block_size)

        block_table_shape = [batch, max_num_blocks_per_seq]
        blockNum = block_table_shape[0] * block_table_shape[1]
        blockNumPerBlock = []
        block_num_min = 0
        for batch_index in range(batch):
            act_seqlen = actual_seq_kvlen[batch_index]
            blockNumPerBlock.append(math.ceil(act_seqlen / block_size))
            block_num_min += math.ceil(act_seqlen / block_size)
        if block_num_min > blockNum:
            raise RuntimeError("block_num_min should be greater than blockNum.")
        block_idx_list = np.arange(0, blockNum, 1)
        np.random.seed(0)
        block_idx_list = np.random.permutation(block_idx_list).astype(np.int32)
        block_idx = 0
        block_table = [-1] * block_table_shape[1]
        block_table = np.tile(block_table, (block_table_shape[0], 1)).astype(np.int32)
        block_table_batch_idx = 0
        for idx in blockNumPerBlock:
            for j in range(idx):
                block_table[block_table_batch_idx][j] = (block_idx_list[block_idx])
                block_idx += 1
            block_table_batch_idx += 1

        max_act_seqlen = -1

        act_seqlen_topk_list = []
        np.random.seed(1)
        for batch_idx in range(batch):
            torch.manual_seed(0 + batch_idx)
            act_seqlen = actual_seq_kvlen[batch_idx]
            max_act_seqlen = max(act_seqlen, max_act_seqlen)
            max_act_seqlen = max(act_seqlen, max_act_seqlen)
            # 计算当前batch的有效块数
            valid_blocks_max = math.ceil(act_seqlen / select_block_size)
            act_seqlen_tail = act_seqlen - (valid_blocks_max - 1) * select_block_size
            valid_blocks_topk = np.random.randint(0, valid_blocks_max)
            if valid_blocks_topk == 0:
                valid_blocks_topk = 1

            act_seqlen_topk = valid_blocks_topk * select_block_size if (
                        valid_blocks_topk < valid_blocks_max) else max_act_seqlen

            block_indices = torch.randperm(valid_blocks_max).numpy().astype(np.int32)  # old
            for topk_idx in range(valid_blocks_topk):
                if block_indices[topk_idx] == (valid_blocks_max - 1):
                    act_seqlen_topk = (valid_blocks_topk - 1) * select_block_size + act_seqlen_tail
                    break

            act_seqlen_topk_list.append(act_seqlen_topk)
            for numHeads_idx in range(num_key_value_heads):
                topk_indices[batch_idx, numHeads_idx, :valid_blocks_topk] = block_indices[0:valid_blocks_topk]
                for vaild_idx in range(valid_blocks_topk, select_block_count):
                    topk_indices[batch_idx, numHeads_idx, vaild_idx] = -1

        return topk_indices, block_table, act_seqlen_topk_list

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    # pylint:disable = huawei-too-many-arguments
    def nsa_select_attention_infer_golden(self, q, k, v, topk_indices, scale, num_heads, num_key_value_heads,
                                            selected_block_size, selected_block_count, block_size, atten_mask,
                                            block_table, actual_seq_qlen, actual_seq_kvlen, layout, act_seqlen_topk_list):

        batch = q.shape[0]
        if layout == 'BSH':
            headDim = q.shape[2] // num_heads
            headDimV = v.shape[2] // num_key_value_heads
            q = q.reshape([batch, 1, num_heads, headDim])
            k = k.reshape([k.shape[0], block_size, num_key_value_heads, headDim])
            v = v.reshape([v.shape[0], block_size, num_key_value_heads, headDimV])
        q = q.numpy()
        k = k.numpy()
        v = v.numpy()
        q_shape = q.shape
        k_shape = k.shape
        v_shape = v.shape
        q_dtype = q.dtype
        numKeyValueHeads = num_key_value_heads
        numHeads = num_heads
        headDim = q_shape[-1]
        headDimV = v_shape[-1]
        max_act_seqlen = max(actual_seq_kvlen)
        batch = q_shape[0]

        k_cache = np.zeros([batch, max_act_seqlen, numKeyValueHeads, headDim])
        v_cache = np.zeros([batch, max_act_seqlen, numKeyValueHeads, headDimV])
        batch = q_shape[0]
        for batch_idx in range(batch):
            act_seqlen = actual_seq_kvlen[batch_idx]
            act_seqlen_topk = act_seqlen_topk_list[batch_idx]
            block_table_cur = block_table[batch_idx]
            for numHeadsIdx in range(numKeyValueHeads):
                idIntopKRecord = -1
                x = 0
                currentTopDeal = -1
                topKIdOffset = 0
                nCopyRowCount = 128
                nActCopyRowCount = 128
                copyRowTimes = math.ceil(act_seqlen_topk / nCopyRowCount)
                for_loop_break = False
                for copyRowIdx in range(copyRowTimes):
                    if for_loop_break:
                        break
                    copyFinishRowCnt = 0
                    curSeqIdx = 128 * copyRowIdx
                    if copyRowIdx == copyRowTimes - 1:
                        nActCopyRowCount = act_seqlen_topk - nCopyRowCount * (copyRowTimes - 1)
                    while (copyFinishRowCnt < nActCopyRowCount):
                        if x == currentTopDeal:
                            topKIdOffset += 1
                        idIntopk = topk_indices[batch_idx, numHeadsIdx, topKIdOffset]
                        if idIntopk == -1:
                            for_loop_break = True
                            break
                        else:
                            if (idIntopKRecord != idIntopk):
                                x = 0 # 当前topk已经处理的数据
                                idIntopKRecord = idIntopk
                                global_start = idIntopk * selected_block_size
                                global_end = act_seqlen if (global_start + selected_block_size > act_seqlen) else (global_start + selected_block_size)
                                currentTopDeal = global_end - global_start # 当前topk能过处理的数据
                            global_start = idIntopk * selected_block_size
                            global_end = act_seqlen if (global_start + selected_block_size > act_seqlen) else (global_start + selected_block_size)
                            global_start += x
                            start_offset = global_start % block_size
                            start_block_idx = global_start // block_size
                            end_block_idx = (global_end - 1) // block_size

                            reaminRowCnt = start_offset
                            copyRowCnt = global_end - global_start if (start_block_idx == end_block_idx) else (block_size - reaminRowCnt)
                            if (copyFinishRowCnt + copyRowCnt > nActCopyRowCount):
                                copyRowCnt = nActCopyRowCount - copyFinishRowCnt
                            block_idx = block_table_cur[start_block_idx]
                            k_cache[batch_idx, curSeqIdx:curSeqIdx + copyRowCnt, numHeadsIdx, :] = k[block_idx, start_offset:(start_offset + copyRowCnt), numHeadsIdx, :]
                            v_cache[batch_idx, curSeqIdx:curSeqIdx + copyRowCnt, numHeadsIdx, :] = v[block_idx, start_offset:(start_offset + copyRowCnt), numHeadsIdx, :]
                            copyFinishRowCnt += copyRowCnt
                            x += copyRowCnt
                            curSeqIdx += copyRowCnt

        group = numHeads // numKeyValueHeads
        y = np.zeros([batch, numHeads, 1, v.shape[-1]])
        q = q.transpose(0, 2, 1, 3)
        k_cache = k_cache.transpose(0, 2, 1, 3)
        v_cache = v_cache.transpose(0, 2, 1, 3)
        k_cache, _ = self._t_broadcastKV_sigle(numHeads, numKeyValueHeads, k_cache, q_dtype)

        v_cache, _ = self._t_broadcastKV_sigle(numHeads, numKeyValueHeads, v_cache, q_dtype)
        for batch_index in range(batch):
            act_seqlen = act_seqlen_topk_list[batch_index]
            y[batch_index:(batch_index + 1), :, :, :] = self.calcu_attention(q[batch_index:(batch_index + 1), :, :, :], k_cache[batch_index:(batch_index + 1), :, 0:act_seqlen, :], v_cache[batch_index:(batch_index + 1), :, 0:act_seqlen, :], scale, q_dtype)
        attention_out = y.reshape(batch, 1, numHeads, v.shape[-1])
        if layout == 'BSH':
            attention_out = attention_out.reshape([batch, 1, numHeads * headDimV])
        if (q_dtype == np.float16):
            attention_out = attention_out.astype(np.float16)
        else:
            attention_out = attention_out.astype(np.float32)

        attention_out = torch.from_numpy(attention_out)
        return attention_out

    # pylint:disable = huawei-too-many-arguments
    def cpu_op_exec(self, query, key, value, topk_indices, scale_value, head_num, key_value_head_num, select_block_size, select_block_count, page_block_size, layout, atten_mask, block_table, actual_seq_qlen, actual_seq_kvlen, act_seqlen_topk_list):
        output = self.nsa_select_attention_infer_golden(query, key, value, topk_indices, scale_value, head_num, key_value_head_num, select_block_size, select_block_count, page_block_size, atten_mask, block_table, actual_seq_qlen, actual_seq_kvlen, layout, act_seqlen_topk_list)
        return output

    # pylint:disable = huawei-too-many-arguments
    def npu_op_exec(self, query, key, value, topk_indices, scale_value, head_num, key_value_head_num, select_block_size, select_block_count, page_block_size, layout, atten_mask, block_table, actual_seq_qlen, actual_seq_kvlen):
        output = torch_npu.npu_nsa_select_attention_infer(query, key, value, topk_indices, scale_value, head_num, key_value_head_num, select_block_size, select_block_count, page_block_size, layout=layout, atten_mask=atten_mask, block_table=block_table, actual_seq_qlen=actual_seq_qlen, actual_seq_kvlen=actual_seq_kvlen)
        return output

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_npu_nsa_compress(self):
        query = torch.randn([123, 1, 768], dtype=torch.float16)
        key = torch.randn([246, 64, 384], dtype=torch.float16)
        value = torch.randn([246, 64, 256], dtype=torch.float16)
        topk_indices = torch.randn([123, 2, 2], dtype=torch.float16)
        block_table = torch.randn([123, 2], dtype=torch.float16)
        scale_value = 2.0
        head_num = 4
        key_value_head_num = 2
        select_block_size = 64
        select_block_count = 2
        page_block_size = 64
        layout = 'BSH'
        actual_seq_qlen = None
        actual_seq_kvlen = [82] * query.size(0)
        atten_mask = None

        topk_indices, block_table, act_seqlen_topk_list = self.generate_topk_indices_and_block_table(query, head_num, key_value_head_num, select_block_size, select_block_count, page_block_size, actual_seq_kvlen)

        npuout = self.npu_op_exec(query.npu(), key.npu(), value.npu(), torch.from_numpy(topk_indices).npu(), scale_value, head_num, key_value_head_num, select_block_size, select_block_count, page_block_size, layout, atten_mask, torch.from_numpy(block_table).npu(), actual_seq_qlen, actual_seq_kvlen)

        gloden_out = self.cpu_op_exec(query, key, value, topk_indices, scale_value, head_num, key_value_head_num,
                                          select_block_size, select_block_count, page_block_size, layout, atten_mask,
                                          block_table, None, actual_seq_kvlen, act_seqlen_topk_list)

        self.assertRtolEqual(gloden_out.to(torch.float16).numpy(), npuout.to(torch.float16).cpu().numpy())

if __name__ == "__main__":
    run_tests()

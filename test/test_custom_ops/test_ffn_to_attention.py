import os
import unittest
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
import time
import random

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class TestFFNToAttention(TestCase):

    @classmethod
    def _init_dist_hccl(cls, rank, world_size, ep_world_size, tp_world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '50000'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        ep_ranks_list = []
        tp_ranks_list = []
        for i in range(tp_world_size):
            ep_ranks_list.append(list(range(i, world_size, tp_world_size)))
        for i in range(ep_world_size):
            tp_ranks_list.append(list(range(i * tp_world_size, (i + 1) * tp_world_size)))
        for i in range(tp_world_size):
            ep_group = dist.new_group(backend='hccl', ranks=ep_ranks_list[i])
            if rank in ep_ranks_list[i]:
                ep_group_tmp = ep_group
        for i in range(ep_world_size):
            tp_group = dist.new_group(backend='hccl', ranks=tp_ranks_list[i])
            if rank in tp_ranks_list[i]:
                tp_group_tmp = tp_group
        return dist, ep_group_tmp, tp_group_tmp

    @classmethod
    def _test_ffn_to_attention(cls, rank, input_list):
        import torchair
        
        bs, k, h, micro_batch_num, attention_worker_num, ffn_worker_num, expert_num, world_size, window_size, ep_world_size, tp_world_size, input_type, \
        x, session_ids, micro_batch_ids, token_ids, expert_offsets, actual_token_num, attn_rank_table, token_info_table_shape, token_data_shape, init_pg, c2p, p2c = input_list
        
        # 准备
        pg, ep_group, tp_group = init_pg(rank, world_size, ep_world_size, tp_world_size)
        ep_hcomm_name = ep_group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)

        if rank >= attention_worker_num:
            target_ranks = list(range(attention_worker_num))
        else:
            target_ranks = list(range(attention_worker_num, world_size))
        ep_group._get_backend(torch.device('npu'))._window_register_and_exchange(window_size, target_ranks)

        # 调用算子
        if rank >= attention_worker_num:
            x = x.npu()
            session_ids = session_ids.to(torch.int32).npu()
            micro_batch_ids = micro_batch_ids.to(torch.int32).npu()
            token_ids = token_ids.to(torch.int32).npu()
            expert_offsets = expert_offsets.to(torch.int32).npu()
            attn_rank_table = attn_rank_table.to(torch.int32).npu()
            
            torch_npu.npu_ffn_to_attention(x=x,
                session_ids = session_ids,
                micro_batch_ids = micro_batch_ids,
                token_ids = token_ids,
                expert_offsets = expert_offsets,
                actual_token_num = torch.tensor([actual_token_num], dtype=torch.int64).npu(),
                attn_rank_table = attn_rank_table,
                group = ep_hcomm_name,
                world_size = world_size,
                token_info_table_shape = token_info_table_shape,
                token_data_shape = token_data_shape)
            c2p.put([rank,
                [0, 0]
            ])
        else:
            time.sleep(5)
            window_addr = ep_group._get_backend(torch.device('npu'))._get_window_mem().data_ptr()
            token_info_npu = torchair.llm_datadist.create_npu_tensors(
                [micro_batch_num, bs, expert_num], torch.int32, [window_addr]
            )[0]
            token_info_size = (token_info_npu.element_size() * token_info_npu.numel() + 512 - 1) // 512 * 512  # 512对齐
            token_data_npu = torchair.llm_datadist.create_npu_tensors(
                [micro_batch_num, bs, expert_num, h], input_type, [window_addr + token_info_size]
            )[0]

            token_info_cpu = token_info_npu.cpu()
            token_data_cpu = token_data_npu.cpu()
            c2p.put([rank, 
                [token_info_cpu, token_data_cpu]
            ])

        p2c.get()

    def _test_multiprocess(self, f, init_pg, input_list):
        bs, k, h, micro_batch_num, attention_worker_num, ffn_worker_num, expert_num, world_size, window_size, ep_world_size, tp_world_size, input_type, \
            x, session_ids, micro_batch_ids, token_ids, expert_offsets, actual_token_num, attn_rank_table, token_info_table_shape, token_data_shape, token_info_golden, token_data_golden, token_data_mask = input_list

        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)
        p2c = ctx.Queue(world_size)
        ps = []

        for i in range(world_size):
            p = ctx.Process(
                target=f,
                args=(i, [bs, k, h, micro_batch_num, attention_worker_num, ffn_worker_num, expert_num, world_size, window_size, ep_world_size, tp_world_size, input_type,
                            x, session_ids, micro_batch_ids, token_ids, expert_offsets, actual_token_num, attn_rank_table, token_info_table_shape, token_data_shape, init_pg, c2p, p2c]))
            p.start()
            ps.append(p)

        token_data_list = [None] * attention_worker_num
        token_info_list = [None] * attention_worker_num
        for _ in range(world_size):
            rank, output = c2p.get()
            if rank < attention_worker_num:
                token_info_list[rank] = output[0]
                token_data_list[rank] = output[1]
        
        for rank in range(attention_worker_num):
            self.assertEqual(token_info_list[rank], token_info_golden[rank],
                             ("rank {} Expect receive tensor {} but got {}.").format(rank, token_info_golden[rank], token_info_list[rank]))

        for rank in range(attention_worker_num):
             self.assertEqual(token_data_list[rank], token_data_golden[rank],
                             ("rank {} Expect receive tensor {} but got {}.").format(rank, token_data_list[rank], token_data_golden[rank]))

        for _ in range(world_size):
            p2c.put(0)

        for p in ps:
            p.join()

    def _construct_excepted_result(self, attention_worker_num, micro_batch_num, bs, expert_num, h, x, session_ids, micro_batch_ids, batch_ids, expert_offsets, actual_token_num):
        token_info_golden = torch.zeros(attention_worker_num, micro_batch_num, bs, expert_num)
        token_data_golden = torch.zeros(attention_worker_num, micro_batch_num, bs, expert_num, h)
        token_data_mask = torch.zeros(attention_worker_num, micro_batch_num, bs, expert_num, h, dtype = torch.bool)

        for token_id in range(actual_token_num):
            cur_attention_rank = session_ids[token_id]
            cur_micro_batch_id = micro_batch_ids[token_id]
            cur_batch_id = batch_ids[token_id]
            cur_expert_offset = expert_offsets[token_id]

            token_info_golden[cur_attention_rank, cur_micro_batch_id, cur_batch_id, cur_expert_offset] = 1
            token_data_golden[cur_attention_rank, cur_micro_batch_id, cur_batch_id, cur_expert_offset, :] = x[token_id]
            token_data_mask[cur_attention_rank, cur_micro_batch_id, cur_batch_id, cur_expert_offset, :] = torch.ones(h, dtype = torch.bool)

        return token_info_golden, token_data_golden, token_data_mask


    @unittest.skip("skip test_ffn_to_attention, no module named torchair")
    @skipIfUnsupportMultiNPU(16)
    @SupportedDevices(['Ascend910_'])
    def test_ffn_to_attention(self):
        ep_world_size = 16
        tp_world_size = 1
        world_size = ep_world_size * tp_world_size
        bs = 16
        h = 7168
        k = 8
        micro_batch_num = 1
        shared_expert_num = 1
        moe_expert_num = 8
        expert_num = shared_expert_num + moe_expert_num
        attention_worker_num = 11
        ffn_worker_num = 5
        actual_token_num = bs * micro_batch_num * attention_worker_num * expert_num
        input_type = torch.bfloat16
        window_size = 1024 * 1024 * 200

        x = torch.empty([actual_token_num, h], dtype = input_type).uniform_(-1024, 1024)
        session_ids = torch.repeat_interleave(torch.arange(attention_worker_num, dtype = torch.int32), repeats = int(actual_token_num / attention_worker_num))
        micro_batch_ids = torch.repeat_interleave(torch.arange(micro_batch_num, dtype = torch.int32), repeats = bs * expert_num).repeat(attention_worker_num)
        batch_ids = torch.repeat_interleave(torch.arange(bs, dtype = torch.int32), repeats = expert_num).repeat(micro_batch_num * attention_worker_num)
        expert_offsets = torch.arange(expert_num, dtype = torch.int32).repeat(int(actual_token_num / expert_num))
        token_info_table_shape = [micro_batch_num, bs, expert_num]
        token_data_shape = [micro_batch_num, bs, expert_num, h]
        attn_rank_table = torch.arange(attention_worker_num, dtype = torch.int32)

        token_info_golden, token_data_golden, token_data_mask = self._construct_excepted_result(attention_worker_num, micro_batch_num, bs, expert_num, h, x, session_ids, micro_batch_ids, batch_ids, expert_offsets, actual_token_num)

        self._test_multiprocess(TestFFNToAttention._test_ffn_to_attention,
                TestFFNToAttention._init_dist_hccl, [bs, k, h, micro_batch_num, attention_worker_num, ffn_worker_num, expert_num, world_size, window_size, ep_world_size, tp_world_size, input_type, \
                                                        x, session_ids, micro_batch_ids, batch_ids, expert_offsets, actual_token_num, attn_rank_table, token_info_table_shape, token_data_shape, token_info_golden, token_data_golden, token_data_mask])


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)
    run_tests()
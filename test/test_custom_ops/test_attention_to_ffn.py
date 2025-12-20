import os
import unittest
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
import copy
import time
import random

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class TestAttentionToFFN(TestCase):

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
    def _test_attention_to_ffn(cls, rank, input_list):
        import torchair

        bs, k, h, micro_batch_num, default_micro_batch_id, default_layer_id, attention_worker_num, ffn_worker_num, world_size, ep_world_size, tp_world_size, moe_expert_num, window_size, input_type, \
        x_list, expert_ids_list, expert_rank_table, ffn_token_info_table_shape, ffn_token_data_shape, attn_token_info_table_shape, init_pg, c2p, p2c = input_list

        # 准备
        pg, ep_group, tp_group = init_pg(rank, world_size, ep_world_size, tp_world_size)
        ep_hcomm_name = ep_group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)

        if rank >= ffn_worker_num:
            target_ranks = list(range(ffn_worker_num))
        else:
            target_ranks = list(range(ffn_worker_num, world_size))
        ep_group._get_backend(torch.device('npu'))._window_register_and_exchange(window_size, target_ranks)

        # 调用算子
        if rank >= ffn_worker_num:
            x = x_list[rank - ffn_worker_num].to(input_type).npu()
            session_id = torch.tensor([rank - ffn_worker_num], dtype = torch.int32).npu()
            micro_batch_id = torch.tensor([default_micro_batch_id], dtype = torch.int32).npu()
            layer_id = torch.tensor([default_layer_id], dtype = torch.int32).npu()
            expert_ids = expert_ids_list[rank - ffn_worker_num].to(torch.int32).npu()
            expert_rank_table_npu = expert_rank_table.to(torch.int32).npu()
            
            torch_npu.npu_attention_to_ffn(x=x,
                session_id = session_id,
                micro_batch_id = micro_batch_id,
                layer_id = layer_id,
                expert_ids = expert_ids,
                expert_rank_table = expert_rank_table_npu,
                group = ep_hcomm_name,
                world_size = world_size,
                ffn_token_info_table_shape = ffn_token_info_table_shape,
                ffn_token_data_shape = ffn_token_data_shape,
                attn_token_info_table_shape = attn_token_info_table_shape,
                moe_expert_num = moe_expert_num,
                scales=None,
                quant_mode=0)
            c2p.put([rank,
                [0, 0]
            ])
        else:
            time.sleep(5)
            window_addr = ep_group._get_backend(torch.device('npu'))._get_window_mem().data_ptr()
            token_info_npu = torchair.llm_datadist.create_npu_tensors(
                [attention_worker_num, micro_batch_num, 1 + 1 + bs * (k + 1)], torch.int32, [window_addr]
            )[0]
            token_info_size = (token_info_npu.element_size() * token_info_npu.numel() + 512 - 1) // 512 * 512  # 512对齐
            token_data_npu = torchair.llm_datadist.create_npu_tensors(
                [attention_worker_num, micro_batch_num, bs, k + 1, h], input_type, [window_addr + token_info_size]
            )[0]

            token_info_cpu = token_info_npu.cpu()
            token_data_cpu = token_data_npu.cpu()
            c2p.put([rank, 
                [token_info_cpu, token_data_cpu]
            ])
        
        p2c.get()

    def _test_multiprocess(self, f, init_pg, input_list):
        bs, k, h, micro_batch_num, default_micro_batch_id, default_layer_id, attention_worker_num, ffn_worker_num, world_size, ep_world_size, tp_world_size, moe_expert_num, window_size, input_type, \
            x, expert_ids, expert_rank_table, ffn_token_info_table_shape, ffn_token_data_shape, attn_token_info_table_shape, token_info_golden, token_data_golden, token_data_mask = input_list
        
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)
        p2c = ctx.Queue(world_size)
        ps = []

        for i in range(world_size):
            p = ctx.Process(
                target=f,
                args=(i, [bs, k, h, micro_batch_num, default_micro_batch_id, default_layer_id, attention_worker_num, ffn_worker_num, world_size, ep_world_size, tp_world_size, moe_expert_num, window_size, input_type,
                            x, expert_ids, expert_rank_table, ffn_token_info_table_shape, ffn_token_data_shape, attn_token_info_table_shape, init_pg, c2p, p2c]))
            p.start()
            ps.append(p)

        token_data_list = [None] * ffn_worker_num
        token_info_list = [None] * ffn_worker_num
        for _ in range(world_size):
            rank, output = c2p.get()
            if rank < ffn_worker_num:
                token_info_list[rank] = output[0]
                token_data_list[rank] = output[1]
        
        for rank in range(ffn_worker_num):
            self.assertEqual(token_info_list[rank], token_info_golden[rank],
                             ("rank {} Expect receive tensor {} but got {}.").format(rank, token_info_golden[rank], token_info_list[rank]))

        for rank in range(ffn_worker_num):
             self.assertEqual(token_data_list[rank], token_data_golden[rank],
                             ("rank {} Expect receive tensor {} but got {}.").format(rank, token_data_list[rank], token_data_golden[rank]))
        
        for _ in range(world_size):
            p2c.put(0)

        for p in ps:
            p.join()

    def _construct_excepted_result(self, x_golden, expert_ids_golden, expert_rank_table, ffn_worker_num, attention_worker_num, micro_batch_num, bs, k, h, X, expert_num, default_layer_id, default_micro_batch_id):
        token_info_golden = torch.zeros(ffn_worker_num, attention_worker_num, micro_batch_num, 1 + 1 + bs * (k + 1), dtype = torch.int32)
        token_data_golden = torch.zeros(ffn_worker_num, attention_worker_num, micro_batch_num, bs, k + 1, h)
        token_data_mask = torch.zeros(ffn_worker_num, attention_worker_num, micro_batch_num, bs, k + 1, h, dtype = torch.bool)

        def load_banance(session_id, layer_id, expert_id):
            idx = session_id % expert_rank_table[layer_id, expert_id, 0]
            dst_rank_id = expert_rank_table[layer_id, expert_id, 1 + 2 * idx]
            dst_expert_id = expert_rank_table[layer_id, expert_id, 2 + 2 * idx]
            return (dst_rank_id, dst_expert_id)

        def attn2ffn(row_idx, x_i, expert_id, attn_id):
            nonlocal token_info_golden, token_data_golden, token_data_mask
            # 处理moe专家
            for topk_id, expert_i in enumerate(expert_id):
                (dst_rank_id, dst_expert_id) = load_banance(attn_id, default_layer_id, expert_i)
                token_data_golden[dst_rank_id, attn_id, default_micro_batch_id, row_idx, topk_id, :h] = x_i
                token_data_mask[dst_rank_id, attn_id, default_micro_batch_id, row_idx, topk_id, :h] = torch.ones(h, dtype = torch.bool)
                token_info_golden[dst_rank_id, attn_id, default_micro_batch_id, 0] = 1
                token_info_golden[dst_rank_id, attn_id, default_micro_batch_id, row_idx * (k + 1) + topk_id + 2] = dst_expert_id

            (dst_rank_id, dst_expert_id) = load_banance(attn_id, default_layer_id, expert_num - 1)
            token_data_golden[dst_rank_id, attn_id, default_micro_batch_id, row_idx, expert_num - 1, :h] = x_i
            token_data_mask[dst_rank_id, attn_id, default_micro_batch_id, row_idx, expert_num - 1, :h] = torch.ones(h, dtype = torch.bool)
            token_info_golden[dst_rank_id, attn_id, default_micro_batch_id, 0] = 1
            token_info_golden[dst_rank_id, attn_id, default_micro_batch_id, row_idx * (k + 1) + k + 2] = dst_expert_id

        for attn_id in range(attention_worker_num):
            for token_id in range(bs):
                attn2ffn(token_id, x_golden[attn_id, X - 1, token_id], expert_ids_golden[attn_id, X - 1, token_id], attn_id)
        token_info_golden[:, :, :, 0] = 1
        
        return token_info_golden, token_data_golden, token_data_mask

    def gen_expert_rank_table(self, moe_rank_ffn_num, experts_per_rank, moe_expert_num, layer_num, ffn_worker_num):
        data = []
        max_len = 0
        valid_rank_id = [i for i in range(moe_rank_ffn_num)] * 2
        valid_exp_id = list(range(moe_expert_num))
        cards = {}

        for i in range(moe_rank_ffn_num):
            start = i * experts_per_rank
            end = start + experts_per_rank
            cards[i] = valid_exp_id[start:end]
        
        for _ in range(layer_num):
            layer_e1 = []
            for _ in range(moe_expert_num):
                cards_h = copy.deepcopy(cards)
                valid_rank_id_h = copy.deepcopy(valid_rank_id)

                i = random.randint(1, moe_rank_ffn_num)
                item = [i]
                for _ in range(i):
                    num_rank_id = random.choice(valid_rank_id_h)
                    item.append(num_rank_id)
                    valid_rank_id_h.remove(num_rank_id)

                    num_exp_id = random.choice(cards_h[num_rank_id])
                    item.append(num_exp_id)
                    cards_h[num_rank_id].remove(num_exp_id)
                
                layer_e1.append(item)
                max_len = max(max_len, len(item))
            layer_e1.append([1, ffn_worker_num - 1, moe_expert_num])
            data.append(layer_e1)
        padded_data = []
        for layer in data:
            padded_layer = []
            for item in layer:
                item_padded = item + [0] * (max_len - len(item))
                padded_layer.append(item_padded)
            padded_data.append(padded_layer)

        return torch.tensor(padded_data)

    @unittest.skip("skip test_attention_to_ffn, no module named torchair")
    @skipIfUnsupportMultiNPU(16)
    @SupportedDevices(['Ascend910_'])
    def test_attention_to_ffn(self):
        ep_world_size = 16
        tp_world_size = 1
        world_size = ep_world_size * tp_world_size
        moe_rank_ffn_num = 4
        experts_per_rank = 2
        X = 1
        bs = 16
        h = 7168
        k = 8
        micro_batch_num = 1
        layer_num = 1
        default_layer_id = 0
        default_micro_batch_id = 0
        shared_expert_num = 1
        moe_expert_num = 8
        expert_num = shared_expert_num + moe_expert_num
        attention_worker_num = 11
        ffn_worker_num = 5
        input_type = torch.bfloat16
        window_size = 1024 * 1024 * 200
        
        x = torch.empty([attention_worker_num, X, bs, h], dtype = input_type).uniform_(-1024, 1024)
        ep_expert_ids = [torch.randperm(moe_expert_num, dtype = torch.int32)[:k] for _ in range(bs * attention_worker_num)]
        expert_ids = torch.cat(ep_expert_ids).view([attention_worker_num, X, bs, k])
        expert_rank_table = self.gen_expert_rank_table(moe_rank_ffn_num, experts_per_rank, moe_expert_num, layer_num, ffn_worker_num)
        ffn_token_info_table_shape = [attention_worker_num, micro_batch_num, 1 + 1 + bs * (k + 1)]
        ffn_token_data_shape = [attention_worker_num, micro_batch_num, bs, k + 1, h]
        attn_token_info_table_shape = [micro_batch_num, bs, k + 1]

        token_info_golden, token_data_golden, token_data_mask = self._construct_excepted_result(x, expert_ids, expert_rank_table, ffn_worker_num, attention_worker_num, micro_batch_num, bs, k, h, X, expert_num, default_layer_id, default_micro_batch_id)

        self._test_multiprocess(TestAttentionToFFN._test_attention_to_ffn,
                TestAttentionToFFN._init_dist_hccl, [bs, k, h, micro_batch_num, default_micro_batch_id, default_layer_id, attention_worker_num, ffn_worker_num, world_size, ep_world_size, tp_world_size, moe_expert_num, window_size, input_type,
                            x, expert_ids, expert_rank_table, ffn_token_info_table_shape, ffn_token_data_shape, attn_token_info_table_shape, token_info_golden, token_data_golden, token_data_mask])

if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)
    run_tests()

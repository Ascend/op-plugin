import os
import unittest
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU

class TestDistributeBarrier(TestCase):
    @classmethod
    def _init_dist_hccl(cls, rank, world_size, ep_world_size, tp_world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '50000'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size = world_size, rank = rank)
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
    def gen_elastic_info(cls, is_elastic, world_size, shared_expert_rank_num, share_broken_card_num,
                         moe_broken_card_num, local_moe, ep_world_size, moe_expert_num):
        if not is_elastic: return None
        elastic_info = torch.zeros(4 + 2 * ep_world_size, dtype = torch.int32)
        elastic_info[0] = is_elastic
        elastic_info[1] = world_size - share_broken_card_num -moe_broken_card_num
        elastic_info[2] = shared_expert_rank_num
        elastic_info[3] = moe_expert_num - local_moe*(moe_broken_card_num + share_broken_card_num)
        table1 = [-1] * ep_world_size
        table2 = [-1] * ep_world_size

        if is_elastic:
            _ = [i for i in range(shared_expert_rank_num)]
            __= [i for i in range(shared_expert_rank_num, world_size)]
            random_seed=24
            torch.manual_seed(random_seed)
            elastic_rank = random.sample(_, elastic_info[2]) + random.sample(__, elastic_info[1] - elastic_info[2])
            elastic_rank.sort()

        for local_rank_id, ep_rank_id in enumerate(elastic_rank):
            if ep_rank_id < ep_world_size:
                table1[ep_rank_id] = local_rank_id
                table2[local_rank_id] = ep_rank_id
        for i in range(ep_world_size):
            elastic_info[4 + i] = table1[i]
        for i in range(ep_world_size):
            elastic_info[4 + ep_world_size + i] = table2[i]
        assert elastic_info.shape[0] == 4 + 2 * ep_world_size

        if is_elastic:
            table1 = elastic_info[4 : 4 + ep_world_size]
            table2 = elastic_info[4 + ep_world_size: 4 + 2 * ep_world_size]
        return elastic_info
    
    @classmethod
    def _test_npu_distribute_barrier(cls, rank, x_ref, time_out, elastic_info,
                                         ep_world_size, tp_world_size, init_pg, c2p, p2c ):
        
        pg, ep_group, tp_group = init_pg(rank, ep_world_size * tp_world_size, ep_world_size, tp_world_size)
        ep_hcomm_name = ep_group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
        tp_hcomm_name = tp_group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)

        if rank in elastic_info[4 + ep_world_size: 4 + 2 * ep_world_size]:
            out = torch_npu._npu_distribute_barrier(x_ref = x_ref.npu(),
                                                   time_out = time_out.npu(),
                                                   elastic_info = elastic_info.npu(),
                                                   group = ep_hcomm_name,
                                                   world_size = ep_world_size)
        else:
            out = None
        if out is not None:
            c2p.put((rank, out.cpu()))
        else:
            c2p.put((rank, None))
        p2c.get()

    def _test_multiprocess(self, f, init_pg, input_list):
        expt_out_list, x_ref, time_out, elastic_info, \
            ep_world_size, tp_world_size = input_list
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(ep_world_size * tp_world_size)
        p2c = ctx.Queue(ep_world_size * tp_world_size)
        ps = []

        for i in range(ep_world_size * tp_world_size):
            p = ctx.Process(
                target=f,
                args=(i, x_ref, time_out, elastic_info, ep_world_size, tp_world_size, init_pg, c2p, p2c))
            p.start()
            ps.append(p)

        for _ in range(ep_world_size * tp_world_size):
            rank, output = c2p.get()
            self.assertEqual(output, expt_out_list[rank],
                             ("rank {} Expect receive tensor {} but got {}.").format(rank, expt_out_list[rank], output))

        for _ in range(ep_world_size * tp_world_size):
            p2c.put(0)

        for p in ps:
            p.join()

    def _construct_excepted_result(self, x_ref, elastic_info, ep_world_size): 
        out_list = []
        for i in elastic_info[4 + ep_world_size: 4 + 2 * ep_world_size]:
            if i == -1:
                out_list.append(None)
            else:
                out_list.append(x_ref)
        return out_list
    
    @skipIfUnsupportMultiNPU(16)
    @SupportedDevices(['Ascend910_93', 'Ascend950'])
    def test_npu_distribute_barrier(self):
        ep_world_size = 8
        tp_world_size = 1
        bs = 8
        h = 7168
        k = 7
        shared_broken_card_num = 0
        shared_expert_rank_num = 0
        moe_broken_card_num = 0
        local_moe= 4
        moe_expert_num = local_moe * (ep_world_size * tp_world_size - shared_expert_rank_num)
        is_elastic = 1
        x_ref = torch.ones(1, dtype = torch.int32)
        time_out = torch.tensor([100000], dtype = torch.int32).npu()
        elastic_info_x1 = TestDistributeBarrier.gen_elastic_info(is_elastic, ep_world_size * tp_world_size,
            shared_expert_rank_num, shared_broken_card_num, moe_broken_card_num, local_moe,ep_world_size,
            moe_expert_num)
        
        for _ in range(ep_world_size):
            expt_out_list_1 = self._construct_excepted_result(x_ref, elastic_info_x1, ep_world_size)

        self._test_multiprocess(TestDistributeBarrier._test_npu_distribute_barrier,
                TestDistributeBarrier._init_dist_hccl, [expt_out_list_1, x_ref, time_out, elastic_info_x1,
                                                           ep_world_size, tp_world_size])
if __name__ == '__main__':
    run_tests()

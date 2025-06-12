import os
import math
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestMoeEPLBUpdateExpert(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bs = 128
        self.k = 8
        self.log_ep_size = 256
        self.pyh_ep_size = 8
        self.F = 5
        self.world_size = 8
        self.expert_ids = []
        self.eplb_table = []
        self.balanced_expert_ids = []
        self.gen_exp_result()

    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '50000'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    @classmethod
    def _test_npu_moe_eplb_update_expert(cls, rank_id, input_list):
        expert_ids, eplb_table, world_size, init_pg, c2p, p2c = input_list
        _ = init_pg(rank_id, world_size)
        out = torch_npu.npu_moe_eplb_update_expert(expert_ids=expert_ids.npu(),
                                                   eplb_table=eplb_table.npu(),
                                                   local_rank_id=rank_id,
                                                   world_size=world_size,
                                                   balance_mode=0)
        c2p.put((rank_id, out.cpu()))
        p2c.get()
    
    def gen_exp_result(self):
        for rank_id in range(self.world_size):
            eplb_table = np.zeros((self.log_ep_size, self.F - 1))
            count_cloumn = np.random.randint(1, self.F, size=(self.log_ep_size, 1))
            all_ranks = np.arange(self.pyh_ep_size)
            for i in range(self.log_ep_size):
                np.random.shuffle(all_ranks)
                for j in range(count_cloumn[i][0]):
                    eplb_table[i][j] = all_ranks[j]
            _expert_ids = torch.from_numpy(np.random.randint(low=0, high=self.log_ep_size, size=(self.bs, self.k))).to(torch.int64)
            _eplb_table = torch.from_numpy(np.hstack((count_cloumn, eplb_table))).to(torch.int32)
            self.expert_ids.append(_expert_ids)
            self.eplb_table.append(_eplb_table)
            _balanced_expert_ids = np.zeros((self.bs, self.k))
            for i in range(self.bs):
                for j in range(self.k):
                    log_ep_id = _expert_ids[i][j]
                    mod_val = math.ceil(self.world_size / _eplb_table[log_ep_id][0])
                    phy_ep_id = _eplb_table[log_ep_id][(rank_id // mod_val) + 1]
                    _balanced_expert_ids[i][j] = phy_ep_id
            self.balanced_expert_ids.append(torch.from_numpy(_balanced_expert_ids).to(torch.int64))

    @SupportedDevices(['Ascend910_'])
    def test_npu_moe_eplb_update_expert(self):
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(self.world_size)
        p2c = ctx.Queue(self.world_size)
        ps = []

        for rank_id in range(self.world_size):
            p = ctx.Process(
                target=self._test_npu_moe_eplb_update_expert,
                args=(rank_id, [self.expert_ids[rank_id], self.eplb_table[rank_id], self.world_size, self._init_dist_hccl, c2p, p2c]))
            p.start()
            ps.append(p)

        for _ in range(self.world_size):
            rank_id, output = c2p.get()
            self.assertEqual(output, self.balanced_expert_ids[rank_id],
                             ("rank {} Expect receive tensor {} but got {}.").format(rank_id, self.balanced_expert_ids[rank_id], output))

        for _ in range(self.world_size):
            p2c.put(0)

        for p in ps:
            p.join()


if __name__ == '__main__':
    run_tests()
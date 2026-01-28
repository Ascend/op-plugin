import os
import math
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestMoeUpdateExpert(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bs = 128
        self.k = 8
        self.log_ep_size = 256
        self.pyh_ep_size = 8
        self.F = 5
        self.is_pruning = True
        self.world_size = 8
        self.balance_mode = 0
        self.expert_ids = []
        self.eplb_table = []
        self.expert_scales = []
        self.pruning_threshold = []
        self.active_mask = []
        self.balanced_expert_ids = []
        self.balanced_active_mask = []
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
    def _test_npu_moe_update_expert(cls, rank_id, input_list):
        expert_ids, eplb_table, world_size, expert_scales, pruning_threshold, active_mask, balance_mode, init_pg, c2p, p2c = input_list
        _ = init_pg(rank_id, world_size)
        out_expert_idx, out_mask = torch_npu.npu_moe_update_expert(expert_ids=expert_ids.npu(),
                                                   eplb_table=eplb_table.npu(),
                                                   local_rank_id=rank_id,
                                                   world_size=world_size,
                                                   expert_scales=expert_scales.npu(),
                                                   pruning_threshold=pruning_threshold.npu(),
                                                   active_mask=active_mask.npu(),
                                                   balance_mode=balance_mode)
        c2p.put((rank_id, out_expert_idx.cpu(), out_mask.cpu()))
        p2c.get()
    
    def gen_exp_result(self):
        for rank_id in range(self.world_size):
            eplb_table = np.zeros((self.log_ep_size, self.F - 1))
            count_column = np.random.randint(1, self.F, size=(self.log_ep_size, 1))
            all_ranks = np.arange(self.pyh_ep_size)
            for i in range(self.log_ep_size):
                np.random.shuffle(all_ranks)
                for j in range(count_column[i][0]):
                    eplb_table[i][j] = all_ranks[j]
            eplb_table = np.hstack((count_column, eplb_table))

            expert_ids = np.random.randint(low=0, high=self.log_ep_size, size=(self.bs, self.k))
            if self.is_pruning:
                expert_scales = -np.sort(-np.random.uniform(low=0, high=0.25, size=(self.bs, self.k)), axis=1)
                pruning_threshold = np.random.uniform(low=0, high=0.15, size=(1, self.k))
                num_true = np.random.randint(0, self.bs + 1)
                active_mask = np.concatenate([np.ones(num_true, dtype=bool), np.zeros(self.bs - num_true, dtype=bool)])
            eplb_table_tensor = torch.from_numpy(eplb_table).to(torch.int32)
            self.eplb_table.append(eplb_table_tensor)
            expert_ids_tensor = torch.from_numpy(expert_ids).to(torch.int32)
            self.expert_ids.append(expert_ids_tensor)
            if self.is_pruning:
                expert_scales_tensor = torch.from_numpy(expert_scales).to(torch.float32)
                self.expert_scales.append(expert_scales_tensor)
                pruning_threshold_tensor = torch.from_numpy(pruning_threshold).to(torch.float32)
                self.pruning_threshold.append(pruning_threshold_tensor)
                active_mask_tensor = torch.from_numpy(active_mask).to(torch.bool)
                self.active_mask.append(active_mask_tensor)

            balanced_expert_ids = np.zeros((self.bs, self.k))
            if self.is_pruning:
                balanced_active_mask = np.zeros((self.bs, self.k))

            for i in range(self.bs):
                for j in range(self.k):
                    log_ep_id = expert_ids_tensor[i][j]
                    if self.balance_mode == 0:
                        mod_val = math.ceil(self.world_size / eplb_table_tensor[log_ep_id][0].item())
                        phy_ep_id = eplb_table_tensor[log_ep_id][(rank_id // mod_val) + 1]
                        balanced_expert_ids[i][j] = phy_ep_id
                    if self.balance_mode == 1:
                        phy_ep_id = eplb_table_tensor[log_ep_id][(i % eplb_table_tensor[log_ep_id][0].item()) + 1]
                        balanced_expert_ids[i][j] = phy_ep_id
                    if self.is_pruning:
                        if not active_mask_tensor[i]:
                            balanced_active_mask[i][j] = 0
                        else:
                            if expert_scales_tensor[i][j] < pruning_threshold_tensor[0][j] * sum(expert_scales_tensor[i]):
                                balanced_active_mask[i][j] = 0
                            else:
                                balanced_active_mask[i][j] = 1
            self.balanced_expert_ids.append(torch.from_numpy(balanced_expert_ids).to(torch.int64))
            self.balanced_active_mask.append(torch.from_numpy(balanced_active_mask).to(torch.bool))        


    @SupportedDevices(['Ascend910_93', 'Ascend950'])
    def test_npu_moe_update_expert(self):
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(self.world_size)
        p2c = ctx.Queue(self.world_size)
        ps = []

        for rank_id in range(self.world_size):
            p = ctx.Process(
                target=self._test_npu_moe_update_expert,
                args=(rank_id, [self.expert_ids[rank_id], self.eplb_table[rank_id], self.world_size, 
                                self.expert_scales[rank_id], self.pruning_threshold[rank_id], self.active_mask[rank_id],
                                self.balance_mode, self._init_dist_hccl, c2p, p2c]))
            p.start()
            ps.append(p)

        for _ in range(self.world_size):
            rank_id, output_0, output_1 = c2p.get()
            self.assertEqual(output_0, self.balanced_expert_ids[rank_id],
                             ("rank {} Expect receive tensor {} but got {}.").format(rank_id, self.balanced_expert_ids[rank_id], output_0))
            self.assertEqual(output_1, self.balanced_active_mask[rank_id],
                             ("rank {} Expect receive tensor {} but got {}.").format(rank_id, self.balanced_active_mask[rank_id], output_1))

        for _ in range(self.world_size):
            p2c.put(0)

        for p in ps:
            p.join()


if __name__ == '__main__':
    run_tests()
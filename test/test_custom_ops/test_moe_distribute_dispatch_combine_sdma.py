import os
import unittest
import numpy as np
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU

# 控制模式
quant_mode = 2                       # 2为动态量化
is_dispatch_scales = True            # 动态量化可选择是否传scales
input_dtype = torch.bfloat16         # 输出dtype
sharedExpertRankNum = 0              # 共享专家数
moeExpertNum = 16                    # moe专家数
bs = 8                               # token数量
h = 1024                             # 每个token的长度
k = 8
random_seed = 0
ep_world_size = 16
moe_rank_num = ep_world_size - sharedExpertRankNum
local_moe_expert_num = moeExpertNum // moe_rank_num
globalBS = bs * ep_world_size
is_shared = (sharedExpertRankNum > 0)
is_quant = (quant_mode > 0)


def gen_unique_topk_array(low, high):
    array = []
    for _ in range(bs):
        top_idx = list(np.arange(low, high, dtype=np.int32))
        np.random.shuffle(top_idx)
        array.append(top_idx[0:k])
    return np.array(array)


def test_npu_moe_distribute_dispatch_combine_sdma_(rank, c2p, p2c):
    torch_npu.npu.set_device(rank)
    init_method = 'tcp://' + "127.0.0.1" + ':' + '50001'
    dist.init_process_group(backend='hccl', world_size=ep_world_size, rank=rank, init_method=init_method)
    ep_group = dist.new_group(backend="hccl", ranks=range(ep_world_size))
    ep_hcomm_info = comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)

    # 创建输入tensor
    x = torch.randn(bs, h, dtype=input_dtype).npu()
    expert_ids = gen_unique_topk_array(0, moeExpertNum).astype(np.int32)
    expert_ids = torch.from_numpy(expert_ids).npu()

    expert_scales = torch.randn(bs, k, dtype=torch.float32).npu()
    scales_shape = (1 + moeExpertNum, h) if sharedExpertRankNum == 0 else (moeExpertNum, h)
    if is_dispatch_scales:
        scales = torch.randn(scales_shape, dtype=torch.float32).npu()
    else:
        scales = None

    torch.npu.synchronize()
    y, expand_idx, comm_cmd_info = torch_npu.npu_moe_distribute_dispatch_setup(
        x=x,
        expert_ids=expert_ids,
        group_ep=ep_hcomm_info,
        ep_world_size=ep_world_size,
        ep_rank_id=rank,
        moe_expert_num=moeExpertNum,
        expert_shard_type=0,
        shared_expert_rank_num=sharedExpertRankNum,
        scales=scales,
        quant_mode=quant_mode,
        global_bs=globalBS)
    torch.npu.synchronize()
    expand_x, dynamic_scales, assist_info_for_combine, expert_token_nums = torch_npu.npu_moe_distribute_dispatch_teardown(
        x=x, 
        y=y, 
        expert_ids=expert_ids,
        comm_cmd_info=comm_cmd_info,
        group_ep=ep_hcomm_info,
        ep_world_size=ep_world_size,
        ep_rank_id=rank,
        moe_expert_num=moeExpertNum,
        expert_shard_type=0,
        shared_expert_rank_num=sharedExpertRankNum,
        quant_mode=quant_mode,
        global_bs=globalBS)
    if is_quant:
        expand_x = expand_x.to(input_dtype)
    torch.npu.synchronize()
    quant_expand_x, comm_cmd_info = torch_npu.npu_moe_distribute_combine_setup(
        expand_x=expand_x,  
        expert_ids=expert_ids,
        assist_info_for_combine=assist_info_for_combine,
        group_ep=ep_hcomm_info,
        ep_world_size=ep_world_size,
        ep_rank_id=rank,
        moe_expert_num=moeExpertNum,
        expert_shard_type=0,
        shared_expert_rank_num=sharedExpertRankNum,
        comm_quant_mode=quant_mode,
        global_bs=globalBS)
    torch.npu.synchronize()
    out = torch_npu.npu_moe_distribute_combine_teardown(
        expand_x=expand_x, 
        quant_expand_x=quant_expand_x,
        expert_ids=expert_ids,
        expand_idx=expand_idx, 
        expert_scales=expert_scales,
        comm_cmd_info=comm_cmd_info,
        group_ep=ep_hcomm_info,
        ep_world_size=ep_world_size,
        ep_rank_id=rank,
        moe_expert_num=moeExpertNum,
        expert_shard_type=0,
        shared_expert_rank_num=sharedExpertRankNum,
        comm_quant_mode=quant_mode,
        global_bs=globalBS)
    c2p.put((rank, out.cpu()))
    p2c.get()


class TestMoeDistributeDispatchCombineSdma(TestCase):
    @skipIfUnsupportMultiNPU(16)
    @SupportedDevices(['Ascend910_'])
    def test_npu_moe_distribute_dispatch(self):
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(ep_world_size)
        p2c = ctx.Queue(ep_world_size)
        p_list = []
        for rank in range(ep_world_size):
            p = ctx.Process(target=test_npu_moe_distribute_dispatch_combine_sdma_, args=(rank, c2p, p2c))
            p.start()
            p_list.append(p)
        for _ in range(ep_world_size):
            rank_id, out = c2p.get()
            print("recv rank", rank_id, "data success")
        for _ in range(ep_world_size):
            p2c.put(0)
        for p in p_list:
            p.join()


if __name__ == '__main__':
    run_tests()

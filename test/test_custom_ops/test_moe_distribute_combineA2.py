import os
import unittest
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU

# ====================== 双机16卡线下验证步骤 ======================

# ------------------------ 环境配置部分 --------------------------
# 1. 基础环境准备
#    1.1 加载CANN环境：source {CANN安装路径}/set_env.sh  
#
#    1.2 conda环境：conda环境安装torch，torch_npu及相关依赖numpy等
#
#    1.3 双机环境变量配置（两台机器均需设置）
#        export HCCL_WHITELIST_DISABLE=1
#        export HCCL_IF_IP="本机实际IP"  # 注意：不同机器需分别设置自己的IP
#        export HCCL_LOGIC_SUPERPOD_ID=0
#
# ------------------------ 代码修改部分 --------------------------
# 2. 代码调整部分
#    2.1 装饰器参数修改
#        将 @skipIfUnsupportMultiNPU(16) 修改为 @skipIfUnsupportMultiNPU(8)
#
#    2.2 初始化进程组配置
#        init_process_group() 中填写统一的主节点IP（例如ip_a）
#
#    2.3 Rank列表修改（双机唯一差异点，注意主节点IP的代码中rank_list一定是0-7）
#        ip_a机器：rank_list = list(np.arange(0, 8))    # 0-7号rank
#        ip_b机器：rank_list = list(np.arange(8, 16))   # 8-15号rank
#
# ------------------------ 数据准备部分 --------------------------
# 3. 数据同步方案
#    3.1 生成并保存数据：在主机ip_a中将gen_combine_input函数输出的数据都用torch.save()保存      
#           例如：torch.save(x_world, "x_world.pt")
#
#    3.2 两机数据同步：在ip_b机器执行 scp user@ip_a:/target/file/path/*.pt /current/script/path
#
#    3.3 双机加载数据：双机注释掉相关生成数据部分代码，并用torch.load()函数加载pt文件            
#           例如：x_world = torch.load("x_world.pt")
#
# ------------------------ 执行验证部分 --------------------------
# 4. 双机同时执行脚本即可

class TestMoeDistributeCombine(TestCase):

    @classmethod
    def init_hccl_comm(cls, rank, ep_world_size):
        torch_npu.npu.set_device(f"npu:{rank%8}")
        dist.init_process_group(backend="hccl", rank=rank, world_size=ep_world_size, init_method='tcp://' + "127.0.0.1"+ ':' + "50000")
        ep_ranks_list = list(np.arange(0, ep_world_size))
        ep_group = dist.new_group(backend="hccl", ranks=ep_ranks_list)
        ep_hcomm_info = ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)

        return ep_hcomm_info, ep_group

    @classmethod
    def _test_npu_moe_distribute_combine(cls, c2p, p2c, expand_x, expert_ids, expand_idx,
                    ep_send_counts, tp_send_counts, expert_scales, rank_id, ep_world_size,
                    moe_expert_num, bs, global_bs, init_pg):
        ep_hcomm_info, ep_group = init_pg(rank_id, ep_world_size)

        expand_x = expand_x.npu()
        expert_ids = expert_ids.npu()
        expand_idx = expand_idx.npu()
        ep_send_counts = ep_send_counts.npu()
        tp_send_counts = tp_send_counts.npu()
        expert_scales = expert_scales.npu()
        x = torch_npu.npu_moe_distribute_combine(
                    expand_x=expand_x,
                    expert_ids=expert_ids,
                    expand_idx=expand_idx,
                    ep_send_counts=ep_send_counts,
                    tp_send_counts=tp_send_counts,
                    expert_scales=expert_scales,
                    group_ep=ep_hcomm_info,
                    ep_world_size=ep_world_size,
                    ep_rank_id=rank_id,
                    moe_expert_num=moe_expert_num,
                    global_bs=global_bs)
        c2p.put((rank_id, x.cpu()))
        p2c.get()

    def chunk_tensor(self, tensor, num_chunks):
        chunk_size = tensor.size(0) // num_chunks
        chunks = []
        for i in range(num_chunks):
            chunk = tensor[i * chunk_size : (i + 1) * chunk_size]
            chunks.append(chunk)
        return chunks

    def calc_expand_idx(self, expert_ids):
        original_shape = expert_ids.shape
        flattened = expert_ids.flatten()
        expand_idx = torch.zeros_like(flattened).to(torch.int32)
        # 记录每个元素的当前计数
        count_dict = {}
        for i in range(len(flattened)):
            value = flattened[i].item()
            count_dict[value] = count_dict.get(value, -1) + 1
            expand_idx[i] = count_dict[value]
        return expand_idx.reshape(original_shape)

    def calc_send_counts_world(self, expert_ids_world, moe_expert_num, world_size):
        bs = expert_ids_world.shape[0] // world_size
        send_counts_world = torch.empty((moe_expert_num * world_size), dtype=torch.int32)
        for rank_id in range(world_size):
            expert_ids = expert_ids_world[rank_id * bs: (rank_id + 1) * bs].flatten()
            send_counts_world[rank_id * moe_expert_num: (rank_id + 1) * moe_expert_num] = torch.bincount(expert_ids, minlength=moe_expert_num)
        return send_counts_world.reshape(world_size, moe_expert_num).T.reshape(world_size, moe_expert_num).cumsum(-1, dtype=torch.int32)

    def gen_combine_input(self, bs: int, k: int, h: int, world_size: int, moe_expert_num: int, dtype=torch.float16):
        # 计算dispatch结果作为combine输入
        local_moe_expert_num = moe_expert_num // world_size
        bs = 8
        global_bs = bs * world_size
        A = local_moe_expert_num * global_bs
        x_world = torch.empty((bs * world_size, h), dtype=torch.float16).uniform_(-5, 5)
        expert_ids_world = torch.argsort(torch.rand(bs * world_size, moe_expert_num), dim=1)[:, :k].to(torch.int32)
        expandx_world = torch.zeros((A * world_size, h), dtype=torch.float16)
        expand_idx_world = torch.empty((bs * world_size, k), dtype=torch.int32)
        send_counts_world = self.calc_send_counts_world(expert_ids_world, moe_expert_num, world_size)
        expert_scales_world = torch.empty((bs * world_size, k), dtype=torch.float32).uniform_(-5, 5)
        tp_send_counts_world = torch.zeros_like(send_counts_world)

        for world in range(world_size):
            expert_ids = expert_ids_world[world * bs: (world + 1) * bs]
            expand_idx_world[world * bs: (world + 1) * bs] = self.calc_expand_idx(expert_ids)

        for world in range(world_size):
            x = x_world[world * bs: (world + 1) * bs]
            expert_ids = expert_ids_world[world * bs: (world + 1) * bs]
            expand_idx = expand_idx_world[world * bs: (world + 1) * bs]

            for i in range(bs):
                for j in range(k):
                    expert_id = expert_ids[i][j].item()
                    dst_rank_id = expert_id // local_moe_expert_num
                    expert_id_in_rank = expert_id % local_moe_expert_num
                    if expert_id_in_rank == 0 and world == 0:
                        base_offset = 0
                    else:
                        base_offset = send_counts_world[dst_rank_id][expert_id_in_rank * world_size + world - 1].item()
                    inner_offset = expand_idx[i][j].item()
                    expandx_world[dst_rank_id * A + base_offset + inner_offset] = x[i]
        return x_world, expandx_world, expert_ids_world, expand_idx_world, send_counts_world, tp_send_counts_world, expert_scales_world

    def _test_multiprocess(self, f, init_pg, input_list):
        golden_out_tensors, expandx, expert_ids, expand_idx, \
            ep_send_counts_world, tp_send_counts_world, expert_scales, ep_world_size, moe_expert_num, bs, global_bs = input_list
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(ep_world_size)
        p2c = ctx.Queue(ep_world_size)
        p_list = []
        rank_list = list(np.arange(0, ep_world_size))
        for rank_id in rank_list:
            p = ctx.Process(target=f, args=(c2p, p2c, expandx[rank_id], expert_ids[rank_id], expand_idx[rank_id], 
                        ep_send_counts_world[rank_id], tp_send_counts_world[rank_id], expert_scales[rank_id], rank_id, ep_world_size, moe_expert_num, bs, global_bs, init_pg))
            p.start()
            p_list.append(p)

        for _ in rank_list:
            rank, output = c2p.get()
            tol = 2 ** (-7) if output.dtype == torch.bfloat16 else 2 ** (-8)
            self.assertRtolEqual(output.float(), golden_out_tensors[rank].float(), tol)

        for _ in rank_list:
            p2c.put(0)

        for p in p_list:
            p.join()

    @skipIfUnsupportMultiNPU(16)
    @SupportedDevices(['Ascend910B'])
    def test_npu_moe_distribute_combine(self):
        ep_world_size = 16
        tp_world_size = 0
        world_size = ep_world_size
        bs = 8
        h = 7168
        k = 8
        sharedExpertRankNum = 1
        moe_expert_num = 16
        global_bs = bs * ep_world_size

        x_world, expandx_world, expert_ids_world, expand_idx_world, ep_send_counts_world, tp_send_counts_world, expert_scales_world = self.gen_combine_input(bs, k, h, ep_world_size, moe_expert_num)
        
        expandx = self.chunk_tensor(expandx_world, ep_world_size)
        expert_ids = self.chunk_tensor(expert_ids_world, ep_world_size)
        expand_idx = self.chunk_tensor(expand_idx_world, ep_world_size)
        expert_scales = self.chunk_tensor(expert_scales_world, ep_world_size)

        # golden
        x_world = x_world.reshape((ep_world_size, bs, h)).unsqueeze(-2).to(torch.float32) # (ep_world_size, bs, 1, h)
        expert_scales_world = expert_scales_world.reshape((ep_world_size, bs, k)).unsqueeze(-1).to(torch.float32) # (ep_world_size, bs, k, 1)
        golden_out_tensors = (x_world * expert_scales_world).sum(dim=-2)

        self._test_multiprocess(TestMoeDistributeCombine._test_npu_moe_distribute_combine,
                TestMoeDistributeCombine.init_hccl_comm, [golden_out_tensors, expandx, expert_ids, expand_idx, 
                        ep_send_counts_world, tp_send_counts_world, expert_scales, ep_world_size, moe_expert_num, bs, global_bs])

if __name__ == '__main__':
    run_tests()

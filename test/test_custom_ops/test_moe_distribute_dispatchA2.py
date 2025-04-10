import os
import unittest
import numpy as np
import random
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
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
#    3.1 生成并保存数据：在主机ip_a中将gen_x()，gen_expert_ids，gen_scale()三个函数输出的数据都用torch.save()保存      
#           例如：torch.save(x_input, "x_input.pt")
#
#    3.2 两机数据同步：在ip_b机器执行 scp user@ip_a:/target/file/path/*.pt /current/script/path
#
#    3.3 双机加载数据：双机注释掉相关生成数据部分代码，并用torch.load()函数加载pt文件            
#           例如：x_input = torch.load("x_input.pt")
#
# ------------------------ 执行验证部分 --------------------------
# 4. 双机同时执行脚本即可

class TestMoeDistributeDispatch(TestCase):
    def chunk_tensor(self, tensor, num_chunks):
        chunk_size = tensor.size(0) // num_chunks
        chunks = []
        for i in range(num_chunks):
            chunk = tensor[i * chunk_size : (i + 1) * chunk_size]
            chunks.append(chunk)
        return chunks

    def gen_x(self, shape, dtype):
        tmp = []
        for i in range(0, shape[0]):
            tmp.extend([i + 1] * shape[1]) # 全1行 全2行
        tmp = np.random.uniform(-5, 5, size=shape)
        return torch.tensor(np.array(tmp).astype(np.float32)).to(dtype).view(shape)

    def gen_expert_ids(self, shape, total_expert_num):
        a = list(np.arange(0, total_expert_num).astype(np.int32))
        tmp = []
        for i in range(0, shape[0]):
            ids = random.sample(a, shape[1])
            tmp = np.append(tmp, ids)
        return torch.tensor(tmp).to(torch.int32).view(shape[0], shape[1])

    def gen_scale(self, shape, has_scale):
        if has_scale:
            return torch.tensor(np.random.uniform(1.0, 1.0, size=shape).astype(np.float32)).to(torch.float32)
        else:
            return None

    def gen_dispatch_golden(self, x, expert_ids, scales, has_scale, k, quant_mode, global_bs, ep_world_size, bs, total_expert_num, expert_num_per_rank):
        expand_x = torch.repeat_interleave(x, k, dim=0)
        if has_scale:
            expand_x = expand_x.to(torch.float32)
            scales_gather = torch.gather(scales, 0, expert_ids.view(-1).to(torch.int64)).view(-1, 1)
            expand_x = torch.mul(expand_x, scales_gather)

        dynamic_scales = None
        if quant_mode == 2:
            expand_x = expand_x.to(torch.float32)
            max_value, _ = torch.max(torch.abs(expand_x), dim=1)
            dynamic_scales = (torch.tensor([127.0]).to(torch.float32) / max_value).view(-1, 1).to(torch.float32)
            expand_x = expand_x * dynamic_scales
            expand_x = expand_x.to(torch.int8)
        else:
            expand_x = expand_x.to(torch.bfloat16)

        expert_ids = expert_ids.view(global_bs * k)
        expert_ids_sorted, sorted_idx = torch.sort(expert_ids, stable=True)
        torch.sort(sorted_idx)
        expand_x_sorted = expand_x[sorted_idx]

        dynamic_scales_sorted = None
        if quant_mode == 2:
            dynamic_scales_sorted = dynamic_scales[sorted_idx].view(-1)

        expert_ids_input = self.chunk_tensor(expert_ids, ep_world_size)
        expand_idx = torch.zeros(size=(global_bs, k)).to(torch.int32)
        for rank_id in range(ep_world_size):
            expert_ids_per_rank = expert_ids_input[rank_id].view(-1)
            unique_expert, inverse_indices = torch.unique(expert_ids_per_rank, sorted=True, return_inverse=True)
            valid_expert_token_num_per_rank = torch.bincount(inverse_indices)
            expand_idx_per_rank = torch.zeros(size=(bs, k)).to(torch.int32).view(-1)
            # 遍历每个唯一值，计算其在原张量中的出现顺序
            for i, value in enumerate(unique_expert):
                indices = (expert_ids_per_rank == value).nonzero(as_tuple=True)[0]
                expand_idx_per_rank[indices] = torch.arange(0, valid_expert_token_num_per_rank[i]).to(torch.int32)
                expand_idx[rank_id * bs : (rank_id + 1) * bs, :] = expand_idx_per_rank.view(bs, k)

        vaild_expert_token_nums = torch.bincount(expert_ids).to(torch.int32)
        expert_token_nums = F.pad(vaild_expert_token_nums, (0, total_expert_num - vaild_expert_token_nums.size(0)), 'constant', 0)
        expert_tokens_num_cumsum = []
        for rank_id in range(ep_world_size):
            count = torch.cumsum(expert_token_nums[rank_id * expert_num_per_rank : (rank_id + 1) * expert_num_per_rank], dim=0)
            expert_tokens_num_cumsum.append(count)

        ep_recv_counts = []
        for expert_id in range(total_expert_num):
            for rank_id in range(ep_world_size):
                count = torch.sum(expert_ids_input[rank_id].eq(expert_id)).item()
                ep_recv_counts.append(count)
        ep_recv_counts = torch.tensor(ep_recv_counts).to(torch.int32)
        ep_recv_counts_cumsum = []
        for rank_id in range(ep_world_size):
            count = torch.cumsum(ep_recv_counts[rank_id * expert_num_per_rank * ep_world_size : (rank_id + 1) * expert_num_per_rank * ep_world_size], dim=0)
            ep_recv_counts_cumsum.append(count)

        actual_tokens = []
        count = 0
        for rank_id in range(ep_world_size):
            count = count + torch.sum(expert_token_nums[rank_id * expert_num_per_rank : (rank_id + 1) * expert_num_per_rank]).item()
            actual_tokens.append(count)
        actual_tokens = torch.tensor(actual_tokens).to(torch.int32)

        return [expand_x_sorted, dynamic_scales_sorted, expand_idx, expert_tokens_num_cumsum, ep_recv_counts_cumsum, None], actual_tokens

    def golden_compare(self, rank_id, golden_tensor_list, golden_actual_tokens_cumsum, npu_result, quant_mode, bs, k):
        result = []
        start_offset_in_golden = golden_actual_tokens_cumsum[rank_id - 1].item() if rank_id > 0 else 0
        end_offset_in_golden = golden_actual_tokens_cumsum[rank_id].item()

        expand_x_golden = golden_tensor_list[0][start_offset_in_golden : end_offset_in_golden, :]
        golden_actual_tokens = golden_actual_tokens_cumsum[rank_id] if rank_id == 0 else golden_actual_tokens_cumsum[rank_id] - golden_actual_tokens_cumsum[rank_id - 1]
        expand_x_npu = npu_result[0][0 : golden_actual_tokens.item(), :]
        if quant_mode == 0:
            self.assertEqual(expand_x_golden, expand_x_npu,
                             ("rank {} Expect receive tensor {} but got {}.").format(rank_id,  expand_x_golden, expand_x_npu))
        else:
            self.assertRtolEqual(expand_x_golden, expand_x_npu, atol=1)

        if quant_mode == 2:
            dynamic_scales_golden = golden_tensor_list[1][start_offset_in_golden : end_offset_in_golden]
            dynamic_scales_npu = npu_result[1][0 : golden_actual_tokens.item()]
            self.assertRtolEqual(dynamic_scales_golden, dynamic_scales_npu, 0.001)

        expand_idx_golden = golden_tensor_list[2][bs * rank_id : bs * (rank_id + 1), :]
        expand_idx_npu = npu_result[2].view(bs, k)
        self.assertEqual(expand_idx_golden, expand_idx_npu,
                             ("rank {} Expect receive tensor {} but got {}.").format(rank_id, expand_idx_golden, expand_idx_npu))

        expert_tokens_num_golden = golden_tensor_list[3][rank_id]
        expert_tokens_num_npu = npu_result[3]
        self.assertEqual(expert_tokens_num_golden, expert_tokens_num_npu,
                             ("rank {} Expect receive tensor {} but got {}.").format(rank_id, expert_tokens_num_golden, expert_tokens_num_npu))

        ep_recv_counts_golden = golden_tensor_list[4][rank_id]
        ep_recv_counts_npu = npu_result[4]
        self.assertEqual(ep_recv_counts_golden, ep_recv_counts_npu,
                             ("rank {} Expect receive tensor {} but got {}.").format(rank_id, ep_recv_counts_golden, ep_recv_counts_npu))

    @classmethod
    def run_dispatch_npu(cls, queue, rank, x, expert_ids, scales, ep_world_size, has_scale, total_expert_num, quant_mode, global_bs):
        torch_npu.npu.set_device(rank)

        dist.init_process_group(backend="hccl", rank=rank, world_size=ep_world_size, init_method='tcp://' + "127.0.0.1" + ':' + "50000")
        # 初始化EP域
        ep_ranks_list = list(np.arange(0, ep_world_size))
        ep_group = dist.new_group(backend="hccl", ranks=ep_ranks_list)
        ep_hcomm_info = ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)

        x_npu = x.npu()
        expert_ids_npu = expert_ids.npu()
        scales_npu = scales.npu() if has_scale else None

        expand_x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts, _ = torch_npu.npu_moe_distribute_dispatch(
                x=x_npu,
                expert_ids=expert_ids_npu,
                group_ep=ep_hcomm_info,
                ep_world_size=ep_world_size,
                ep_rank_id=rank,
                moe_expert_num=total_expert_num,
                scales=scales_npu,
                quant_mode=quant_mode,
                global_bs = global_bs
            )
        queue.put((rank, [expand_x.cpu(), dynamic_scales.cpu(), expand_idx.cpu(), expert_token_nums.cpu(), ep_recv_counts.cpu(), None]))

    @skipIfUnsupportMultiNPU(16)
    @SupportedDevices(['Ascend910B'])
    def test_npu_moe_distribute_dispatch(self):
        has_scale = False
        quant_mode = 0
        ep_world_size = 16
        tp_world_size = 0
        world_size = ep_world_size
        bs = 8
        h = 7168
        k = 8
        sharedExpertRankNum = 0
        moeExpertNum = 16
        global_bs = bs * ep_world_size
        expert_num_per_rank = 1
        total_expert_num = world_size * expert_num_per_rank

        input_dtype = torch.bfloat16
        x_shape = (global_bs, h)
        expert_ids_shape = (global_bs, k)
        scales_shape = (total_expert_num, h)

        x = self.gen_x(x_shape, input_dtype)
        expert_ids = self.gen_expert_ids(expert_ids_shape, total_expert_num)
        scales = self.gen_scale(scales_shape, has_scale)

        x_input = self.chunk_tensor(x, ep_world_size)
        expert_ids_input = self.chunk_tensor(expert_ids, ep_world_size)
        scales_input = scales

        golden_tensor_list, golden_actual_tokens = self.gen_dispatch_golden(x, expert_ids, scales, has_scale, k, quant_mode, global_bs, ep_world_size, bs, total_expert_num, expert_num_per_rank)

        p_list = []
        rank_list = list(range(0, ep_world_size))

        from torch.multiprocessing import Manager
        manager = Manager()
        result_queue = manager.Queue()
        mp.set_start_method("forkserver", force=True)
        for rank_id in rank_list:
            p = mp.Process(target=TestMoeDistributeDispatch.run_dispatch_npu, args=(result_queue, rank_id, x_input[rank_id], expert_ids_input[rank_id], scales_input, 
                                                                                    ep_world_size, has_scale, total_expert_num, quant_mode, global_bs))
            p.start()
            p_list.append(p)

        results = {}
        for p in p_list:
            p.join()
            rank_id, rank_result = result_queue.get()
            results[rank_id] = rank_result

        for rank_id in rank_list:
            self.golden_compare(rank_id, golden_tensor_list, golden_actual_tokens, results[rank_id], quant_mode, bs, k)

if __name__ == '__main__':
    run_tests()

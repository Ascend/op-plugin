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


class TestMoeDistributeCombineAddRmsNorm(TestCase):

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
    def _test_npu_moe_distribute_combine_add_rms_norm(cls, rank, input_list):
        expand_x, scales1_list, topk1_list, expand_idx, ep_send_counts, tp_send_counts,\
            ep_world_size, tp_world_size, globalBS, sharedExpertRankNum, moeExpertNum, init_pg, c2p, p2c = input_list
        topk = topk1_list[rank // tp_world_size]
        expert_scales = scales1_list[rank // tp_world_size]
        pg, ep_group, tp_group = init_pg(rank, ep_world_size * tp_world_size, ep_world_size, tp_world_size)
        ep_hcomm_name = ep_group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
        tp_hcomm_name = tp_group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)

        expand_x = expand_x.to(torch.bfloat16).npu()
        topk = topk.npu()
        expand_idx = expand_idx.npu()
        ep_send_counts = ep_send_counts.npu()
        tp_send_counts = tp_send_counts.npu()
        expert_scales = expert_scales.npu()
        residual_x = torch.tensor(np.random.uniform(0, 1, size=[topk.shape[0], 1, expand_x.shape[1]])).to(torch.bfloat16).npu()
        gamma = torch.tensor(np.random.uniform(0, 1, size=[expand_x.shape[1]])).to(torch.bfloat16).npu()
        out = torch_npu.npu_moe_distribute_combine_add_rms_norm(expand_x=expand_x,
                                                   expert_ids=topk,
                                                   expand_idx=expand_idx,
                                                   ep_send_counts=ep_send_counts,
                                                   expert_scales=expert_scales,
                                                   residual_x=residual_x,
                                                   gamma=gamma,
                                                   group_ep=ep_hcomm_name,
                                                   ep_world_size=ep_world_size,
                                                   ep_rank_id=int(rank // tp_world_size),
                                                   moe_expert_num=moeExpertNum,
                                                   tp_send_counts=tp_send_counts,
                                                   group_tp=tp_hcomm_name,
                                                   tp_world_size=tp_world_size,
                                                   tp_rank_id=int(rank % tp_world_size),
                                                   expert_shard_type=0,
                                                   shared_expert_rank_num=sharedExpertRankNum,
                                                   global_bs=globalBS)
        c2p.put((rank, out[0].cpu()))
        p2c.get()

    def _test_multiprocess(self, f, init_pg, input_list):
        expt_out_list, expand_x_list, scales1_list, topk1_list,\
            idx_list, ep_recvCount_list, tp_recvCount_list, ep_world_size, tp_world_size, globalBS,\
            sharedExpertRankNum, moeExpertNum = input_list
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(ep_world_size * tp_world_size)
        p2c = ctx.Queue(ep_world_size * tp_world_size)
        ps = []

        for i in range(ep_world_size * tp_world_size):
            p = ctx.Process(
                target=f,
                args=(i, [expand_x_list[i], scales1_list, topk1_list,
                    idx_list[i], ep_recvCount_list[i], tp_recvCount_list[i], ep_world_size, tp_world_size, globalBS,
                    sharedExpertRankNum, moeExpertNum, init_pg, c2p, p2c]))
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

    def _chunk_tensor(self, tensor, num_chunks):
        chunk_size = tensor.size(0) // num_chunks
        chunks = []
        for i in range(num_chunks):
            chunk = tensor[i * chunk_size:(i + 1) * chunk_size]
            chunks.append(chunk)
        return chunks

    def _construct_idx(self, tensor, ep_world_size):
        num_groups = ep_world_size
        group_size = tensor.size(0) // num_groups
        split_tensors = torch.split(tensor, group_size)
        count_tensor = torch.zeros_like(tensor)
        for i, split_tensor in enumerate(split_tensors):
            start_idx = i * group_size
            end_idx = start_idx + group_size
            count_dict = {}
            for j, num in enumerate(split_tensor):
                num = num.item()
                count_dict[num] = count_dict.get(num, -1) + 1
                count_tensor[start_idx + j] = count_dict[num]
        return count_tensor.to(torch.int32)

    def _gen_recvCount(self, tensor, bs, ep_world_size, moeExpertNum, sharedExpertRankNum):
        segment_length = tensor.numel() // ep_world_size
        result_tensor = torch.zeros(moeExpertNum, ep_world_size, dtype=torch.int32)
        for i in range(ep_world_size):
            start_idx = i * segment_length
            end_idx = start_idx + segment_length if i < ep_world_size - 1 else tensor.numel()
            segment = tensor[start_idx:end_idx]
            counts = torch.bincount(segment, minlength=moeExpertNum)
            result_tensor[:, i] = counts
        shared = torch.zeros(sharedExpertRankNum, ep_world_size, dtype=torch.int32)
        for i in range(sharedExpertRankNum):
            for j in range(ep_world_size):
                if i == j:
                    shared[i][j] = bs
                elif j >= sharedExpertRankNum:
                    shared[i][j] = int(bs // sharedExpertRankNum)
        result_tensor = torch.cat((shared, result_tensor), dim=0)
        return result_tensor.flatten()
    
    class ConstructArgs:
        def __init__(self, x1_list, topk1_list, bs, h, k, globalBS, 
                     sharedExpertRankNum, moeExpertNum, ep_world_size, tp_world_size, scales1):
            self.x1_list = x1_list
            self.topk1_list = topk1_list
            self.bs = bs
            self.h = h
            self.k = k
            self.globalBS = globalBS
            self.sharedExpertRankNum = sharedExpertRankNum
            self.moeExpertNum = moeExpertNum
            self.ep_world_size = ep_world_size
            self.tp_world_size = tp_world_size
            self.scales1 = scales1

    def _construct_excepted_result(self, args):
        col_idx = torch.arange(0, args.globalBS * args.k, dtype=torch.int32)
        row_idx = col_idx.view(args.k, -1).permute(1, 0)
        mapping = dict(zip(map(int, row_idx.flatten()), map(int, col_idx.flatten())))
        row_idx = row_idx.reshape([args.globalBS, args.k]).contiguous()

        x1 = torch.cat(args.x1_list, dim=0).view(-1, args.h)
        topk1 = torch.cat(args.topk1_list, dim=0).view(-1, args.k)

        expandX1, expand_row1, expand_expert1 = torch_npu.npu_moe_init_routing(x1.npu(), row_idx=row_idx.npu(),
                                                                       expert_idx=topk1.npu(),
                                                                       active_num=args.globalBS)

        expandX1 = expandX1.cpu().view(-1, args.h)
        expand_row1 = expand_row1.cpu()
        expand_expert1 = expand_expert1.cpu()
        j = 0
        result_idx = np.zeros(args.globalBS * args.k).astype(int)
        for i in expand_row1:
            result_idx[int(i)] = mapping[j]
            j += 1
        middle_idx1 = torch.tensor(result_idx.astype(np.int32))

        shared_list = []
        shared_tokens = []
        for i in range(args.sharedExpertRankNum):
            tmp_list = []
            shared_tokens.append(args.bs * (int(args.moeExpertNum / args.sharedExpertRankNum) + 1))
            tmp_list.append(x1[(args.bs * i):(args.bs * (i + 1)), :])
            for j in range(int(args.moeExpertNum / args.sharedExpertRankNum)):
                tmp_list.append(x1[(args.bs * (i + (j + 1) * args.sharedExpertRankNum)):(args.bs * (i + (j + 1) * args.sharedExpertRankNum + 1)), :])
            tmp_list = torch.cat(tmp_list, dim=0).to(torch.bfloat16)
            shared_list.append(tmp_list)
        shared_x1 = shared_list
        token1 = torch.cat((torch.tensor(shared_tokens), torch.bincount(expand_expert1)))

        expand_x_list = []
        for i in range(args.sharedExpertRankNum):
            expand_x_list.append(shared_x1[i])
            shared_x1[i] = shared_x1[i] + shared_x1[i]
        sums1 = 0
        local = int(args.moeExpertNum // (args.ep_world_size - args.sharedExpertRankNum))
        A = int(args.globalBS * local)
        for i in range(args.sharedExpertRankNum, args.ep_world_size):
            start1 = sums1
            end1 = sums1 + int(token1[i])
            sums1 = end1
            pad = torch.tensor(np.random.uniform(0, 1, size=[args.tp_world_size * A - int(token1[i]), args.h])).to(torch.bfloat16)
            expand_x_list.append(torch.cat((expandX1[start1:end1, :], pad)))
            expandX1[start1:end1, :] = expandX1[start1:end1, :] + expandX1[start1:end1, :]

        shared_x1 = torch.cat(shared_x1, dim=0).view(-1, args.h)
        combine_x1_shared = []
        for i in range(args.ep_world_size):
            if i < args.sharedExpertRankNum:
                start_idx = i * int(args.globalBS // args.sharedExpertRankNum)
                end_idx = i * int(args.globalBS // args.sharedExpertRankNum) + args.bs
                combine_x1_shared.append(shared_x1[start_idx:end_idx, :])
            else:
                startIdx = int((i % args.sharedExpertRankNum * args.globalBS // args.sharedExpertRankNum) + i // args.sharedExpertRankNum * args.bs)
                endIdx = startIdx + int(args.bs)
                combine_x1_shared.append(expandX1[startIdx:endIdx, :])
        combine_x1_shared = torch.cat(combine_x1_shared, dim=0).flatten()

        result_list = [None] * len(middle_idx1)
        for i, pos in enumerate(middle_idx1):
            result_list[int(pos)] = expandX1[i].to(torch.float32)
        result_list = [t * s for t, s in zip(result_list, args.scales1.flatten())]
        group_sums = []
        for i in range(args.globalBS):
            start_idx = i * args.k
            end_idx = start_idx + args.k
            group_tensors = result_list[start_idx:end_idx]
            group_sum = torch.stack(group_tensors).sum(dim=0)
            group_sums.append(group_sum)
        combine_x1 = torch.cat(group_sums) + combine_x1_shared

        combine_x1 = combine_x1.to(torch.bfloat16).view(-1, args.h)
        out_list = []
        sums = 0
        for _ in range(args.ep_world_size):
            start_idx = sums
            sums = sums + args.bs
            end_idx = sums
            out_list.append(combine_x1[start_idx:end_idx, :])

        topk1_list = torch.cat(args.topk1_list).flatten()
        idx1 = self._construct_idx(topk1_list, args.ep_world_size)
        idx1 = self._chunk_tensor(idx1, args.ep_world_size)
        idx_list = []
        for i in range(args.ep_world_size):
            idx_list.append(idx1[i])

        recvCount1 = self._gen_recvCount(topk1_list, args.bs, args.ep_world_size, args.moeExpertNum, args.sharedExpertRankNum)
        for i in range(args.ep_world_size):
            sums1 = 0
            for j in range(args.ep_world_size):
                sums1 = recvCount1[i * args.ep_world_size + j] + sums1
                recvCount1[i * args.ep_world_size + j] = sums1
        recvCount1 = self._chunk_tensor(recvCount1, args.ep_world_size)
        ep_recvCount_list = []
        for i in range(args.ep_world_size):
            ep_recvCount_list.append(recvCount1[i])

        tp_recvCount_list = []
        for i in range(args.ep_world_size):
            tp_recvCount_list.append(torch.tensor([int(token1[i])]).to(torch.int32))

        return expand_x_list, out_list, idx_list, ep_recvCount_list, tp_recvCount_list

    @skipIfUnsupportMultiNPU(16)
    @SupportedDevices(['Ascend910_'])
    def test_npu_moe_distribute_combine_add_rms_norm(self):
        ep_world_size = 8
        tp_world_size = 1
        bs = 8
        h = 7168
        k = 7
        sharedExpertRankNum = 1
        moeExpertNum = 7
        globalBS = bs * ep_world_size
        dtype = np.float16
        data_format = -1
        topk = torch.tile(torch.arange(k), (bs,)).int().view(-1, k)
        topk1_list = []
        x1_shape = [dtype, data_format, [bs, h]]
        x1_list = []
        scales1_shape = [np.float32, data_format, [bs, k]]
        scales1_list = []
        for _ in range(ep_world_size):
            x1, _ = create_common_tensor(x1_shape, -1, 1)
            x1_list.append(x1)
            topk1_list.append(topk)
            scales1, _ = create_common_tensor(scales1_shape, -1, 1)
            scales1_list.append(scales1)
        construcat_args = self.ConstructArgs(x1_list, topk1_list, bs, h, k,
                                             globalBS, sharedExpertRankNum, moeExpertNum, ep_world_size, tp_world_size,
                                             torch.cat(scales1_list))
        expand_x_list, expt_out_list, idx_list, ep_recvCount_list, tp_recvCount_list = self._construct_excepted_result(construcat_args)
        self._test_multiprocess(TestMoeDistributeCombineAddRmsNorm._test_npu_moe_distribute_combine_add_rms_norm,
                TestMoeDistributeCombineAddRmsNorm._init_dist_hccl, [expt_out_list, expand_x_list, scales1_list, topk1_list,
                idx_list, ep_recvCount_list, tp_recvCount_list, ep_world_size, tp_world_size, globalBS,
                sharedExpertRankNum, moeExpertNum])


if __name__ == '__main__':
    run_tests()

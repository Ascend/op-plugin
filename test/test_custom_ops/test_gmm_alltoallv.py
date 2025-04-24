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

class TestGmmAlltoAllv(TestCase):
    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '50000'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist
    
    @classmethod
    def _test_npu_gmm_alltoallv(cls, rank, dtype, c2p, init_pg, input_list1, input_list2, expertTokenNum):
        bsk, gmmWeight, mmX, mmWeight, is_trans_gmm_weight, is_trans_mm_weight = input_list1
        epWorldSize, e_epWorldSize, mc2_send_counts, mc2_recv_counts, balance = input_list2
        e = e_epWorldSize // epWorldSize
        pg = init_pg(rank, epWorldSize)
        group = pg.distributed_c10d._get_default_group()
        if torch.__version__ >= '2.0':
            hcom_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
        else:
            hcom_name = group.get_hccl_comm_name(rank)
        gmm_x_shape = (mc2_recv_counts[rank].sum(), gmmX.size(1))
        gmmX = torch.rand(gmm_x_shape).to(dtype)
        goldenOut = TestGmmAlltoAllv._construct_golden_output(rank, dtype, input_list1, input_list2, expertTokenNum)
        send_counts = mc2_recv_counts[rank].reshape(epWorldSize, e).transpose().flatten()
        recv_counts = mc2_send_counts[rank]
        gmmX = gmmX.npu()
        gmmWeight = gmmWeight.npu()
        if mmX is not None:
            mmX = mmX.npu()
        if mmWeight is not None:
            mmWeight = mmWeight.npu()
        gmmYOut, mmYOut = torch_npu.npu_alltoallv_gmm(gmm_x=gmmX,
                                                                gmm_weight=gmmWeight,
                                                                send_counts_tensor=None,
                                                                recv_counts_tensor=None,
                                                                mm_x=mmX,
                                                                mm_weight=mmWeight,
                                                                hcom=hcom_name,
                                                                ep_world_size=epWorldSize,
                                                                send_counts=send_counts,
                                                                recv_counts=recv_counts,
                                                                trans_gmm_weight=is_trans_gmm_weight,
                                                                trans_mm_weight=is_trans_mm_weight)
        if mmYOut is not None:
            mmYOut = mmYOut.cpu()
        gmmYGolden, mmYGolden = goldenOut
        c2p.put((rank, gmmYOut.cpu(), mmYOut))
        if (mmYGolden is not None) or (mmYOut is not None):
            for golden_i, out_i in zip(mmYGolden, mmYOut):
                assert torch.allclose(golden_i, out_i, rtol=0.005, atol=0.005)
        pg.barrier()


    @classmethod
    def _construct_golden_output(cls, rank, dtype, input_list1, input_list2, expertTokenNum):
        bsk, gmmWeight, mmX, mmWeight, is_trans_gmm_weight, is_trans_mm_weight = input_list1
        epWorldSize, e_epWorldSize, mc2_send_counts, mc2_recv_counts, balance = input_list2
        e = e_epWorldSize // epWorldSize
        hccl_send_counts = torch.tensor(np.sum(mc2_recv_counts[rank].reshape(-1, e), axis=1).reshape(epWorldSize)).npu().to(torch.int64).to(torch.device('cpu')).numpy()
        hccl_recv_counts = torch.tensor(np.sum(mc2_send_counts[rank].reshape(-1, e), axis=1).reshape(epWorldSize)).npu().to(torch.int64).to(torch.device('cpu')).numpy()
        gmmX = gmmX.npu()
        gmmWeight = gmmWeight.npu()
        if is_trans_gmm_weight:
            gmmWeight = gmmWeight.permute(0, 2, 1)
        num_tokens_per_local_expert = torch.tenser(np.sum(mc2_recv_counts[rank].reshape(-1, e), axis=0).reshape(e)).npu().to(torch.int64)
        alltoAllvGolden = torch_npu.npu_grouped_matmul(
            x=[gmmX],
            weight=[gmmWeight],
            group_list=num_tokens_per_local_expert,
            group_list_type=1,
            split_item=3
        )
        unpermuteGolden = TestGmmAlltoAllv.unpermute_npu(alltoAllvGolden[0], e, epWorldSize, expertTokenNum, rank)
        gmmYGolden = torch.empty((bsk, gmmWeight.size(2)), dtype=dtype).npu()
        dist.all_to_all_single(gmmYGolden, unpermuteGolden, hccl_recv_counts, hccl_send_counts)
        mmGolden = None
        if (mmX is not None) and (mmWeight is not None):
            mmGolden = torch.matmul(mmX.npu(), mmWeight.transpose(1, 0).npu()) if is_trans_mm_weight else torch.matmul(mmX.npu(), mmWeight.npu())
        if mmGolden is not None:
            mmGolden = mmGolden.cpu()
        return gmmYGolden[0].cpu(), mmGolden

    @classmethod
    def unpermute_npu(cls, tokens, exp_per_card, epWorldSize, expertTokenNum, rank):
        recv = torch.zeros(epWorldSize, exp_per_card).to(torch.int64)
        for i in range(epWorldSize):
            tmp1 = expertTokenNum[i][rank * exp_per_card : (rank + 1) * exp_per_card]
            recv[i:] = torch.tensor(tmp1)
        tmp1 = recv.t()
        sum_list = torch.cumsum(tmp1, dim=1)
        indices_list = []
        for i in range(exp_per_card):
            tmp = []
            for j in range(epWorldSize):
                if j == 0:
                    tmp.append(list(range(0, sum_list[i][j])))
                else:
                    tmp.append(list(range(sum_list[i][j - 1], sum_list[i][j])))
            indices_list.append(tmp)
        selected = []
        for i in range(epWorldSize):
            for j in range(exp_per_card):
                indices = torch.tensor(indices_list[j][i], dtype=torch.long)
                selected_rows = tokens.index_select(dim=0, index=indices.npu())
                selected.append(selected_rows)
        return torch.cat(selected, dim=0).to(tokens.dtype)
    
    def _test_multiprocess(self, f, init_pg, input_list1, input_list2, dtype, expertTokenNum):
        ctx = mp.get_context("spawn")
        bsk, gmmWeight, mmX, mmWeight, is_trans_gmm_weight, is_trans_mm_weight = input_list1
        epWorldSize, e_epWorldSize, mc2_send_counts, mc2_recv_counts, balance = input_list2
        c2p = ctx.Queue(epWorldSize)
        ps = []
        for i in range(epWorldSize):
            p = ctx.Process(
                target=f,
                args=(i, dtype, c2p, init_pg, input_list1, input_list2, expertTokenNum)
            )
            p.start()
            ps.append(p)
        for _ in range(epWorldSize):
            c2p.get()
        for p in ps:
            p.join()
    
    def generate_matrix(self, e, ep_world_size, bsk, balance=True, name="alltoallv_gmm", max_iter=10000):
        if name is not None:
            import hashlib
            hash_bytes = hashlib.sha256(name.encode()).digest()
            seed = int.from_bytes(hash_bytes[:4], 'big')
            np.random.seed(seed)
        row_size = ep_world_size
        col_size = e * ep_world_size
        matrix = []
        if balance:
            avg = bsk // col_size
            tail_num = bsk % col_size
            matrix = np.full((row_size, col_size), avg)
            matrix[:, -1] += tail_num
        else:
            part_col_size = ep_world_size
            part_sum = bsk // e
            tail_sum = bsk % e
            matrix = np.hstack([np.random.multinominal(part_sum - part_col_size, [1 / part_col_size] * part_col_size, size=row_size) + 1 for _ in range(e)])
            matrix[:, -1] += tail_sum
        return matrix
    
    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend910_'])
    def test_npu_alltoallv_gmm(self):
        dtype = torch.float16
        is_balance = True
        e = 32
        BS = 4096
        K = 8
        H1 = 7168
        N1 = 4096
        H2 = 0
        N2 = 0
        is_trans_gmm_weight = False
        is_trans_mm_weight = False
        bsk = BS * K
        epWorldSize = 8
        e_epWorldSize = e * epWorldSize
        expertTokenNum = torch.tensor(self.generate_matrix(e, epWorldSize, bsk, balance=is_balance))
        mc2_send_counts = self.generate_matrix(e, epWorldSize, bsk, balance=is_balance)
        mc2_recv_counts = np.hstack(np.split(mc2_send_counts.reshape(-1, e), epWorldSize, axis=0))
        gmm_weight_shape = (e, H1, N1)
        gmmWeight = torch.rand(gmm_weight_shape).to(dtype)
        if is_trans_gmm_weight:
            gmmWeight = gmmWeight.transpose(0, 1, 2)
        mm_x_shape = (BS, H2)
        mm_weight_shape = (H2, N2)
        mmX = torch.rand(mm_x_shape)
        mmWeight = torch.rand(mm_weight_shape)
        if is_trans_mm_weight:
            mmWeight = mmWeight.transpose(1, 0)
        if (H2 > 0) and (N2 > 0):
            mmX = mmX.to(dtype)
            mmWeight = mmWeight.to(dtype)
        else:
            mmX = None
            mmWeight = None
        self._test_multiprocess(
            TestGmmAlltoAllv._test_npu_gmm_alltoallv,
            TestGmmAlltoAllv._init_dist_hccl,
            [bsk, gmmWeight, mmX, mmWeight, is_trans_gmm_weight, is_trans_mm_weight],
            [epWorldSize, e_epWorldSize, mc2_send_counts, mc2_recv_counts, is_balance],
            dtype, expertTokenNum)

if __name__ == "__main__":
    run_tests()
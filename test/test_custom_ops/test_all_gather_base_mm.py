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



class TestAllGatherBaseMm(TestCase):

    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '50000'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    @classmethod
    def _test_npu_all_gather_base_mm(cls, rank, input_list):
        x1_list, x2_list, world_size, init_pg, c2p = input_list
        x1 = x1_list[rank]
        x2 = x2_list[rank]
        pg = init_pg(rank, world_size)
        group = pg.distributed_c10d._get_default_group()
        if torch.__version__ > '2.0':
            hcom_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
        else:
            hcom_name = group.get_hccl_comm_name(rank)

        x1 = x1.npu()
        x2 = x2.npu()
        out, gather_out = torch_npu.npu_all_gather_base_mm(x1,
                                                           x2,
                                                           hcom_name,
                                                           world_size,
                                                           bias=None,
                                                           gather_index=0,
                                                           gather_output=True,
                                                           comm_turn=0)

        c2p.put((rank, out.cpu(), gather_out.cpu()))
        pg.barrier()

    def _test_multiprocess(self, f, init_pg, input_list):
        expt_out_list, expt_gather, x1, x2, world_size = input_list
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)
        ps = []

        for i in range(world_size):
            p = ctx.Process(
                target=f,
                args=(i, [x1, x2, world_size, init_pg, c2p]))
            p.start()
            ps.append(p)

        for _ in range(world_size):
            rank, output, gather_output = c2p.get()
            self.assertEqual(output, expt_out_list[rank],
                             ("rank {} Expect receive tensor {} but got {}.").format(rank, expt_out_list[rank], output))
            self.assertEqual(gather_output, expt_gather,
                             ("rank {} Expect receive tensor {} but got {}.").format(rank, expt_gather, gather_output))

        for p in ps:
            p.join()

    def _construct_excepted_result(self, x1_list, x2_list, world_size):
        gather_out = torch.cat(x1_list)
        out_list = []
        out_dtype = gather_out.dtype
        for i in range(world_size):
            out_list.append(torch.matmul(gather_out.npu(), x2_list[i].npu()).to(out_dtype).cpu())
        return out_list, gather_out

    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend910B'])
    def test_npu_all_gather_base_mm(self):
        world_size = 8
        dtype = np.float16
        data_format = -1
        x1_shape = [dtype, data_format, [16, 512]]
        x2_shape = [dtype, data_format, [512, 256]]
        x1_list = []
        x2_list = []
        for _ in range(world_size):
            x1, _ = create_common_tensor(x1_shape, -1, 1)
            x2, _ = create_common_tensor(x2_shape, -1, 1)
            x1_list.append(x1)
            x2_list.append(x2)
        expt_out_list, expt_gather = self._construct_excepted_result(x1_list, x2_list, world_size)
        self._test_multiprocess(TestAllGatherBaseMm._test_npu_all_gather_base_mm,
                                TestAllGatherBaseMm._init_dist_hccl, [expt_out_list, expt_gather, x1_list, x2_list, world_size])


if __name__ == '__main__':
    run_tests()

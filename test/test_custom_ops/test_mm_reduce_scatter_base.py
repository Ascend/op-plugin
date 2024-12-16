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


class TestMmReduceScatterBase(TestCase):

    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '50000'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    @classmethod
    def _test_npu_mm_reduce_scatter_base(cls, rank, input_list):
        x1, x2, world_size, init_pg, c2p = input_list
        pg = init_pg(rank, world_size)
        group = pg.distributed_c10d._get_default_group()
        if torch.__version__ > '2.0':
            hcom_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
        else:
            hcom_name = group.get_hccl_comm_name(rank)

        x1 = x1.npu()
        x2 = x2.npu()
        out = torch_npu.npu_mm_reduce_scatter_base(x1,
                                               x2,
                                               hcom_name,
                                               world_size,
                                               reduce_op='sum',
                                               bias=None,
                                               comm_turn=0)

        c2p.put((rank, out.cpu()))
        pg.barrier()

    def _test_multiprocess(self, f, init_pg, input_list):
        expt_out_list, x1_list, x2_list, world_size = input_list
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)
        ps = []

        for i in range(world_size):
            p = ctx.Process(
                target=f,
                args=(i, [x1_list[i], x2_list[i], world_size, init_pg, c2p]))
            p.start()
            ps.append(p)

        for _ in range(world_size):
            rank, output = c2p.get()
            self.assertRtolEqual(output, expt_out_list[rank], 0.05, 0.05)

        for p in ps:
            p.join()

    def _construct_excepted_result(self, x1_list, x2_list, world_size):
        out = None
        for i in range(world_size):
            x1 = x1_list[i]
            x2 = x2_list[i]
            out_single = torch.matmul(x1.to(torch.float), x2.to(torch.float))
            if out is None:
                out = out_single
            else:
                out = torch.add(out, out_single)
        out = out.to(x1_list[0].dtype)
        index = x1_list[0].shape[0] // world_size
        return [out[index * i:index * (i + 1), :] for i in range(world_size)]

    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend910B'])
    def test_npu_mm_reduce_scatter_base(self):
        world_size = 8
        dtype = np.float16
        data_format = -1
        x1_shape = [dtype, data_format, [128, 512]]
        x2_shape = [dtype, data_format, [512, 256]]
        x1_list = []
        x2_list = []
        for _ in range(world_size):
            x1, _ = create_common_tensor(x1_shape, -1, 1)
            x2, _ = create_common_tensor(x2_shape, -1, 1)
            x1_list.append(x1)
            x2_list.append(x2)
        expt_out_list = self._construct_excepted_result(x1_list, x2_list, world_size)
        self._test_multiprocess(TestMmReduceScatterBase._test_npu_mm_reduce_scatter_base,
                                TestMmReduceScatterBase._init_dist_hccl, [expt_out_list, x1_list, x2_list, world_size])


if __name__ == '__main__':
    run_tests()

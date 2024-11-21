import os
import unittest

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from torch.distributed.distributed_c10d import ReduceOp
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU



class TestMmAllReduceBase(TestCase):

    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '50000'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    @classmethod
    def _test_npu_mm_all_reduce_base(cls, rank, input_list):
        x1, x2, scale, world_size, init_pg, c2p = input_list
        pg = init_pg(rank, world_size)
        group = pg.distributed_c10d._get_default_group()
        if torch.__version__ > '2.0.1':
            hcom_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
        else:
            hcom_name = group.get_hccl_comm_name(rank)

        x1 = x1.npu()
        x2 = x2.npu()
        scale = scale.npu()
        out = torch_npu.npu_mm_all_reduce_base(x1, x2, hcom_name, reduce_op='sum', dequant_scale=scale,
                                               bias=None, comm_turn=0)

        c2p.put((rank, out.cpu()))
        pg.barrier()

    def _test_multiprocess(self, f, init_pg, input_list):
        expt_out_list, x1_list, x2_list, scale, world_size = input_list
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)
        ps = []

        for i in range(world_size):
            p = ctx.Process(
                target=f,
                args=(i, [x1_list[i], x2_list[i], scale, world_size, init_pg, c2p]))
            p.start()
            ps.append(p)

        for _ in range(world_size):
            rank, output = c2p.get()
            self.assertEqual(output, expt_out_list[rank],
                             ("rank {} Expect receive tensor {} but got {}.").format(rank, expt_out_list, output))

        for p in ps:
            p.join()

    def _construct_excepted_result(self, x1_list, x2_list, scale, world_size):
        out = None
        out_list = []
        for i in range(world_size):
            x1 = x1_list[i]
            x2 = x2_list[i]
            out_mm = torch.matmul(x1, x2)
            out_single = torch.mul(out_mm, scale)
            if out is None:
                out = out_single
            else:
                out = torch.add(out, out_single)
        for i in range(world_size):
            out_list.append(out.to(torch.bfloat16))
        return out_list

    @skipIfUnsupportMultiNPU(2)
    @SupportedDevices(['Ascend910B'])
    def test_npu_mm_all_reduce_base(self):
        world_size = 2
        dtype = torch.float16
        dtype_quant = np.int8
        dtype_dequant = np.float16
        data_format = -1
        x1_shape = [dtype_quant, data_format, [1, 256]]
        x2_shape = [dtype_quant, data_format, [256, 256]]
        scale_shape = [dtype_dequant, data_format, [256]]
        x1_list = []
        x2_list = []
        scale, _ = create_common_tensor(scale_shape, 1, 100) # 量化场景scale的取值范围
        scale = scale.to(torch.bfloat16) 
        for _ in range(world_size):
            x1, _ = create_common_tensor(x1_shape, -1, 1)
            x2, _ = create_common_tensor(x2_shape, -1, 1)
            x1_list.append(x1)
            x2_list.append(x2)
        expt_out_list = self._construct_excepted_result(x1_list, x2_list, scale, world_size)
        self._test_multiprocess(TestMmAllReduceBase._test_npu_mm_all_reduce_base,
                                TestMmAllReduceBase._init_dist_hccl,
                                [expt_out_list, x1_list, x2_list, scale, world_size])


if __name__ == '__main__':
    run_tests()

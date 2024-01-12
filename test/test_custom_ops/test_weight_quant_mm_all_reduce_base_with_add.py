import os
import unittest

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


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
        x1, x2, scale, offset, x3, world_size, init_pg, c2p = input_list
        pg = init_pg(rank, world_size)
        group = pg.distributed_c10d._get_default_group()
        if torch.__version__ > '2.0.1':
            hcom_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
        else:
            hcom_name = group.get_hccl_comm_name(rank)

        x1 = x1.npu()
        x2 = x2.npu()
        scale = scale.npu()
        offset = offset.npu()
        x3 = x3.npu()
        out = torch_npu.npu_mm_all_reduce_base(x1, x2, hcom_name, reduce_op='sum', antiquant_scale=scale,
                                               antiquant_offset=offset, x3=x3, bias=None, comm_turn=0)

        c2p.put((rank, out.cpu()))
        pg.barrier()

    def _test_multiprocess(self, f, init_pg, input_list):
        expt_out_list, x1_list, x2_list, scale, offset, x3, world_size = input_list
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)
        ps = []

        for i in range(world_size):
            p = ctx.Process(
                target=f,
                args=(i, [x1_list[i], x2_list[i], scale, offset, x3, world_size, init_pg, c2p]))
            p.start()
            ps.append(p)

        for _ in range(world_size):
            rank, output = c2p.get()
            self.assertEqual(output, expt_out_list[rank],
                             ("rank {} Expect receive tensor {} but got {}.").format(rank, expt_out_list, output))

        for p in ps:
            p.join()

    def _construct_excepted_result(self, x1_list, x2_list, scale, offset, x3, world_size):
        out = None
        out_list = []
        out_dtype = np.float16
        for i in range(world_size):
            x1 = x1_list[i]
            x2 = x2_list[i]
            weight = torch.add(x2, offset)
            dequant = torch.mul(weight, scale)
            mm_result = torch.matmul(x1, dequant)
            out_single = torch.add(mm_result, x3)
            if out is None:
                out = out_single
            else:
                out = torch.add(out, out_single)
        for i in range(world_size):
            out_list.append(out.to(x1_list[0].dtype))
        return out_list

    @skipIfUnsupportMultiNPU(8)
    @unittest.skipIf(DEVICE_NAME != 'Ascend910B',
        "OP `MatmulAllReduce` is only supported on 910B, skip this ut for this device type!")
    def test_npu_mm_all_reduce_base(self):
        world_size = 8
        dtype = np.float16
        dtype_quant = np.int8
        data_format = -1
        x1_shape = [dtype, data_format, [1, 256]]
        x2_shape = [dtype_quant, data_format, [256, 256]]
        scale_shape = [dtype, data_format, [256]]
        offset_shape = [dtype, data_format, [256]]
        add_shape = [dtype, data_format, [1, 256]]
        x1_list = []
        x2_list = []
        scale, _ = create_common_tensor(scale_shape, 0.0010, 0.0110) # 量化场景scale的取值范围
        offset, _ = create_common_tensor(offset_shape, -1, 1)
        add, _ = create_common_tensor(add_shape, -1, 1)
        for _ in range(world_size):
            x1, _ = create_common_tensor(x1_shape, -1, 1)
            x2, _ = create_common_tensor(x2_shape, -1, 1)
            x1_list.append(x1)
            x2_list.append(x2)
        expt_out_list = self._construct_excepted_result(x1_list, x2_list, scale, offset, add, world_size)
        self._test_multiprocess(TestMmAllReduceBase._test_npu_mm_all_reduce_base,
                                TestMmAllReduceBase._init_dist_hccl,
                                [expt_out_list, x1_list, x2_list, scale, offset, add, world_size])


if __name__ == '__main__':
    run_tests()

# Copyright (c) 2025, Huawei Technologies.All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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


class TestQuantReduceScatter(TestCase):
    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '50000'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    @classmethod
    def _test_npu_quant_reduce_scatter(cls, rank, input_list):
        x, scales, world_size, init_pg, c2p = input_list
        pg = init_pg(rank, world_size)
        group = pg.distributed_c10d._get_default_group()
        if torch.__version__ > '2.0.1':
            hcom_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
        else:
            hcom_name = group.get_hccl_comm_name(rank)

        x = x.npu()
        scales = scales.npu()
        out_put = torch_npu.npu_quant_reduce_scatter(x, scales, hcom_name, world_size, reduce_op='sum',
                                                     output_dtype=0, x_dtype=None, scales_dtype=None)

        c2p.put((rank, out_put.cpu()))
        pg.barrier()

    def _test_multiprocess(self, f, init_pg, input_list):
        expt_out_list, x_list, scales_list, world_size = input_list
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)
        ps = []

        for i in range(world_size):
            p = ctx.Process(
                target=f,
                args=(i, [x_list[i], scales_list[i], world_size, init_pg, c2p]))
            p.start()
            ps.append(p)

        for _ in range(world_size):
            rank, out_put = c2p.get()
            self.assertEqual(out_put, expt_out_list[rank],
                             ("rank {} Expect receive tensor {} but got {}.").format(rank, expt_out_list, out_put))

        for p in ps:
            p.join()

    def _construct_excepted_result(self, x_list, scales_list, world_size):
        out = None
        for i in range(world_size):
            x = x_list[i]
            scales = scales_list[i]
            out_single = torch.mul(x.to(torch.int8), scales.to(torch.float32))
            if out is None:
                out = out_single
            else:
                out = torch.add(out, out_single)
        out = out.to(torch.float32)
        index = x_list[0].shape[0] // world_size
        return [out[index * i:index * (i + 1), :] for i in range(world_size)]

    @skipIfUnsupportMultiNPU(2)
    @SupportedDevices(['Ascend910_95'])
    def test_npu_quant_reduce_scatter(self):
        world_size = 2
        x_dtype = np.int8
        scales_dtype = np.float32
        data_format = -1
        # T-G量化
        x_shape = [x_dtype, data_format, [2048, 5120]]
        scales_shape = [scales_dtype, data_format, [2048, 40]]
        x_list = []
        scales_list = []
        for _ in range(world_size):
            x, _ = create_common_tensor(x_shape, -1, 1)
            x_list.append(x)
            scales, _ = create_common_tensor(scales_shape, 1, 100)
            scales_list.append(scales)
        expt_out_list = self._construct_excepted_result(x_list, scales_list, world_size)
        self._test_multiprocess(TestQuantReduceScatter._test_npu_quant_reduce_scatter,
                                TestQuantReduceScatter._init_dist_hccl,
                                [expt_out_list, x_list, scales_list, world_size])

if __name__ == '__main__':
    run_tests()

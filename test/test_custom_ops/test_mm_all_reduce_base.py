"""
Test cases for torch_npu.npu_mm_all_reduce_base API

用例说明（共20个用例）：

非量化场景（x1和x2类型一致）：
  - 用例1: x1=float16, x2=float16, 2维输入
  - 用例2: x1=bfloat16, x2=bfloat16, 2维输入
  - 用例3: x1=float16, x2=float16, 3维输入
  - 用例4: x1=bfloat16, x2=bfloat16, 3维输入

伪量化场景（x1=float16/bfloat16, x2=int8, 需要antiquant_scale/offset）：
  - 用例5: x1=float16, x2=int8, perchannel伪量化
  - 用例6: x1=bfloat16, x2=int8, perchannel伪量化
  - 用例7: x1=float16, x2=int8, pertensor伪量化
  - 用例8: x1=bfloat16, x2=int8, pertensor伪量化

全量化场景（x1=int8, x2=int8, 需要dequant_scale）：
  - 用例9: x1=int8, x2=int8, dequant_scale=int64
  - 用例10: x1=int8, x2=int8, dequant_scale=bfloat16
  - 用例11: x1=int8, x2=int8, dequant_scale=float32, pertoken_scale=float32
  - 用例12: x1=int8, x2=int8, dequant_scale=bfloat16, pertoken_scale=float32

扩展场景（使用x1_dtype/x2_dtype参数支持float8/hifloat8）：
  - 用例13: x1=float8_e4m3fn, x2=float8_e4m3fn
  - 用例14: x1=float8_e5m2, x2=float8_e5m2
  - 用例15: x1=hifloat8, x2=hifloat8
  - 用例16: x1=float8_e4m3fn, x2=float8_e5m2
  - 用例17: x1=float8_e5m2, x2=float8_e4m3fn
  - 用例18: x1=hifloat8, x2=float8_e4m3fn
  - 用例19: x1=float8_e4m3fn, x2=hifloat8
  - 用例20: x1=hifloat8, x2=float8_e5m2

约束说明：
  - x1支持2维或3维，x2必须是2维
  - 非量化场景：x1和x2数据类型需一致
  - 伪量化场景：x1为float16/bfloat16，x2为int8
  - 全量化场景：x1和x2都为int8
  - 支持1、2、4、8卡，仅支持hccs链路all mesh组网
"""
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
        x1, x2, world_size, init_pg, c2p = input_list
        pg = init_pg(rank, world_size)
        group = pg.distributed_c10d._get_default_group()
        if torch.__version__ > '2.0.1':
            hcom_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
        else:
            hcom_name = group.get_hccl_comm_name(rank)

        x1 = x1.npu()
        x2 = x2.npu()
        out = torch_npu.npu_mm_all_reduce_base(x1, x2, hcom_name, reduce_op='sum', bias=None, comm_turn=0)

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

        for p in ps:
            p.join()

    def _construct_excepted_result(self, x1_list, x2_list, world_size):
        out = None
        out_list = []
        for i in range(world_size):
            x1 = x1_list[i]
            x2 = x2_list[i]
            out_single = torch.matmul(x1.to(torch.float), x2.to(torch.float))
            if out is None:
                out = out_single
            else:
                out = torch.add(out, out_single)
        for i in range(world_size):
            out_list.append(out.to(x1_list[0].dtype))
        return out_list

    def _construct_excepted_result_3d(self, x1_list, x2_list, world_size):
        out = None
        out_list = []
        for i in range(world_size):
            x1 = x1_list[i]
            x2 = x2_list[i]
            b, s, k = x1.shape
            x1_2d = x1.reshape(b * s, k)
            out_single = torch.matmul(x1_2d.to(torch.float), x2.to(torch.float))
            out_single = out_single.reshape(b, s, -1)
            if out is None:
                out = out_single
            else:
                out = torch.add(out, out_single)
        for i in range(world_size):
            out_list.append(out.to(x1_list[0].dtype))
        return out_list

    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend910B', 'Ascend950'])
    def test_npu_mm_all_reduce_base_fp16_fp16_2d(self):
        """
        用例1: 非量化场景 - x1=float16, x2=float16, 2维输入
        """
        world_size = 8
        m, k, n = 128, 512, 256
        x1_list = []
        x2_list = []
        for _ in range(world_size):
            x1 = torch.randn(m, k, dtype=torch.float16).uniform_(-1, 1)
            x2 = torch.randn(k, n, dtype=torch.float16).uniform_(-1, 1)
            x1_list.append(x1)
            x2_list.append(x2)
        expt_out_list = self._construct_excepted_result(x1_list, x2_list, world_size)
        self._test_multiprocess(TestMmAllReduceBase._test_npu_mm_all_reduce_base,
                                TestMmAllReduceBase._init_dist_hccl, [expt_out_list, x1_list, x2_list, world_size])

    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend910B', 'Ascend950'])
    def test_npu_mm_all_reduce_base_bf16_bf16_2d(self):
        """
        用例2: 非量化场景 - x1=bfloat16, x2=bfloat16, 2维输入
        """
        world_size = 8
        m, k, n = 128, 512, 256
        x1_list = []
        x2_list = []
        for _ in range(world_size):
            x1 = torch.randn(m, k, dtype=torch.bfloat16).uniform_(-1, 1)
            x2 = torch.randn(k, n, dtype=torch.bfloat16).uniform_(-1, 1)
            x1_list.append(x1)
            x2_list.append(x2)
        expt_out_list = self._construct_excepted_result(x1_list, x2_list, world_size)
        self._test_multiprocess(TestMmAllReduceBase._test_npu_mm_all_reduce_base,
                                TestMmAllReduceBase._init_dist_hccl, [expt_out_list, x1_list, x2_list, world_size])

    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend910B', 'Ascend950'])
    def test_npu_mm_all_reduce_base_fp16_fp16_3d(self):
        """
        用例3: 非量化场景 - x1=float16, x2=float16, 3维输入
        """
        world_size = 8
        b, s, k, n = 2, 64, 512, 256
        x1_list = []
        x2_list = []
        for _ in range(world_size):
            x1 = torch.randn(b, s, k, dtype=torch.float16).uniform_(-1, 1)
            x2 = torch.randn(k, n, dtype=torch.float16).uniform_(-1, 1)
            x1_list.append(x1)
            x2_list.append(x2)
        expt_out_list = self._construct_excepted_result_3d(x1_list, x2_list, world_size)
        self._test_multiprocess(TestMmAllReduceBase._test_npu_mm_all_reduce_base,
                                TestMmAllReduceBase._init_dist_hccl, [expt_out_list, x1_list, x2_list, world_size])

    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend910B', 'Ascend950'])
    def test_npu_mm_all_reduce_base_bf16_bf16_3d(self):
        """
        用例4: 非量化场景 - x1=bfloat16, x2=bfloat16, 3维输入
        """
        world_size = 8
        b, s, k, n = 2, 64, 512, 256
        x1_list = []
        x2_list = []
        for _ in range(world_size):
            x1 = torch.randn(b, s, k, dtype=torch.bfloat16).uniform_(-1, 1)
            x2 = torch.randn(k, n, dtype=torch.bfloat16).uniform_(-1, 1)
            x1_list.append(x1)
            x2_list.append(x2)
        expt_out_list = self._construct_excepted_result_3d(x1_list, x2_list, world_size)
        self._test_multiprocess(TestMmAllReduceBase._test_npu_mm_all_reduce_base,
                                TestMmAllReduceBase._init_dist_hccl, [expt_out_list, x1_list, x2_list, world_size])

    @classmethod
    def _test_npu_mm_all_reduce_base_quant(cls, rank, input_list):
        x1, x2, scale, offset, world_size, init_pg, c2p = input_list
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
        out = torch_npu.npu_mm_all_reduce_base(x1, x2, hcom_name, reduce_op='sum', antiquant_scale=scale,
                                               antiquant_offset=offset, bias=None, comm_turn=0)

        c2p.put((rank, out.cpu()))
        pg.barrier()

    def _test_multiprocess_quant(self, f, init_pg, input_list):
        expt_out_list, x1_list, x2_list, scale, offset, world_size = input_list
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)
        ps = []

        for i in range(world_size):
            p = ctx.Process(
                target=f,
                args=(i, [x1_list[i], x2_list[i], scale, offset, world_size, init_pg, c2p]))
            p.start()
            ps.append(p)

        for p in ps:
            p.join()

    def _construct_excepted_result_quant(self, x1_list, x2_list, scale, offset, world_size):
        out = None
        out_list = []
        for i in range(world_size):
            x1 = x1_list[i]
            x2 = x2_list[i]
            weight = torch.add(x2.to(torch.float32), offset.to(torch.float32))
            dequant = torch.mul(weight, scale.to(torch.float32))
            out_single = torch.matmul(x1.to(torch.float32), dequant)
            if out is None:
                out = out_single
            else:
                out = torch.add(out, out_single)
        for i in range(world_size):
            out_list.append(out.to(x1_list[0].dtype))
        return out_list

    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend910B', 'Ascend950'])
    def test_npu_mm_all_reduce_base_fp16_int8_perchannel(self):
        """
        用例5: 伪量化场景 - x1=float16, x2=int8, perchannel伪量化
        """
        world_size = 8
        m, k, n = 1, 256, 256
        x1_list = []
        x2_list = []
        scale = torch.randn(n, dtype=torch.float16).uniform_(0.001, 0.01)
        offset = torch.randn(n, dtype=torch.float16).uniform_(-1, 1)
        for _ in range(world_size):
            x1 = torch.randn(m, k, dtype=torch.float16).uniform_(-1, 1)
            x2 = torch.randint(-128, 127, (k, n), dtype=torch.int8)
            x1_list.append(x1)
            x2_list.append(x2)
        expt_out_list = self._construct_excepted_result_quant(x1_list, x2_list, scale, offset, world_size)
        self._test_multiprocess_quant(TestMmAllReduceBase._test_npu_mm_all_reduce_base_quant,
                                      TestMmAllReduceBase._init_dist_hccl,
                                      [expt_out_list, x1_list, x2_list, scale, offset, world_size])

    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend910B', 'Ascend950'])
    def test_npu_mm_all_reduce_base_bf16_int8_perchannel(self):
        """
        用例6: 伪量化场景 - x1=bfloat16, x2=int8, perchannel伪量化
        """
        world_size = 8
        m, k, n = 1, 256, 256
        x1_list = []
        x2_list = []
        scale = torch.randn(n, dtype=torch.bfloat16).uniform_(0.001, 0.01)
        offset = torch.randn(n, dtype=torch.bfloat16).uniform_(-1, 1)
        for _ in range(world_size):
            x1 = torch.randn(m, k, dtype=torch.bfloat16).uniform_(-1, 1)
            x2 = torch.randint(-128, 127, (k, n), dtype=torch.int8)
            x1_list.append(x1)
            x2_list.append(x2)
        expt_out_list = self._construct_excepted_result_quant(x1_list, x2_list, scale, offset, world_size)
        self._test_multiprocess_quant(TestMmAllReduceBase._test_npu_mm_all_reduce_base_quant,
                                      TestMmAllReduceBase._init_dist_hccl,
                                      [expt_out_list, x1_list, x2_list, scale, offset, world_size])

    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend910B', 'Ascend950'])
    def test_npu_mm_all_reduce_base_fp16_int8_pertensor(self):
        """
        用例7: 伪量化场景 - x1=float16, x2=int8, pertensor伪量化
        """
        world_size = 8
        m, k, n = 1, 256, 256
        x1_list = []
        x2_list = []
        scale = torch.randn(1, dtype=torch.float16).uniform_(0.001, 0.01)
        offset = torch.randn(1, dtype=torch.float16).uniform_(-1, 1)
        for _ in range(world_size):
            x1 = torch.randn(m, k, dtype=torch.float16).uniform_(-1, 1)
            x2 = torch.randint(-128, 127, (k, n), dtype=torch.int8)
            x1_list.append(x1)
            x2_list.append(x2)
        expt_out_list = self._construct_excepted_result_quant(x1_list, x2_list, scale, offset, world_size)
        self._test_multiprocess_quant(TestMmAllReduceBase._test_npu_mm_all_reduce_base_quant,
                                      TestMmAllReduceBase._init_dist_hccl,
                                      [expt_out_list, x1_list, x2_list, scale, offset, world_size])

    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend910B', 'Ascend950'])
    def test_npu_mm_all_reduce_base_bf16_int8_pertensor(self):
        """
        用例8: 伪量化场景 - x1=bfloat16, x2=int8, pertensor伪量化
        """
        world_size = 8
        m, k, n = 1, 256, 256
        x1_list = []
        x2_list = []
        scale = torch.randn(1, dtype=torch.bfloat16).uniform_(0.001, 0.01)
        offset = torch.randn(1, dtype=torch.bfloat16).uniform_(-1, 1)
        for _ in range(world_size):
            x1 = torch.randn(m, k, dtype=torch.bfloat16).uniform_(-1, 1)
            x2 = torch.randint(-128, 127, (k, n), dtype=torch.int8)
            x1_list.append(x1)
            x2_list.append(x2)
        expt_out_list = self._construct_excepted_result_quant(x1_list, x2_list, scale, offset, world_size)
        self._test_multiprocess_quant(TestMmAllReduceBase._test_npu_mm_all_reduce_base_quant,
                                      TestMmAllReduceBase._init_dist_hccl,
                                      [expt_out_list, x1_list, x2_list, scale, offset, world_size])

    @classmethod
    def _test_npu_mm_all_reduce_base_dequant(cls, rank, input_list):
        x1, x2, dequant_scale, world_size, init_pg, c2p = input_list
        pg = init_pg(rank, world_size)
        group = pg.distributed_c10d._get_default_group()
        if torch.__version__ > '2.0.1':
            hcom_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
        else:
            hcom_name = group.get_hccl_comm_name(rank)

        x1 = x1.npu()
        x2 = x2.npu()
        dequant_scale = dequant_scale.npu()
        out = torch_npu.npu_mm_all_reduce_base(x1, x2, hcom_name, reduce_op='sum', dequant_scale=dequant_scale,
                                               bias=None, comm_turn=0)

        c2p.put((rank, out.cpu()))
        pg.barrier()

    def _test_multiprocess_dequant(self, f, init_pg, input_list):
        expt_out_list, x1_list, x2_list, dequant_scale, world_size = input_list
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)
        ps = []

        for i in range(world_size):
            p = ctx.Process(
                target=f,
                args=(i, [x1_list[i], x2_list[i], dequant_scale, world_size, init_pg, c2p]))
            p.start()
            ps.append(p)

        for p in ps:
            p.join()

    def _construct_excepted_result_dequant(self, x1_list, x2_list, dequant_scale, world_size):
        out = None
        out_list = []
        for i in range(world_size):
            x1 = x1_list[i]
            x2 = x2_list[i]
            out_mm = torch.matmul(x1.to(torch.float32), x2.to(torch.float32))
            out_single = torch.mul(out_mm, dequant_scale.to(torch.float32))
            if out is None:
                out = out_single
            else:
                out = torch.add(out, out_single)
        for i in range(world_size):
            out_list.append(out.to(torch.float16))
        return out_list

    @skipIfUnsupportMultiNPU(2)
    @SupportedDevices(['Ascend910B', 'Ascend950'])
    def test_npu_mm_all_reduce_base_int8_int8_dequant_int64(self):
        """
        用例9: 全量化场景 - x1=int8, x2=int8, dequant_scale=int64
        """
        world_size = 2
        m, k, n = 1, 256, 256
        x1_list = []
        x2_list = []
        scale = torch.randint(1, 100, (n,), dtype=torch.int64)
        for _ in range(world_size):
            x1 = torch.randint(-128, 127, (m, k), dtype=torch.int8)
            x2 = torch.randint(-128, 127, (k, n), dtype=torch.int8)
            x1_list.append(x1)
            x2_list.append(x2)
        expt_out_list = self._construct_excepted_result_dequant(x1_list, x2_list, scale, world_size)
        self._test_multiprocess_dequant(TestMmAllReduceBase._test_npu_mm_all_reduce_base_dequant,
                                        TestMmAllReduceBase._init_dist_hccl,
                                        [expt_out_list, x1_list, x2_list, scale, world_size])

    @skipIfUnsupportMultiNPU(2)
    @SupportedDevices(['Ascend910B', 'Ascend950'])
    def test_npu_mm_all_reduce_base_int8_int8_dequant_bf16(self):
        """
        用例10: 全量化场景 - x1=int8, x2=int8, dequant_scale=bfloat16
        """
        world_size = 2
        m, k, n = 1, 256, 256
        x1_list = []
        x2_list = []
        scale = torch.randn(n, dtype=torch.bfloat16).uniform_(0.001, 0.01)
        for _ in range(world_size):
            x1 = torch.randint(-128, 127, (m, k), dtype=torch.int8)
            x2 = torch.randint(-128, 127, (k, n), dtype=torch.int8)
            x1_list.append(x1)
            x2_list.append(x2)
        expt_out_list = self._construct_excepted_result_dequant(x1_list, x2_list, scale, world_size)
        self._test_multiprocess_dequant(TestMmAllReduceBase._test_npu_mm_all_reduce_base_dequant,
                                        TestMmAllReduceBase._init_dist_hccl,
                                        [expt_out_list, x1_list, x2_list, scale, world_size])

    @classmethod
    def _test_npu_mm_all_reduce_base_dequant_pertoken(cls, rank, input_list):
        x1, x2, dequant_scale, pertoken_scale, world_size, init_pg, c2p = input_list
        pg = init_pg(rank, world_size)
        group = pg.distributed_c10d._get_default_group()
        if torch.__version__ > '2.0.1':
            hcom_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
        else:
            hcom_name = group.get_hccl_comm_name(rank)

        x1 = x1.npu()
        x2 = x2.npu()
        dequant_scale = dequant_scale.npu()
        pertoken_scale = pertoken_scale.npu()
        out = torch_npu.npu_mm_all_reduce_base(x1, x2, hcom_name, reduce_op='sum', dequant_scale=dequant_scale,
                                               pertoken_scale=pertoken_scale, bias=None, comm_turn=0)

        c2p.put((rank, out.cpu()))
        pg.barrier()

    def _test_multiprocess_dequant_pertoken(self, f, init_pg, input_list):
        expt_out_list, x1_list, x2_list, dequant_scale, pertoken_scale, world_size = input_list
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)
        ps = []

        for i in range(world_size):
            p = ctx.Process(
                target=f,
                args=(i, [x1_list[i], x2_list[i], dequant_scale, pertoken_scale, world_size, init_pg, c2p]))
            p.start()
            ps.append(p)

        for p in ps:
            p.join()

    def _construct_excepted_result_dequant_pertoken(self, x1_list, x2_list, dequant_scale, pertoken_scale, world_size):
        out = None
        out_list = []
        for i in range(world_size):
            x1 = x1_list[i]
            x2 = x2_list[i]
            out_mm = torch.matmul(x1.to(torch.float32), x2.to(torch.float32))
            out_single = torch.mul(out_mm, dequant_scale.to(torch.float32))
            out_single = torch.mul(out_single, pertoken_scale.unsqueeze(1).to(torch.float32))
            if out is None:
                out = out_single
            else:
                out = torch.add(out, out_single)
        for i in range(world_size):
            out_list.append(out.to(torch.float16))
        return out_list

    @skipIfUnsupportMultiNPU(2)
    @SupportedDevices(['Ascend910B', 'Ascend950'])
    def test_npu_mm_all_reduce_base_int8_int8_dequant_fp32_pertoken_fp32(self):
        """
        用例11: 全量化场景 - x1=int8, x2=int8, dequant_scale=float32, pertoken_scale=float32
        """
        world_size = 2
        m, k, n = 1, 256, 256
        x1_list = []
        x2_list = []
        scale = torch.randn(n, dtype=torch.float32).uniform_(0.001, 0.01)
        pertoken_scale = torch.randn(m, dtype=torch.float32).uniform_(0.001, 0.01)
        for _ in range(world_size):
            x1 = torch.randint(-128, 127, (m, k), dtype=torch.int8)
            x2 = torch.randint(-128, 127, (k, n), dtype=torch.int8)
            x1_list.append(x1)
            x2_list.append(x2)
        expt_out_list = self._construct_excepted_result_dequant_pertoken(x1_list, x2_list, scale, pertoken_scale, world_size)
        self._test_multiprocess_dequant_pertoken(TestMmAllReduceBase._test_npu_mm_all_reduce_base_dequant_pertoken,
                                                 TestMmAllReduceBase._init_dist_hccl,
                                                 [expt_out_list, x1_list, x2_list, scale, pertoken_scale, world_size])

    @skipIfUnsupportMultiNPU(2)
    @SupportedDevices(['Ascend910B', 'Ascend950'])
    def test_npu_mm_all_reduce_base_int8_int8_dequant_bf16_pertoken_fp32(self):
        """
        用例12: 全量化场景 - x1=int8, x2=int8, dequant_scale=bfloat16, pertoken_scale=float32
        """
        world_size = 2
        m, k, n = 1, 256, 256
        x1_list = []
        x2_list = []
        scale = torch.randn(n, dtype=torch.bfloat16).uniform_(0.001, 0.01)
        pertoken_scale = torch.randn(m, dtype=torch.float32).uniform_(0.001, 0.01)
        for _ in range(world_size):
            x1 = torch.randint(-128, 127, (m, k), dtype=torch.int8)
            x2 = torch.randint(-128, 127, (k, n), dtype=torch.int8)
            x1_list.append(x1)
            x2_list.append(x2)
        expt_out_list = self._construct_excepted_result_dequant_pertoken(x1_list, x2_list, scale, pertoken_scale, world_size)
        self._test_multiprocess_dequant_pertoken(TestMmAllReduceBase._test_npu_mm_all_reduce_base_dequant_pertoken,
                                                 TestMmAllReduceBase._init_dist_hccl,
                                                 [expt_out_list, x1_list, x2_list, scale, pertoken_scale, world_size])

    @classmethod
    def _test_npu_mm_all_reduce_base_fp8(cls, rank, input_list):
        x1, x2, dequant_scale, world_size, x1_dtype, x2_dtype, output_dtype, init_pg, c2p = input_list
        pg = init_pg(rank, world_size)
        group = pg.distributed_c10d._get_default_group()
        if torch.__version__ > '2.0.1':
            hcom_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
        else:
            hcom_name = group.get_hccl_comm_name(rank)

        x1 = x1.npu()
        x2 = x2.npu()
        dequant_scale = dequant_scale.npu()
        out = torch_npu.npu_mm_all_reduce_base(x1, x2, hcom_name, reduce_op='sum', dequant_scale=dequant_scale,
                                               bias=None, comm_turn=0, x1_dtype=x1_dtype, x2_dtype=x2_dtype,
                                               y_dtype=output_dtype)

        c2p.put((rank, out.cpu()))
        pg.barrier()

    def _test_multiprocess_fp8(self, f, init_pg, input_list):
        expt_out_list, x1_list, x2_list, dequant_scale, world_size, x1_dtype, x2_dtype, output_dtype = input_list
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)
        ps = []

        for i in range(world_size):
            p = ctx.Process(
                target=f,
                args=(i, [x1_list[i], x2_list[i], dequant_scale, world_size, x1_dtype, x2_dtype, output_dtype, init_pg, c2p]))
            p.start()
            ps.append(p)

        for p in ps:
            p.join()

    def _construct_excepted_result_fp8(self, x1_list, x2_list, dequant_scale, world_size, output_dtype):
        out = None
        out_list = []
        for i in range(world_size):
            x1 = x1_list[i].to(torch.float32)
            x2 = x2_list[i].to(torch.float32)
            out_mm = torch.matmul(x1, x2)
            out_single = torch.mul(out_mm, dequant_scale.to(torch.float32))
            if out is None:
                out = out_single
            else:
                out = torch.add(out, out_single)
        for i in range(world_size):
            out_list.append(out.to(output_dtype))
        return out_list

    @skipIfUnsupportMultiNPU(2)
    @SupportedDevices(['Ascend950'])
    def test_npu_mm_all_reduce_base_fp8_e4m3fn_fp8_e4m3fn(self):
        """
        用例13: 扩展场景 - x1=float8_e4m3fn, x2=float8_e4m3fn
        """
        world_size = 2
        m, k, n = 16, 256, 256
        x1 = torch_npu.npu_dtype_cast(torch.randint(-5, 5, (m, k)), torch.float8_e4m3fn)
        x2 = torch_npu.npu_dtype_cast(torch.randint(-5, 5, (k, n)), torch.float8_e4m3fn)
        dequant_scale = torch.randn(n, dtype=torch.float32).uniform_(0.001, 0.01)
        x1_list = [x1.clone() for _ in range(world_size)]
        x2_list = [x2.clone() for _ in range(world_size)]
        expt_out_list = self._construct_excepted_result_fp8(x1_list, x2_list, dequant_scale, world_size, torch.bfloat16)
        self._test_multiprocess_fp8(TestMmAllReduceBase._test_npu_mm_all_reduce_base_fp8,
                                    TestMmAllReduceBase._init_dist_hccl,
                                    [expt_out_list, x1_list, x2_list, dequant_scale, world_size,
                                     torch_npu.hifloat8, torch_npu.hifloat8, torch.bfloat16])

    @skipIfUnsupportMultiNPU(2)
    @SupportedDevices(['Ascend950'])
    def test_npu_mm_all_reduce_base_fp8_e5m2_fp8_e5m2(self):
        """
        用例14: 扩展场景 - x1=float8_e5m2, x2=float8_e5m2
        """
        world_size = 2
        m, k, n = 16, 256, 256
        x1 = torch_npu.npu_dtype_cast(torch.randint(-5, 5, (m, k)), torch.float8_e5m2)
        x2 = torch_npu.npu_dtype_cast(torch.randint(-5, 5, (k, n)), torch.float8_e5m2)
        dequant_scale = torch.randn(n, dtype=torch.float32).uniform_(0.001, 0.01)
        x1_list = [x1.clone() for _ in range(world_size)]
        x2_list = [x2.clone() for _ in range(world_size)]
        expt_out_list = self._construct_excepted_result_fp8(x1_list, x2_list, dequant_scale, world_size, torch.bfloat16)
        self._test_multiprocess_fp8(TestMmAllReduceBase._test_npu_mm_all_reduce_base_fp8,
                                    TestMmAllReduceBase._init_dist_hccl,
                                    [expt_out_list, x1_list, x2_list, dequant_scale, world_size,
                                     torch_npu.hifloat8, torch_npu.hifloat8, torch.bfloat16])

    @skipIfUnsupportMultiNPU(2)
    @SupportedDevices(['Ascend950'])
    def test_npu_mm_all_reduce_base_hifloat8_hifloat8(self):
        """
        用例15: 扩展场景 - x1=hifloat8, x2=hifloat8
        """
        world_size = 2
        m, k, n = 16, 256, 256
        x1_int8 = torch.randint(-5, 5, (m, k), dtype=torch.int8)
        x2_int8 = torch.randint(-5, 5, (k, n), dtype=torch.int8)
        x1 = torch_npu.HiFloat8Tensor.to_hifloat8(x1_int8)
        x2 = torch_npu.HiFloat8Tensor.to_hifloat8(x2_int8)
        dequant_scale = torch.randn(n, dtype=torch.float32).uniform_(0.001, 0.01)
        x1_list = [x1.clone() for _ in range(world_size)]
        x2_list = [x2.clone() for _ in range(world_size)]
        expt_out_list = self._construct_excepted_result_fp8(x1_list, x2_list, dequant_scale, world_size, torch.bfloat16)
        self._test_multiprocess_fp8(TestMmAllReduceBase._test_npu_mm_all_reduce_base_fp8,
                                    TestMmAllReduceBase._init_dist_hccl,
                                    [expt_out_list, x1_list, x2_list, dequant_scale, world_size,
                                     torch_npu.hifloat8, torch_npu.hifloat8, torch.bfloat16])

    @skipIfUnsupportMultiNPU(2)
    @SupportedDevices(['Ascend950'])
    def test_npu_mm_all_reduce_base_fp8_e4m3fn_fp8_e5m2(self):
        """
        用例16: 扩展场景 - x1=float8_e4m3fn, x2=float8_e5m2
        """
        world_size = 2
        m, k, n = 16, 256, 256
        x1 = torch_npu.npu_dtype_cast(torch.randint(-5, 5, (m, k)), torch.float8_e4m3fn)
        x2 = torch_npu.npu_dtype_cast(torch.randint(-5, 5, (k, n)), torch.float8_e5m2)
        dequant_scale = torch.randn(n, dtype=torch.float32).uniform_(0.001, 0.01)
        x1_list = [x1.clone() for _ in range(world_size)]
        x2_list = [x2.clone() for _ in range(world_size)]
        expt_out_list = self._construct_excepted_result_fp8(x1_list, x2_list, dequant_scale, world_size, torch.bfloat16)
        self._test_multiprocess_fp8(TestMmAllReduceBase._test_npu_mm_all_reduce_base_fp8,
                                    TestMmAllReduceBase._init_dist_hccl,
                                    [expt_out_list, x1_list, x2_list, dequant_scale, world_size,
                                     torch_npu.hifloat8, torch_npu.hifloat8, torch.bfloat16])

    @skipIfUnsupportMultiNPU(2)
    @SupportedDevices(['Ascend950'])
    def test_npu_mm_all_reduce_base_fp8_e5m2_fp8_e4m3fn(self):
        """
        用例17: 扩展场景 - x1=float8_e5m2, x2=float8_e4m3fn
        """
        world_size = 2
        m, k, n = 16, 256, 256
        x1 = torch_npu.npu_dtype_cast(torch.randint(-5, 5, (m, k)), torch.float8_e5m2)
        x2 = torch_npu.npu_dtype_cast(torch.randint(-5, 5, (k, n)), torch.float8_e4m3fn)
        dequant_scale = torch.randn(n, dtype=torch.float32).uniform_(0.001, 0.01)
        x1_list = [x1.clone() for _ in range(world_size)]
        x2_list = [x2.clone() for _ in range(world_size)]
        expt_out_list = self._construct_excepted_result_fp8(x1_list, x2_list, dequant_scale, world_size, torch.bfloat16)
        self._test_multiprocess_fp8(TestMmAllReduceBase._test_npu_mm_all_reduce_base_fp8,
                                    TestMmAllReduceBase._init_dist_hccl,
                                    [expt_out_list, x1_list, x2_list, dequant_scale, world_size,
                                     torch_npu.hifloat8, torch_npu.hifloat8, torch.bfloat16])

    @skipIfUnsupportMultiNPU(2)
    @SupportedDevices(['Ascend950'])
    def test_npu_mm_all_reduce_base_hifloat8_fp8_e4m3fn(self):
        """
        用例18: 扩展场景 - x1=hifloat8, x2=float8_e4m3fn
        """
        world_size = 2
        m, k, n = 16, 256, 256
        x1_int8 = torch.randint(-5, 5, (m, k), dtype=torch.int8)
        x1 = torch_npu.HiFloat8Tensor.to_hifloat8(x1_int8)
        x2 = torch_npu.npu_dtype_cast(torch.randint(-5, 5, (k, n)), torch.float8_e4m3fn)
        dequant_scale = torch.randn(n, dtype=torch.float32).uniform_(0.001, 0.01)
        x1_list = [x1.clone() for _ in range(world_size)]
        x2_list = [x2.clone() for _ in range(world_size)]
        expt_out_list = self._construct_excepted_result_fp8(x1_list, x2_list, dequant_scale, world_size, torch.bfloat16)
        self._test_multiprocess_fp8(TestMmAllReduceBase._test_npu_mm_all_reduce_base_fp8,
                                    TestMmAllReduceBase._init_dist_hccl,
                                    [expt_out_list, x1_list, x2_list, dequant_scale, world_size,
                                     torch_npu.hifloat8, torch_npu.hifloat8, torch.bfloat16])

    @skipIfUnsupportMultiNPU(2)
    @SupportedDevices(['Ascend950'])
    def test_npu_mm_all_reduce_base_fp8_e4m3fn_hifloat8(self):
        """
        用例19: 扩展场景 - x1=float8_e4m3fn, x2=hifloat8
        """
        world_size = 2
        m, k, n = 16, 256, 256
        x1 = torch_npu.npu_dtype_cast(torch.randint(-5, 5, (m, k)), torch.float8_e4m3fn)
        x2_int8 = torch.randint(-5, 5, (k, n), dtype=torch.int8)
        x2 = torch_npu.HiFloat8Tensor.to_hifloat8(x2_int8)
        dequant_scale = torch.randn(n, dtype=torch.float32).uniform_(0.001, 0.01)
        x1_list = [x1.clone() for _ in range(world_size)]
        x2_list = [x2.clone() for _ in range(world_size)]
        expt_out_list = self._construct_excepted_result_fp8(x1_list, x2_list, dequant_scale, world_size, torch.bfloat16)
        self._test_multiprocess_fp8(TestMmAllReduceBase._test_npu_mm_all_reduce_base_fp8,
                                    TestMmAllReduceBase._init_dist_hccl,
                                    [expt_out_list, x1_list, x2_list, dequant_scale, world_size,
                                     torch_npu.hifloat8, torch_npu.hifloat8, torch.bfloat16])

    @skipIfUnsupportMultiNPU(2)
    @SupportedDevices(['Ascend950'])
    def test_npu_mm_all_reduce_base_hifloat8_fp8_e5m2(self):
        """
        用例20: 扩展场景 - x1=hifloat8, x2=float8_e5m2
        """
        world_size = 2
        m, k, n = 16, 256, 256
        x1_int8 = torch.randint(-5, 5, (m, k), dtype=torch.int8)
        x1 = torch_npu.HiFloat8Tensor.to_hifloat8(x1_int8)
        x2 = torch_npu.npu_dtype_cast(torch.randint(-5, 5, (k, n)), torch.float8_e5m2)
        dequant_scale = torch.randn(n, dtype=torch.float32).uniform_(0.001, 0.01)
        x1_list = [x1.clone() for _ in range(world_size)]
        x2_list = [x2.clone() for _ in range(world_size)]
        expt_out_list = self._construct_excepted_result_fp8(x1_list, x2_list, dequant_scale, world_size, torch.bfloat16)
        self._test_multiprocess_fp8(TestMmAllReduceBase._test_npu_mm_all_reduce_base_fp8,
                                    TestMmAllReduceBase._init_dist_hccl,
                                    [expt_out_list, x1_list, x2_list, dequant_scale, world_size,
                                     torch_npu.hifloat8, torch_npu.hifloat8, torch.bfloat16])


if __name__ == '__main__':
    run_tests()

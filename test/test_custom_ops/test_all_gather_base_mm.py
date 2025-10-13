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
        x1_list, x2_list, x1_scale_list, x2_scale_list, world_size, comm_mode, output_dtype, init_pg, c2p = input_list
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
        x1_scale = x1_scale_list[rank].npu() if x1_scale_list else None
        x2_scale = x2_scale_list[rank].npu() if x2_scale_list else None
        out, gather_out = torch_npu.npu_all_gather_base_mm(x1,
                                                           x2,
                                                           hcom_name,
                                                           world_size,
                                                           bias=None,
                                                           x1_scale=x1_scale,
                                                           x2_scale=x2_scale,
                                                           gather_index=0,
                                                           gather_output=True,
                                                           output_dtype=output_dtype,
                                                           comm_turn=0,
                                                           comm_mode=comm_mode)
        c2p.put((rank, out.cpu().numpy(), gather_out.cpu().numpy()))
        pg.barrier()

    def _test_multiprocess(self, f, init_pg, input_list):
        expt_out_list, expt_gather, x1, x2, x1_scale_list, x2_scale_list, world_size, comm_mode, output_dtype = input_list
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)
        ps = []

        for i in range(world_size):
            p = ctx.Process(
                target=f,
                args=(i, [x1, x2, x1_scale_list, x2_scale_list, world_size, comm_mode, output_dtype, init_pg, c2p]))
            p.start()
            ps.append(p)

        for _ in range(world_size):
            rank, output, gather_output = c2p.get()
            output, gather_output = torch.from_numpy(output), torch.from_numpy(gather_output)
            self.assertEqual(output, expt_out_list[rank],
                             ("rank {} Expect receive tensor {} but got {}.").format(rank, expt_out_list[rank], output))
            self.assertEqual(gather_output, expt_gather,
                             ("rank {} Expect receive tensor {} but got {}.").format(rank, expt_gather, gather_output))
        for p in ps:
            p.join()

    def _construct_excepted_result(self, x1_list, x2_list, world_size, x1_scale_list=None, x2_scale_list=None, output_dtype=None):
        gather_out = torch.cat(x1_list)
        if x1_scale_list:
            x1_scale = torch.cat(x1_scale_list)
        out_list = []
        if output_dtype:
            out_dtype = output_dtype
        else:
            out_dtype = gather_out.dtype
        for i in range(world_size):
            gather_out_npu, x2_list_npu = gather_out.npu(), x2_list[i].npu()
            if x1_scale_list:
                mm_res = torch_npu.npu_quant_matmul(x1=gather_out_npu, x2=x2_list_npu, scale=x2_scale_list[i].squeeze(0).npu(), pertoken_scale=x1_scale.squeeze(-1).npu(), output_dtype=out_dtype)
            elif x2_scale_list:
                mm_res = torch_npu.npu_quant_matmul(x1=gather_out_npu, x2=x2_list_npu, scale=x2_scale_list[i].squeeze(0).npu(), output_dtype=out_dtype)
            else:
                mm_res = torch.matmul(gather_out_npu, x2_list_npu)
            out_list.append(mm_res.to(out_dtype).cpu())
        return out_list, gather_out_npu.cpu()

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
        for comm_mode in ['aiv', 'ai_cpu']:
            self._test_multiprocess(TestAllGatherBaseMm._test_npu_all_gather_base_mm,
                                TestAllGatherBaseMm._init_dist_hccl, [expt_out_list, expt_gather, x1_list, x2_list, None, None, world_size, comm_mode, None])

    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend910B'])
    def test_npu_all_gather_quant_mm(self):
        world_size = 8
        m, k, n = 16, 512, 256
        output_dtype = torch.float16
        x1_list = []
        x2_list = []
        x1_scale_list = []
        x2_scale_list = []
        for _ in range(world_size):
            x1 = torch.randint(-10, 10, size=(m, k), dtype=torch.int8)
            x2 = torch.randint(-10, 10, size=(k, n), dtype=torch.int8)
            x1_scale = torch.randn((m, 1), dtype=torch.float32)
            x2_scale = torch.randn((1, n), dtype=torch.float32)
            x1_list.append(x1)
            x2_list.append(x2)
            x1_scale_list.append(x1_scale)
            x2_scale_list.append(x2_scale)
        expt_out_list, expt_gather = self._construct_excepted_result(x1_list, x2_list, world_size, x1_scale_list, x2_scale_list, output_dtype)
        self._test_multiprocess(TestAllGatherBaseMm._test_npu_all_gather_base_mm,
                                TestAllGatherBaseMm._init_dist_hccl, [expt_out_list, expt_gather, x1_list, x2_list, x1_scale_list, x2_scale_list, world_size, 'aiv', output_dtype])


if __name__ == '__main__':
    run_tests()

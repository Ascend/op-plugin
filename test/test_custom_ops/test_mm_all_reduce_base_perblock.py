import os
import copy
import unittest

import numpy as np
import torch
import multiprocessing
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
    def _test_npu_mm_all_reduce_base_perblock(cls, rank, input_list):
        x1, x2, world_size, init_pg, c2p = input_list
        pg = init_pg(rank, world_size)
        group = pg.distributed_c10d._get_default_group()
        if torch.__version__ > '2.0.1':
            hcom_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
        else:
            hcom_name = group.get_hccl_comm_name(rank)

        x1 = x1.npu()
        x2 = x2.npu()
        out = torch_npu.npu_mm_all_reduce_base(x1, x2, hcom_name, reduce_op='sum', bias=None, comm_turn=0, pertoken_scale=pertoken_scale, dequant_scale=dequant_scale,
                                               y_dtype=y_dtype, x1_dtype=x1_dtype, x2_dtype=x2_dtype, pertoken_scale_dtype=pertoken_scale_dtype, dequant_scale_dtype=dequant_scale_dtype,
                                               group_sizes=group_sizes)

        c2p.put((rank, out.cpu()))
        pg.barrier()

    def _test_multiprocess(self, f, init_pg, input_list):
        expt_out_list, x1_list, x2_list, world_size, x1_scale_list, x2_scale_list, y_dtype, x1_dtype, x2_dtype, pertoken_scale_dtype, dequant_scale_dtype, group_sizes = input_list
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)
        ps = []

        for i in range(world_size):
            p = ctx.Process(
                target=f,
                args=(i, [x1_list[i], x2_list[i], world_size, x1_scale_list[i], x2_sacle_list[i], 
                      y_dtype, x1_dtype, x2_dtype, pertoken_scale_dtype, dequant_scale_dtype, group_sizes, init_pg, c2p]))
            p.start()
            ps.append(p)

        for _ in range(world_size):
            rank, output = c2p.get()
            self.assertRtolEqual(output, expt_out_list[rank], 0.05, 0.05)

        for p in ps:
            p.join()

    def _trans_np_hifuint8_tensor_to_float32(in_tensor):
        shape_tensor = in_tensor.shape
        multi_shape = np.prod(shape_tensor)
        print(f"[trans_np_hifuint8_tensor_to_float32]total size:{multi_shape}")
        # 扁平化输入张量以便分块
        flat_tensor = in_tensor.reshape(multi_shape)
        # 分块处理：根据CPU核心数确定块大小
        num_processes = multiprocessing.cpu_count()
        # 每个进程处理4块，平衡并行效率和内存占用
        chunk_size = max(1, multi_shape // (num_processes * 4))
        chunks = [flat_tensor[i:i + chunk_size] for i in range(0, multi_shape, chunk_size)]
        # 多进程并行处理所有块
        with multiprocessing.Pool(processes=num_processes) as pool:
            chunk_results = pool.map(_process_chunk, chunks)

        # 合并结果并恢复原始形状
        out_tensor = np.concatenate(chunk_results).reshape(shape_tensor).astype(np.float32)
        return out_tensor

    def _fetch_batch_broadcast(batch_x1, batch_x2):
        batch_out = copy.deepcopy(batch_x1) if len(batch_x1) > len(batch_x2) else copy.deepcopy(batch_x2)
        # 当x1和x2 batch维度不相同时且均不为0时，需要根据x1和x2中batch情况进行做broadcast 处理
        # 前面的batch_out只是获取到最大维度的batch，但存在列外，如[2,2,2,2,1,1]和[4,4,4,1,1], 实际batchout应为[2,4,4,4]
        min_len, max_len = 0, 0
        if batch_x2 != batch_x1 and batch_x1 and batch_x2:
            min_len = min(len(batch_x1), len(batch_x2))
            max_len = max(len(batch_x1), len(batch_x2))

        # 更新 batch_out 的前 min_len 个元素
        for idx in range(min_len):
            batch_out[-(idx + 1)] = max(batch_x1[-(idx + 1)], batch_x2[-(idx + 1)])

        # 如果 batch_x1 更长，更新剩余部分
        if len(batch_x1) > len(batch_x2):
            for idx in range(min_len, max_len):
                batch_out[-(idx + 1)] = batch_x1[-(idx + 1)]
        # 如果 batch_x2 更长，更新剩余部分
        else:
            for idx in range(min_len, max_len):
                batch_out[-(idx + 1)] = batch_x2[-(idx + 1)]
        return batch_out

    def _value_batch_broadcast(x, batch):
        new_x_shape = batch + list(x.shape[-2:])
        x = torch.broadcast_to(x, new_x_shape)
        return x

    def _per_block_cpu_compute(group_size, x1, x2, x1_scale, x2_scale):
        if x1.dim() != x1_scale.dim():
            print(f"[ERROR] x1.dim() != x1_scale.dim(),x1.dim()={x1.dim()}, x1_scale.dim()={x1_scale.dim()}")
        if x2.dim() != x2_scale.dim():
            print(f"[ERROR] x2.dim() != x2_scale.dim(),x2.dim()={x2.dim()}, x2_scale.dim()={x2_scale.dim()}")
        batch_x1 = np.array(x1.shape[:-2]).astype(int).tolist()
        batch_x2 = np.array(x2.shape[:-2]).astype(int).tolist()
        # 获取x1和x2broadcast的batch维度
        batch_out = self._fetch_batch_broadcast(batch_x1, batch_x2)
        # 基于batch_out的对x1和x1_scale 进行broadcast，如果维度不相等
        if batch_x1 != batch_out:
            x1 = self._value_batch_broadcast(x1, batch_out)
            x1_scale = self._value_batch_broadcast(x1_scale, batch_out)
        # 基于batch_out的对x2和x2_scale 进行broadcast，如果维度不相等
        if batch_x2 != batch_out:
            x2 = self._value_batch_broadcast(x2, batch_out)
            x2_scale = self._value_batch_broadcast(x2_scale, batch_out)

        batch_all = 1
        is2dim = True
        if batch_out != []:
            is2dim = False
            batch_all = np.prod(batch_out)
            x1 = x1.reshape([batch_all] + list(x1.shape[-2:]))
            x2 = x2.reshape([batch_all] + list(x2.shape[-2:]))
            x1_scale = x1_scale.reshape([batch_all] + list(x1_scale.shape[-2:]))
            x2_scale = x2_scale.reshape([batch_all] + list(x2_scale.shape[-2:]))
        m = x1.shape[-2]  # 非转置情况下，m是倒数第二维
        k = x1.shape[-1]  # 非转置情况下，k是倒数第一维
        n = x2.shape[-1]  # 非转置情况下，n是倒数第一维
        out = torch.zeros(m, n)
        if x2_scale.dim() > 2 or x1_scale.dim() > 2:
            out = torch.zeros(batch_all, m, n)
        group_size_m, group_size_n, group_size_k = group_size
        for i in range(batch_all):
            for m_idx in range((m + group_size_m - 1) // group_size_m):
                m_start = m_idx * group_size_m
                m_end = min((m_idx + 1) * group_size_m, m)
                for n_idx in range((n + group_size_n - 1) // group_size_n):
                    n_start = n_idx * group_size_n
                    n_end = min((n_idx + 1) * group_size_n, n)
                    for k_idx in range((k + group_size_k - 1) // group_size_k):
                        k_start = k_idx * group_size_k
                        k_end = min((k_idx + 1) * group_size_k, k)
                        if batch_all == 1 and is2dim:
                            block_output = torch.matmul(x1[m_start:m_end, k_start:k_end],
                                                        x2[k_start:k_end, n_start:n_end]) * x1_scale[m_idx, k_idx] * x2_scale[k_idx, n_idx]
                            out[m_start:m_end, n_start:n_end] += block_output
                        else:
                            out[i, m_start:m_end, n_start:n_end] += torch.matmul(x1[i, m_start:m_end, k_start:k_end],
                                                                                x2[i, k_start:k_end, n_start:n_end]) * x1_scale[
                                                                        i, m_idx, k_idx] * x2_scale[i, k_idx, n_idx]
        if x2_scale.dim() > 2 or x1_scale.dim() > 2:
            out_shape = batch_out
            out_shape.append(m)
            out_shape.append(n)
            out = torch.reshape(out, out_shape)
        else:
            out = torch.reshape(out, [m, n])
        return out

    def _construct_excepted_result(self, x1_list, x2_list, world_size, x1_scale_list, x2_scale_list, 
                                   y_dtype, x1_dtype, x2_dtype, pertoken_scale_dtype, dequant_scale_dtype, group_sizes):
        out = None
        out_list = []
        for i in range(world_size):
            x1 = x1_list[i]
            x2 = x2_list[i]
            x1_scale = x1_scale_list[i]
            x2_scale = x2_scale_list[i]
            x1 = torch.from_numpy(self._trans_np_hifuint8_tensor_to_float32(self.array(x1), x1_dtype))
            x2 = torch.from_numpy(self._trans_np_hifuint8_tensor_to_float32(self.array(x2), x2_dtype))
            x1_scale = x1_scale.to(torch.float32)
            x2_scale = x2_scale.to(torch.float32)
            out_single = self._per_block_cpu_compute(group_sizes, x1, x2, x1_scale, x2_scale)
            if out is None:
                out = out_single
            else:
                out = torch.add(out, out_single)
        for i in range(world_size):
            out_list.append(out.to(x1_list[0].dtype))
        return out_list

    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend910_95'])
    def test_npu_mm_all_reduce_base_perblock(self):
        world_size = 8
        x1_dtype = np.uint8
        x2_dtype = np.uint8
        pertoken_scale_dtype = np.float32
        dequant_scale_dtype = np.float32
        y_dtype = np.float32
        group_sizes = [128, 128, 128]
        data_format = -1
        x1_shape = [x1_dtype, data_format, [128, 512]]
        x2_shape = [x2_dtype, data_format, [512, 256]]
        x1_scale_shape = [pertoken_scale_dtype, data_format, [1, 4]]
        x2_scale_shape = [dequant_scale_dtype, data_format, [4, 2]]
        x1_list = []
        x2_list = []
        x1_scale_list = []
        x2_sacle_list = []
        for _ in range(world_size):
            x1, _ = create_common_tensor(x1_shape, -1, 1)
            x2, _ = create_common_tensor(x2_shape, -1, 1)
            x1_scale, _ = create_common_tensor(x1_scale_shape, -1, 1)
            x2_scale, _ = create_common_tensor(x2_scale_shape, -1, 1)
            x1_list.append(x1)
            x2_list.append(x2)
            x1_scale_list.append(x1_scale)
            x2_scale_list.append(x2_scale)
        expt_out_list = self._construct_excepted_result(x1_list, x2_list, world_size, x1_scale_list, x2_scale_list, 
                                                        y_dtype, x1_dtype, x2_dtype, pertoken_scale_dtype, dequant_scale_dtype, group_sizes)
        self._test_multiprocess(TestMmAllReduceBase._test_npu_mm_all_reduce_base_perblock,
                                TestMmAllReduceBase._init_dist_hccl, [expt_out_list, x1_list, x2_list, world_size, x1_scale_list, x2_scale_list, 
                                                                      y_dtype, x1_dtype, x2_dtype, pertoken_scale_dtype, dequant_scale_dtype, group_sizes])
      

if __name__ == '__main__':
    run_tests()

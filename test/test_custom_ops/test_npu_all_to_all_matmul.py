import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class TestAlltoAllMatmul(TestCase):
    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '50000'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist


    @classmethod
    def _test_npu_all_to_all_matmul(cls, rank, input_list):
        # Unpack inputs
        x1, x2, bias, world_size, init_pg, c2p = input_list

        # Initialize process group
        pg = init_pg(rank, world_size)
        group = pg.distributed_c10d._get_default_group()

        # Get HCCL communication name
        if torch.__version__ > '2.0.1':
            hcom_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
        else:
            hcom_name = group.get_hccl_comm_name(rank)

        # Move tensors to NPU
        x1 = x1.npu()
        x2 = x2.npu()
        bias = bias.npu()

        # Call NPU operator
        out, _ = torch_npu.npu_all_to_all_matmul(x1, x2, hcom_name, world_size, bias=bias, all2all_axes=[-2,-1], all2all_out_flag=True)

        # Return result
        c2p.put((rank, out.cpu()))
        pg.barrier()

    def _test_multiprocess(self, f, init_pg, input_list): # f function, init_pg function, input_list list[]
        # Input: expt_out_list, x1_list, x2_list, bias_list, ..., world_size
        expt_out_list, x1_list, x2_list, bias_list, world_size = input_list

        # mp是python中用于多进程并行计算的模块
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)
        ps = []

        for i in range(world_size):
            p = ctx.Process(
                target=f,
                # args for line30
                args=(i, [x1_list[i], x2_list[i], bias_list[i], world_size, init_pg, c2p])
            )
            p.start()
            ps.append(p)

        # Collect results
        for _ in range(world_size):
            rank, out_put = c2p.get()
            self.assertEqual(out_put, expt_out_list[rank],
                             ("rank {} Expect receive tensor {} but got {}.").format(rank, expt_out_list, out_put))

        # Wait for all processes to finish
        for p in ps:
            p.join()

    def _construct_excepted_result(self, x1_list, x2_list, bias_list, world_size):
        # 存储每个rank的golden输出
        expt_out_list = [None] * world_size
        for i in range(world_size):
            # 1、初始化入参，获取当前rank的输入
            x1 = x1_list[i].npu().to(torch.float32)
            x2 = x2_list[i].npu().to(torch.float32)

            # 2、准备进行alltoall通信前的shape重排
            temp_shape = [world_size, x1.shape[0] // world_size, x1.shape[1]]
            x1_reshape = x1.view(temp_shape).contiguous()

            # 3、准备一个全0张量，准备进行alltoall通信
            # alltoall_out是alltoall通信后的结果，因为接口需要，这里生成一个全0的张量，shape为[rankSize, BS/rankSize, H1]
            alltoall_out = torch.zeros(temp_shape, dtype=torch.float32).npu()

            # 4、开始alltoall通信
            # dist.all_to_all_single是torch仓的alltoall通信接口，第一个参数为输出output，第二个参数为输入input
            dist.all_to_all_single(alltoall_out, x1_reshape)
            # alltoall通信结束，alltoall_out已经是通信后的结果

            # 5、重排alltoall_out，准备执行matmul
            alltoall_shape = [temp_shape[1], temp_shape[0] * temp_shape[2]]
            alltoall_out = alltoall_out.permute(1, 0, 2).reshape(alltoall_shape)

            # 6、根据是否有bias，执行matmul
            if bias_list:
                bias = bias_list[i].npu().to(torch.float32)
                out = torch.matmul(alltoall_out, x2) + bias
            else:
                # 此时的out是matmul的输出，后续进行alltoall通信
                out = torch.matmul(alltoall_out, x2)

            # 7、存储当前rank的golden输出
            expt_out_list[i] = out

        return expt_out_list

    @skipIfUnsupportMultiNPU(2)
    @SupportedDevices(['Ascend910_95', 'Ascend950'])
    def test_npu_matmul_all_to_all(self):
        # 设备卡数
        world_size = 2
        # 初始化dtype，因为matmul_all_to_all的x1、x2和output的dtype一致，bias可以等于x1也可以为float32，这里统一使用bfloat16
        dtype = torch.bfloat16  # Simulate supported dtype
        # 初始化shape
        m, k, n = 16, 32, 32
        # 初始化format，取-1是因为后续要使用create_common_tensor生成张量，format为-1时，不会做特殊处理。
        data_format = -1
        # 初始化create_common_tensor需要的参数
        x1_shape = [dtype, data_format, [m, k]]
        x2_shape = [dtype, data_format, [k, n]]
        bias_shape = [dtype, data_format, [n]]
        # 生成tensors (one per NPU)
        x1_list = []
        x2_list = []
        bias_list = []
        for _ in range(world_size):
            # create_common_tensor接收四个参数：
            # 1、item：[dtype，format，shape[]]。2、minValue：int。3、maxValue：int。4、device=None
            # 两个返回值，cpu_input，npu_input。npu_input需要和device联合使用，npu_input=tensor.to(device)
            x1, _ = create_common_tensor(x1_shape, -1, 1)
            x1_list.append(x1)
            x2, _ = create_common_tensor(x2_shape, -1, 1)
            x2_list.append(x2)
            bias, _ = create_common_tensor(bias_shape, -1, 1)
            bias_list.append(bias)
        expt_out_list = self._construct_excepted_result(x1_list, x2_list, bias_list, world_size)
        # 启动多线程
        self._test_multiprocess(
            TestAlltoAllMatmul._test_npu_all_to_all_matmul,
            TestAlltoAllMatmul._init_dist_hccl,
            [expt_out_list, x1_list, x2_list, bias_list, world_size]
        )


if __name__ == '__main__':
    run_tests()

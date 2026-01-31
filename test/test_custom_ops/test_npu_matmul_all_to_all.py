import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class TestMatmulAlltoAll(TestCase):
    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '50000'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    
    @classmethod
    def _test_npu_matmul_all_to_all(cls, rank, input_list):
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
        out = torch_npu.npu_matmul_all_to_all(x1, x2, hcom_name, world_size, bias=bias, all2all_axes=[-1,-2])

        # Return result
        c2p.put((rank, out.cpu()))
        pg.barrier()
        
    def _test_multiprocess(self, f, init_pg, input_list): # f function, init_pg function, input_list list[]
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

            # 2、根据是否有bias，执行matmul
            if bias_list:
                bias = bias_list[i].npu().to(torch.float32)
                out = torch.matmul(x1, x2) + bias
            else:
                # 此时的out是matmul的输出，后续进行alltoall通信
                out = torch.matmul(x1, x2)

            # 3、准备进行alltoall通信前的shape重排
            # 需要抽一个rankSize卡数的维度放在tensor最外侧，让alltoall可以根据卡数做通信
            # 需要用到：view进行形状重排 + permute进行维度重排
            temp_shape = [out.shape[0], world_size, out.shape[1] // world_size]
            # perOut是matmul计算后，alltoall通信前，一切就绪的中间态张量，经过了前面shape的view和permute
            perOut = out.view(temp_shape).permute(1, 0, 2).contiguous()

            # 4、准备一个全0张量，准备进行alltoall通信
            # alltoall_out是alltoall通信后的结果，因为接口需要，这里生成一个全0的张量，shape为[rankSize, BS, H2/rankSize]
            per_shape = [temp_shape[1], temp_shape[0], temp_shape[2]]
            alltoall_out = torch.zeros(per_shape, dtype=torch.float32).npu()

            # 5、开始alltoall通信
            # dist.all_to_all_single是torch仓的alltoall通信接口，第一个参数为输出output，第二个参数为输入input
            dist.all_to_all_single(alltoall_out, perOut)
            # alltoall通信结束，alltoall_out已经是通信后的结果

            # 6、还原通信结果为MatmulAlltoAll算子的实际输出shape，[BS*rankSize,H2/rankSize]
            real_out_shape = [temp_shape[0] * temp_shape[1], temp_shape[2]]
            real_output = alltoall_out.view(real_out_shape)

            # 7、存储当前rank的golden输出
            expt_out_list[i] = real_output

        return expt_out_list

    @skipIfUnsupportMultiNPU(2)
    @SupportedDevices(['Ascend950'])
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
            TestMatmulAlltoAll._test_npu_matmul_all_to_all,
            TestMatmulAlltoAll._init_dist_hccl,
            [expt_out_list, x1_list, x2_list, bias_list, world_size]
        )


if __name__ == '__main__':
    run_tests()

# torch\_npu.npu\_matmul\_all\_to\_all

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>            |    √     |

## 功能说明

- 接口功能：完成Matmul计算、Permute(保证通信后地址连续)和AlltoAll通信的融合，**先计算后通信**。

- 计算公式：假设x1的shape为(BS, H1)，x2的shape为(H1, H2)，rankSize为NPU卡数。

  $$
  computeOut = x1 @ x2 + bias \\
  permutedOut = computeOut.view(BS, rankSize, H2/rankSize).permute(1, 0, 2) \\
  output = AlltoAll(permutedOut).view(rankSize*BS, H2/rankSize)
  $$

## 函数原型

```python
torch_npu.npu_matmul_all_to_all(x1, x2, hcom, world_size, bias=None, all2all_axes=None) -> Tensor
```

## 参数说明

- **x1**（`Tensor`）：必选输入，表示融合算子的左矩阵输入，也是Matmul计算的左矩阵，对应公式中的x1。数据类型支持bfloat16、float16，维度只能为2D，shape为(BS, H1)，数据格式支持ND，不支持非连续Tensor，支持第一维度为0的空Tensor。
- **x2**（`Tensor`）：必选输入，表示融合算子的右矩阵输入，也是Matmul计算的右矩阵，对应公式中的x2。数据类型与x1一致，维度只能为2D，shape为(H1, H2)，数据格式支持ND，支持转置非连续Tensor。
- **hcom**（`str`）：必选输入，Host侧标识列组的字符串，即通信域名称，通过get_hccl_comm_name接口获取。
- **world_size**（`int`）：必选输入，通信域内的rank总数，对应公式中的rankSize，支持范围[2, 4, 8, 16]。
- **bias**（`Tensor`）：可选输入，矩阵乘运算后累加的偏置，对应公式中的bias。数据类型由输入x1和x2决定，当x1和x2为float16时，bias的数据类型为float16；当x1和x2为bfloat16时，bias的数据类型为float32。维度只能为1D，shape为(H2)，数据类型支持ND。
- **all2all_axes**（`List[int]`）：可选输入，AlltoAll和Permute数据交换的方向，支持为空或者[-1, -2]，表示将Matmul结果由(BS, H2)转为(BS*rankSize, H2/rankSize)。

## 返回值说明

- **y**（`Tensor`）：计算输出，表示最终的计算结果output，数据类型与输入x1或者x2保持一致，支持2维，shape为(BS*rankSize, H2/rankSize)，数据格式支持ND，不支持非连续的Tensor。

## 约束说明

- 该接口支持训练、推理场景下使用。
- A3场景下，该接口支持单算子模式，不支持图模式。
- 除x1以外的输入参数均不支持空Tensor。
- 通信域名称hcom不支持传入空字符串，长度取值范围为[1, 127]。
- 输入参数Tensor中shape使用的变量说明：
    - BS：输入左矩阵的第一维度大小，表示输入序列sequence的条数，BS*rankSize取值范围为[0, 2147483647]。
    - H1：输入左矩阵的第二维度大小和输入右矩阵的第一维度大小，表示隐藏层维度，取值范围为[1, 65535]。
    - H2：输入右矩阵的第二维度大小，表示输出序列sequence的长度，取值范围为[2, 2147483647]，必须整除rankSize。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp

    def run_npu_matmul_all_to_all(rank, world_size, master_ip, master_port, x1_shape, x2_shape):
        torch_npu.npu.set_device(rank)
        init_method = 'tcp://' + master_ip + ':' + master_port
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > '2.0.1':
            hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcom_info = default_pg.get_hccl_comm_name(rank)
        x1_tensor = torch.randn(x1_shape, dtype=torch.float16).npu() 
        x2_tensor = torch.randn(x2_shape, dtype=torch.float16).npu()    
        output = torch_npu.npu_matmul_all_to_all(
            x1_tensor,
            x2_tensor,
            hcom_info,
            world_size
        )
        print("output: ", output)

    if __name__ == "__main__":
        worksize = 2 # npu卡数
        master_ip = '127.0.0.1' # 通信地址
        master_port = '50001' # 通信端口
        x1_shape = [1024, 256] # x1的输入shape
        x2_shape = [256, 3072] # x2的输入shape
        mp.spawn(
            run_npu_matmul_all_to_all,
            args=(worksize, master_ip, master_port, x1_shape, x2_shape),
            nprocs=worksize,
        )
    ```

# torch_npu.npu_mm_reduce_scatter_base

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>    | √  |

## 功能说明

-   API功能：TP切分场景下，实现matmul和reduce\_scatter的融合，融合算子内部实现计算和通信流水并行。

-   计算公式：
    $x1$代表输入`input`
    $$
    output = reducescatter(x1 \mathbin{@} x2 + bias)
    $$

>**说明：**<br> 
>使用该接口时，请确保驱动固件包和CANN包都为配套的8.0.RC2版本或者配套的更高版本，否则将会引发报错，比如BUS ERROR等。

## 函数原型

```
torch_npu.npu_mm_reduce_scatter_base(input, x2, hcom, world_size, *, reduce_op='sum', bias=None, comm_turn=0) -> Tensor
```

## 参数说明

- **input** (`Tensor`)：必选参数。数据类型支持`float16`、`bfloat16`，数据格式支持$ND$，输入shape支持2维，形如\(m, k\)。
- **x2** (`Tensor`)：必选参数。数据类型与`input`一致，数据格式支持$ND$，输入shape支持2维，形如\(k, n\)。轴满足matmul算子入参要求，k轴相等，且k轴取值范围为\[256, 65535\)，m轴需要整除`world_size`。
- **hcom** (`str`)：必选参数。通信域handle名，通过get\_hccl\_comm\_name接口获取。
- **world\_size** (`int`)：必选参数。通信域内的rank总数。
    -   <term>Atlas A2 训练系列产品</term>支持2、4、8卡，支持hccs链路all mesh组网（每张卡和其它卡两两相连）。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>支持2、4、8、16、32卡，支持hccs链路double ring组网（多张卡按顺序组成一个圈，每张卡只和左右卡相连）。

- <strong>*</strong>：必选参数，代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **reduce\_op** (`str`)：可选参数。reduce操作类型，当前仅支持'sum'，默认值为'sum'。
- **bias** (`Tensor`)：可选参数。数据类型支持`float16`、`bfloat16`，数据格式支持$ND$格式。数据类型需要和`input`保持一致。`bias`仅支持一维，且维度大小与output的第1维大小相同。当前版本暂不支持`bias`输入为非0的场景。
- **comm\_turn** (`int`)：可选参数。表示rank间通信切分粒度，默认值为0，表示默认的切分方式。当前版本仅支持输入0。

## 返回值说明

`Tensor`

数据类型和`input`保持一致，shape维度和`input`保持一致。

## 约束说明

-   该接口仅在训练场景下使用。
-   该接口支持图模式（PyTorch 2.1.0版本）。
-   `input`不支持输入转置后的tensor，`x2`转置后输入，需要满足shape的第一维大小与`input`的最后一维相同，满足matmul的计算条件。
-   <term>Atlas A2 训练系列产品</term>：一个模型中的通算融合算子（AllGatherMatmul、MatmulReduceScatter、MatmulAllReduce），仅支持相同通信域。

## 调用示例

-   单算子模式调用

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp
    def run_mm_reduce_scatter_base(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype):
        torch_npu.npu.set_device(rank)
        init_method = 'tcp://' + master_ip + ':' + master_port
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > '2.0.1':
            hcomm_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcomm_info = default_pg.get_hccl_comm_name(rank)
    
        input_ = torch.randn(x1_shape, dtype=dtype).npu()
        weight = torch.randn(x2_shape, dtype=dtype).npu()
        output = torch_npu.npu_mm_reduce_scatter_base(input_, weight, hcomm_info, world_size)
    
    if __name__ == "__main__":
        worksize = 8
        master_ip = '127.0.0.1'
        master_port = '50001'
        x1_shape = [128, 512]
        x2_shape = [512, 64]
        dtype = torch.float16
    
        mp.spawn(run_mm_reduce_scatter_base, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype), nprocs=worksize)
    ```

-   图模式调用

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp
    class MM_REDUCESCATTER_GRAPH_Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, input, weight, hcomm_info, world_size, reduce_op):
            output = torch_npu.npu_mm_reduce_scatter_base(input, weight, hcomm_info, world_size,
                                                          reduce_op=reduce_op)
            return output
    def define_model(model, graph_type):
        import torchair
        if graph_type == 1:  # 传统入图模式，静态shape+在线编译场景
            npu_backend = torchair.get_npu_backend(compiler_config=None)
            model = torch.compile(model, backend=npu_backend, dynamic=False)
        elif graph_type == 2:  # ACLNN入图模式，动态shape+二进制
            npu_backend = torchair.get_npu_backend(compiler_config=None)
            model = torch.compile(model, backend=npu_backend, dynamic=True)
        else:
            print("Error type")
        return model
    def get_graph(input, weight, hcomm_info, world_size):
        model = MM_REDUCESCATTER_GRAPH_Model()
        model = define_model(model, 2)
        model_output = model(input, weight, hcomm_info, world_size, reduce_op="sum")
        return model_output
    def run_mm_reduce_scatter_base(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype):
        torch_npu.npu.set_device(rank)
        init_method = 'tcp://' + master_ip + ':' + master_port
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > '2.0.1':
            hcomm_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcomm_info = default_pg.get_hccl_comm_name(rank)
        input = torch.randn(x1_shape, dtype=dtype).npu()
        weight = torch.randn(x2_shape, dtype=dtype).npu()
        output = get_graph(input, weight, hcomm_info, world_size)
        print("output:", output)
    if __name__ == "__main__":
        worksize = 8
        master_ip = '127.0.0.1'
        master_port = '50001'
        x1_shape = [128, 512]
        x2_shape = [512, 64]
        dtype = torch.float16
        mp.spawn(run_mm_reduce_scatter_base, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype), nprocs=worksize)
    ```


# torch\_npu.npu\_all\_gather\_base\_mm<a name="ZH-CN_TOPIC_0000001979420567"></a>

## 功能说明<a name="zh-cn_topic_0000001694916914_section14441124184110"></a>

TP切分场景下，实现allgather和matmul的融合，实现通信和计算流水并行。

>**说明：**<br>  
>使用该接口时，请确保驱动固件包和CANN包都为配套的8.0.RC2版本或者配套的更高版本，否则将会引发报错，比如BUS ERROR等。

## 函数原型<a name="zh-cn_topic_0000001694916914_section776431568"></a>

```
torch_npu.npu_all_gather_base_mm(Tensor input, Tensor x2, str hcom, int world_size, *, Tensor? bias=None, int gather_index=0, bool gather_output=True, int comm_turn=0) -> (Tensor, Tensor)
```

## 参数说明<a name="zh-cn_topic_0000001694916914_section112637109429"></a>

-   input：Tensor类型，数据类型支持float16、bfloat16，数据格式支持ND，输入shape支持2维，形如\(m, k\)、\(k, n\)，轴满足matmul算子入参要求，k轴相等，且k轴取值范围为\[256, 65535\)。
-   x2：Tensor类型，数据类型、输入shape维度需要和input保持一致，数据格式支持ND。
-   hcom：String类型，通信域handle名，通过get\_hccl\_comm\_name接口获取。
-   world\_size：int类型，通信域内的rank总数。
    -   <term>Atlas A2 训练系列产品</term>：支持2、4、8卡，支持hccs链路all mesh组网（每张卡和其它卡两两相连）。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持2、4、8、16、32卡，支持hccs链路double ring组网（多张卡按顺序组成一个圈，每张卡只和左右卡相连）。

-   \*：代表其之前的变量是位置相关，按照顺序输入，必选；之后的变量是键值对赋值的，位置无关，可选（不输入会使用默认值）。
-   bias：Tensor类型，可选参数，数据类型支持float16、bfloat16，数据格式支持ND格式。数据类型需要和input保持一致。bias仅支持一维，且维度大小与output的第1维大小相同。**当前版本暂不支持bias输入为非0的场景。**
-   gather\_index：int类型，表示gather操作对象，0：对input做gather，1：对x2做gather。默认值0。**当前版本仅支持输入0。**
-   gather\_output：bool类型，表示是否需要gather输出。默认值true。
-   comm\_turn：int类型，表示rank间通信切分粒度，默认值：0，表示默认的切分方式。**当前版本仅支持输入0。**

## 返回值说明<a name="zh-cn_topic_0000001694916914_section15236153161410"></a>
-   Tensor：第一个输出Tensor是allgather+matmul的结果。
-   Tensor：第二个输出Tensor是allgather的结果。

## 约束说明<a name="zh-cn_topic_0000001694916914_section19106152201519"></a>

-   该接口支持训练场景下使用。
-   该接口支持图模式（PyTorch 2.1版本）。
-   input不支持输入转置后的tensor，x2转置后输入，需要满足shape的第一维大小与x1的最后一维相同，满足matmul的计算条件。
-   <term>Atlas A2 训练系列产品</term>：一个模型中的通算融合算子（AllGatherMatmul、MatmulReduceScatter、MatmulAllReduce），仅支持相同通信域。

## 支持的型号<a name="zh-cn_topic_0000001694916914_section1414151813182"></a>

<term>Atlas A2 训练系列产品</term>

<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>

## 调用示例<a name="zh-cn_topic_0000001694916914_section14459801435"></a>

-   单算子模式调用

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp
    def run_all_gather_base_mm(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype):
        torch_npu.npu.set_device(rank)
        init_method = 'tcp://' + master_ip + ':' + master_port
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > '2.0.1':
            hcomm_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcomm_info = default_pg.get_hccl_comm_name(rank)
    
        tensor_allgather_shape = x1_shape
        single_shape = [x1_shape[0] // world_size, x1_shape[1]]
    
        input_ = torch.randn(single_shape, dtype=dtype).npu()
        weight = torch.randn(x2_shape, dtype=dtype).npu()
        output, gather_out = torch_npu.npu_all_gather_base_mm(input_, weight, hcomm_info, world_size)
    
    if __name__ == "__main__":
        worksize = 8
        master_ip = '127.0.0.1'
        master_port = '50001'
        x1_shape = [128, 512]
        x2_shape = [512, 64]
        dtype = torch.float16
    
        mp.spawn(run_all_gather_base_mm, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype), nprocs=worksize)
    ```

-   图模式调用

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp
    class ALLGATHER_MM_GRAPH_Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, input, weight, hcomm_info, world_size, gather_output):
            output, gather_output = torch_npu.npu_all_gather_base_mm(input, weight, hcomm_info, world_size,
                                                                     gather_output=gather_output)
            return output, gather_output
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
    def get_graph(input, weight, hcomm_info, world_size, gather_output):
        model = ALLGATHER_MM_GRAPH_Model()
        model = define_model(model, 2)
        model_output = model(input, weight, hcomm_info, world_size, gather_output=gather_output)
        output_npu = model_output[0]
        gather_output_npu = model_output[1]
        return output_npu, gather_output_npu
    def run_all_gather_base_mm(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype):
        torch_npu.npu.set_device(rank)
        init_method = 'tcp://' + master_ip + ':' + master_port
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > '2.0.1':
            hcomm_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcomm_info = default_pg.get_hccl_comm_name(rank)
        single_shape = [x1_shape[0] // world_size, x1_shape[1]]
        input = torch.randn(single_shape, dtype=dtype).npu()
        weight = torch.randn(x2_shape, dtype=dtype).npu()
        is_gather_out = True
        output, gather_out = get_graph(input, weight, hcomm_info, world_size, is_gather_out)
        print("output:", output)
    if __name__ == "__main__":
        worksize = 8
        master_ip = '127.0.0.1'
        master_port = '50001'
        x1_shape = [128, 512]
        x2_shape = [512, 64]
        dtype = torch.float16
        mp.spawn(run_all_gather_base_mm, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype), nprocs=worksize)
    ```


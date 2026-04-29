# torch_npu.npu_mm_reduce_scatter_base

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>    | √  |

## 功能说明

- API功能：TP切分场景下，实现matmul和reduce\_scatter的融合，融合算子内部实现计算和通信流水并行。支持如下两种量化方式：
    - perchannel 量化：针对权重（Weight）张量，沿输出特征维度（Output Channel）为每个通道维护独立的缩放因子与零点。
    - pertoken 量化：针对激活（Activation）张量，沿序列长度维度（Sequence Length / Token）为每个 Token 维护独立的量化参数。

- 计算公式：
    $x_1$代表输入`input`, 场景由 `comm_mode` 与 `x1` 数据类型共同决定：`comm_mode` 为 `ai_cpu`时，始终为基础场景；`comm_mode` 为 `aiv` 时，`x1` 为 `float16` 或 `bfloat16` 走基础场景，`x1` 为 `int8` 走量化场景。

    - 基础场景：
        $$
        output = reducescatter(x_1 \mathbin{@} x_2 + bias)
        $$
                                
    - 量化场景：
        $$
        output = reducescatter((x1\_scale * x2\_scale) * (x_1 \mathbin{@} x_2 + bias))
        $$
        量化场景scale参数使用说明：
        - 仅提供`x2_scale`时：
            $$
            output = reducescatter(x2\_scale * (x_1 \mathbin{@} x_2 + bias))
            $$
        - 同时提供`x1_scale`和`x2_scale`时：
            $$
            output = reducescatter((x1\_scale * x2\_scale) * (x_1 \mathbin{@} x_2 + bias))
            $$
        - 其中，`x1_scale`按\(m, 1\)广播，`x2_scale`按\(1, n\)广播。

> [!NOTE]   
> 使用该接口时，请确保驱动固件包和CANN包都为配套的8.0.RC2版本或者配套的更高版本，否则将会引发报错，比如BUS ERROR等。

## 函数原型

```python
torch_npu.npu_mm_reduce_scatter_base(input, x2, hcom, world_size, *, reduce_op='sum', bias=None, x1_scale=None, x2_scale=None, comm_turn=0, output_dtype=None, comm_mode=None) -> Tensor
```

## 参数说明

- **input** (`Tensor`)：必选参数。数据类型支持`float16`、`bfloat16`、`int8`，数据格式支持$ND$，输入shape支持2维，形如\(m, k\)。
- **x2** (`Tensor`)：必选参数。数据类型与`input`一致，数据格式支持$ND$、$NZ$。$NZ$仅在`comm_mode`为`aiv`时支持。输入shape支持2维，形如\(k, n\)。轴满足matmul算子入参要求，k轴相等，且k轴取值范围为\[256, 65535\)，m轴需要整除`world_size`。
- **hcom** (`str`)：必选参数。通信域handle名，通过get\_hccl\_comm\_name接口获取。
- **world\_size** (`int`)：必选参数。通信域内的rank总数。
    - <term>Atlas A2 训练系列产品</term>支持2、4、8卡，支持HCCS链路all mesh组网（每张卡和其它卡两两相连）。
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>支持2、4、8、16、32卡，支持HCCS链路double ring组网（多张卡按顺序组成一个圈，每张卡只和左右卡相连）。

- <strong>*</strong>：必选参数，代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **reduce\_op** (`str`)：可选参数。reduce操作类型，当前仅支持'sum'，默认值为'sum'。
- **bias** (`Tensor`)：可选参数。数据类型支持`float16`、`bfloat16`，数据格式支持$ND$格式。数据类型需要和`input`保持一致。`bias`仅支持一维，且维度大小与output的第1维大小相同。当前版本暂不支持`bias`输入为非0的场景。
- **x1\_scale** (`Tensor`)：可选参数。mm左矩阵反量化参数。数据类型支持`float32`，数据格式支持$ND$格式。数据维度为\(m, 1\), 支持pertoken量化。
- **x2\_scale** (`Tensor`)：可选参数。mm右矩阵反量化参数。数据类型支持`float32`、`int64`，数据格式支持$ND$格式。数据维度为\(1, n\), 支持perchannel量化。如需传入`int64`数据类型的，需要提前调用torch_npu.npu_trans_quant_param来获取`int64`数据类型的`x2_scale`。
- **comm\_turn** (`int`)：可选参数。表示rank间通信切分粒度，默认值为0，表示默认的切分方式。当前版本仅支持输入0。
- **output_dtype** (`ScalarType`)：可选参数。表示输出数据类型。仅支持在量化场景且`x1_scale`和`x2_scale`均为`float32`时，可指定输出数据类型为`bfloat16`或`float16`，默认值为`bfloat16`。
- **comm\_mode** (`str`)：可选参数。表示通信模式，支持`ai_cpu`、`aiv`两种模式。`ai_cpu`模式仅支持基础场景。`aiv`模式支持基础场景和量化场景。默认值为`ai_cpu`。

## 返回值说明

`Tensor`

输出shape为\(m // world_size, n\)。
基础场景时数据类型和`input`保持一致。
量化场景下，`x2_scale`为`int64`数据类型时，输出数据类型为`float16`。`x1_scale`和`x2_scale`均为`float32`时, 输出数据类型由`output_dtype`指定，默认为`bfloat16`。

## 约束说明

- `input`不支持输入转置后的tensor，`x2`转置后输入，需要满足shape的第一维大小与`input`的最后一维相同，满足matmul的计算条件。
- `world_size`必须等于实际通信域中的rank总数，且`input`的m轴必须能够被`world_size`整除。
- **通信模式约束**
  - `comm_mode`为`ai_cpu`时：
       - 该接口仅在训练场景下使用。
       - 该接口支持图模式。
       - <term>Atlas A2 训练系列产品</term>：一个模型中的通算融合算子（AllGatherMatmul、MatmulReduceScatter、MatmulAllReduce），仅支持相同通信域。
  - `comm_mode`为`aiv`时，训练和推理场景均可使用。  

- **comm_mode支持矩阵**

|   场景   | `comm_mode` |     `input`数据类型      |  是否支持 |
| -------- | --------- | -------------------- | -------- |
| 基础场景 | `ai_cpu` | `float16` / `bfloat16` | 支持 |
| 量化场景 | `ai_cpu` |       `int8`           | 不支持 |
| 基础场景 |  `aiv`  |  `float16` / `bfloat16` | 支持 |
| 量化场景 |  `aiv`  |         `int8`          | 支持 |

- **scale参数组合约束**

|  场景   |  `x1_scale`  | `x2_scale`  |     输出数据类型 |
| ------- | ---------- | --------- |   ----------- |
| 基础场景 |    不传    |  不传     |   与`input`一致 |
| 量化场景 |  不传      | `float32` |  由`output_dtype`指定，默认`bfloat16` |
| 量化场景 | `float32` | `float32` |  由`output_dtype`指定，默认`bfloat16` |
| 量化场景 | `float32` |  `int64`  |  由`output_dtype`指定，默认`bfloat16` |
| 量化场景 |  不传     |  `int64`  |   由`output_dtype`指定，默认`bfloat16` |

## 调用示例

- 单算子模式调用

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

- 图模式调用

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

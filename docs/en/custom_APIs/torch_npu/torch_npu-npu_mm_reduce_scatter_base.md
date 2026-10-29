# torch_npu.npu_mm_reduce_scatter_base

> [!NOTICE]  
> This API has changed in this version. For details about the changes, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->

## Function

- **API function**: In TP split scenarios, this API fuses the Matmul computation and the ReduceScatter communication, computing first and then communicating. The fused operator implements computation and communication pipeline parallelism internally.
    <!-- npu="A3,910b" id4 -->
    - The following two quantization modes are supported (quantization scenarios are supported only by <term>Atlas A2 products</term> and <term>Atlas A3 products</term>):
      - Fuses matrix multiplication (MatMul) and reduce_scatter collective communication operations in tensor parallelism (TP) scenarios. The fused kernel implements pipelined parallelism between computation and communication. The following quantization modes are supported:
      - `perchannel` quantization: For weight tensors, an independent scale factor and zero point are maintained for each channel along the output channel dimension (`Output Channel`).
    <!-- end id4 -->
- Calculation formulas:
    `x1` (that is, `input`) represents the left matrix of the Matmul computation, `x2` represents the right matrix of the Matmul computation, and `bias` represents the bias of the Matmul computation. The scenario is determined jointly by `comm_mode` and the data type of `input`:

    <!-- npu="A3,910b" id5 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: When `comm_mode` is `ai_cpu`, the basic scenario is always used. When `comm_mode` is `aiv` (in the `aiv` mode, the AI Vector core is used to complete communication tasks), the basic scenario is used when `input` is `torch.float16` or `torch.bfloat16`, and the quantization scenario is used when `input` is `torch.int8`.
    <!-- end id5 -->
    <!-- npu="950" id6 -->
    - <term>Ascend 950PR&950DT products</term>: Only the basic scenario is supported.
    <!-- end id6 -->

    Basic scenario:
    $$
    output = ReduceScatter(x_1 \mathbin{@} x_2 + bias)
    $$

    <!-- npu="A3,910b" id34 -->
    Quantization scenario (supported only by <term>Atlas A2 products</term> and <term>Atlas A3 products</term>):
    $$
    output = ReduceScatter((x1\_scale * x2\_scale) * (x_1 \mathbin{@} x_2 + bias))
    $$
    Usage of scale parameters in the quantization scenario:
    - When only `x2_scale` is provided:
        $$
        output = ReduceScatter(x2\_scale * (x_1 \mathbin{@} x_2 + bias))
        $$
    - When both `x1_scale` and `x2_scale` are provided:
        $$
        output = ReduceScatter((x1\_scale * x2\_scale) * (x_1 \mathbin{@} x_2 + bias))
        $$
    - `x1_scale` is broadcast to shape `[m, 1]`, and `x2_scale` is broadcast to shape `[1, n]`.

    <!-- end id34 -->

> [!NOTE]
> When using this API, ensure that the driver firmware package and CANN package are both the matching version `8.0.RC2` or later. Otherwise, errors such as `BUS ERROR` may occur.

## Prototype

```python
torch_npu.npu_mm_reduce_scatter_base(input, x2, hcom, world_size, *, reduce_op='sum', bias=None, x1_scale=None, x2_scale=None, comm_turn=0, output_dtype=None, comm_mode=None) -> Tensor
```

## Parameters

- **`input`** (`Tensor`): Required. Left matrix input for the Matmul computation. The data layout supports ND. The input shape supports 2 dimensions, in the form `(m, k)`.

    <!-- npu="A3,910b" id7 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: The data type can be `torch.float16`, `torch.bfloat16`, or `torch.int8`.
    <!-- end id7 -->
    <!-- npu="950" id8 -->
    - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.float16` or `torch.bfloat16`.
    <!-- end id8 -->

- **`x2`** (`Tensor`): Required. The data type must be identical to that of `input`. The data layout can be `ND` or `NZ`. `NZ` is supported only when `comm_mode` is set to `aiv`. This parameter must be 2D with shape `(k, n)`. The axes must satisfy the input requirements of the MatMul operator, with the `k` dimensions equal (that is, the last dimension of `input` must equal the first dimension of `x2`). The value of `k` must be in the range `[256, 65535)`, and the `m` dimension of `input` must be divisible by `world_size`.

    <!-- npu="A3,910b" id9 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: The data layout supports ND and NZ. NZ is supported only when `comm_mode` is `aiv`.
    <!-- end id9 -->
    <!-- npu="950" id10 -->
    - <term>Ascend 950PR&950DT products</term>: Only the ND data layout is supported.
    <!-- end id10 -->

- **`hcom`** (`str`): Required. Communicator handle name obtained by calling the `get_hccl_comm_name` API.
- **`world_size`** (`int`): Required. Total number of ranks within the communication domain.

    <!-- npu="910b" id11 -->
    - <term>Atlas A2 products</term>: Configurations of 2, 4, and 8 ranks are supported. All-mesh networking over HCCS links is supported, where each rank is connected to every other rank.
    <!-- end id11 -->
    <!-- npu="A3" id12 -->
    - <term>Atlas A3 products</term>: Configurations of 2, 4, 8, 16, and 32 ranks are supported. Double-ring networking over HCCS links is supported, where multiple ranks form a ring sequentially and each rank connects only to its adjacent left and right ranks.
    <!-- end id12 -->
    <!-- npu="950" id13 -->
    - <term>Ascend 950PR&950DT products</term>: 2, 4, 8, 16, 32, and 64 devices are supported. CCU supports only intra-node UB domain interconnection, while the AI CPU supports inter-node UB domain interconnection.
    <!-- end id13 -->

- **`*`**: Position delimiter used to distinguish positional arguments from keyword arguments. Variables before it are position-dependent and must be passed in order; variables after it are optional keyword arguments and can be passed in any order using key-value pairs. If not specified, their default values are used.
- **`reduce_op`** (`str`): Optional. Type of the reduce operation. Only the default value `'sum'` is supported.
- **`bias`** (`Tensor`): Optional. The data type can be `torch.float16` or `torch.bfloat16`. The data layout can be ND. The data type must be identical to that of `input`. This parameter must be a 1D tensor, where the size must be identical to that of the 1st dimension of `output`. In the current version, non-zero `bias` inputs are not supported.

    <!-- npu="A3,910b" id14 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: **The scenario where the `bias` input is non-zero is not supported currently.**
    <!-- end id14 -->
    <!-- npu="950" id15 -->
    - <term>Ascend 950PR&950DT products</term>: The scenario where the `bias` input is non-zero is supported.
    <!-- end id15 -->

- **`x1_scale`** (`Tensor`): Optional. Dequantization parameter for the left matrix of the MatMul operation. The data type can be `torch.float32`. The data layout can be ND. The shape of this parameter is `(m, 1)`. `pertoken` quantization is supported.

    <!-- npu="A3,910b" id16 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: The data type can be `torch.float32`.
    <!-- end id16 -->
    <!-- npu="950" id17 -->
    - <term>Ascend 950PR&950DT products</term>: **This parameter is not supported currently.**
    <!-- end id17 -->

- **`x2_scale`** (`Tensor`): Optional. Dequantization parameter for the right matrix of mm. The data dimension is `(1, n)`, and perchannel quantization is supported. The data layout supports ND.

    <!-- npu="A3,910b" id18 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: The data type can be `torch.float32` or `torch.int64`. To pass the `torch.int64` data type, call torch_npu.npu_trans_quant_param in advance to obtain `x2_scale` of the `torch.int64` data type.
    <!-- end id18 -->
    <!-- npu="950" id19 -->
    - <term>Ascend 950PR&950DT products</term>: **This parameter is not supported currently.**
    <!-- end id19 -->

- **`comm_turn`** (`int`): Optional. Communication splitting granularity between ranks. The default value is `0`, indicating the default splitting mode. Currently, only the value `0` is supported.
- **`output_dtype`** (`ScalarType`): Optional. Output data type. This parameter can be specified as `torch.bfloat16` or `torch.float16` only in quantization scenarios where both `x1_scale` and `x2_scale` are `torch.float32`. The default value is `torch.bfloat16`.

    <!-- npu="950" id20 -->
    - <term>Ascend 950PR&950DT products</term>: **This parameter is not supported currently.**
    <!-- end id20 -->

- **`comm_mode`** (`str`): Optional. Communication mode. Valid values are `ai_cpu` or `aiv`. The `ai_cpu` mode supports only the basic scenario. The `aiv` mode supports both the basic scenario and the quantization scenario. The default value is `ai_cpu`.

    <!-- npu="A3,910b" id21 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: The `ai_cpu` and `aiv` modes are supported. In the `aiv` mode, the AI Vector core is used to complete communication tasks. The `ai_cpu` mode supports only the basic scenario, and the `aiv` mode supports both the basic and quantization scenarios.
    <!-- end id21 -->
    <!-- npu="950" id22 -->
    - <term>Ascend 950PR&950DT products</term>: The `ai_cpu` and `ccu` modes are supported, and only the basic scenario is supported.
    <!-- end id22 -->

## Return Values

`Tensor`

The shape of the output tensor is `(m // world_size, n)`.

<!-- npu="A3,910b" id23 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: In the basic scenario, the data type is the same as that of `input`. In quantization scenarios, when `x2_scale` is `torch.int64`, the output data type is `torch.float16`; when both `x1_scale` and `x2_scale` are `torch.float32`, the output data type is specified by `output_dtype`, and the default is `torch.bfloat16`.
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950PR&950DT products</term>: Only the basic scenario is supported. The data type is the same as that of `input`.
<!-- end id24 -->

## Constraints

- This API supports single-operator mode and TorchAir graph mode.
- `input` does not support transposed input. If `x2` is transposed, the size of its first dimension must match the last dimension of `input`, satisfying the requirements of the MatMul operation.
- `world_size` must equal the total number of ranks in the actual communication domain, and the `m` dimension of `input` must be divisible by `world_size`.
- **Communication Mode Constraints**

  <!-- npu="A3,910b" id25 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: `comm_mode` supports the `ai_cpu` and `aiv` modes. The `ai_cpu` mode is used only in training scenarios, and the `aiv` mode can be used in both training and inference scenarios.
  <!-- end id25 -->
  <!-- npu="950" id26 -->
  - <term>Ascend 950PR&950DT products</term>: `comm_mode` supports the `ai_cpu` and `ccu` modes, and both modes are used only in training scenarios. The total data volume of the ReduceScatter collective communication must not exceed 16*256MB. The total collective communication data volume is calculated as m \* n \* sizeof(output_dtype). Due to different shapes, the internal implementation of the operator may differ, and the actually supported total communication volume may be slightly smaller than this value.
  <!-- end id26 -->

- **`comm_mode` support matrix**:

  <!-- npu="A3,910b" id27 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - Scenario: Basic scenario
    - `comm_mode`: `ai_cpu`
    - `input` data type: `torch.float16` / `torch.bfloat16`
    - Supported: Yes
  <!-- end id27 -->
  <!-- npu="A3,910b" id28 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - Scenario: Quantization scenario
    - `comm_mode`: `ai_cpu`
    - `input` data type: `torch.int8`
    - Supported: No
  <!-- end id28 -->
  <!-- npu="A3,910b" id29 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - Scenario: Basic scenario
    - `comm_mode`: `aiv`
    - `input` data type: `torch.float16` / `torch.bfloat16`
    - Supported: Yes
  <!-- end id29 -->
  <!-- npu="A3,910b" id30 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - Scenario: Quantization scenario
    - `comm_mode`: `aiv`
    - `input` data type: `torch.int8`
    - Supported: Yes
  <!-- end id30 -->
  <!-- npu="950" id31 -->
  - <term>Ascend 950PR&950DT products</term>:
    - Scenario: Basic scenario
    - `comm_mode`: `ai_cpu` / `ccu`
    - `input` data type: `torch.float16` / `torch.bfloat16`
    - Supported: Yes
  <!-- end id31 -->
  <!-- npu="950" id32 -->
  - <term>Ascend 950PR&950DT products</term>:
    - Scenario: Quantization scenario
    - `comm_mode`: -
    - `input` data type: -
    - Supported: No (only the basic scenario is supported)
  <!-- end id32 -->
<!-- npu="A3,910b" id33 -->
- **scale parameter combination constraints** (quantization scenarios, supported only by <term>Atlas A2 products</term> and <term>Atlas A3 products</term>)

    |  Scenario   |  `x1_scale`  | `x2_scale`  |     Output Data Type |
    | ------- | ---------- | --------- |   ----------- |
    | Basic scenario |    Not passed    |  Not passed     |   Same as `input` |
    | Quantization scenario |  Not passed      | `torch.float32` |  Specified by `output_dtype`, default `torch.bfloat16` |
    | Quantization scenario | `torch.float32` | `torch.float32` |  Specified by `output_dtype`, default `torch.bfloat16` |
    | Quantization scenario | `torch.float32` |  `torch.int64`  |  Specified by `output_dtype`, default `torch.bfloat16` |
    | Quantization scenario |  Not passed     |  `torch.int64`  |   Specified by `output_dtype`, default `torch.bfloat16` |
<!-- end id33 -->

## Examples

- Single-operator call

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

- Graph mode call

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
        if graph_type == 1:  # Traditional graph input mode, static shape + online compilation
            npu_backend = torchair.get_npu_backend(compiler_config=None)
            model = torch.compile(model, backend=npu_backend, dynamic=False)
        elif graph_type == 2:  # ACLNN graph input mode, dynamic shape + binary file.
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

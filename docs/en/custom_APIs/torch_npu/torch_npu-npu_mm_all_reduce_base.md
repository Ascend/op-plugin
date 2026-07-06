# torch_npu.npu_mm_all_reduce_base

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 training products/Atlas A2 inference products</term>   | √  |

## Function

- Description: Fuses matrix multiplication (MatMul) and all_reduce collective communication operations in tensor parallelism (TP) scenarios. The fused kernel implements pipelined parallelism between computation and communication.

- Formula:
    $$
    output = allreduce\left(x1 \mathbin{@} \left((x2 + antiquantOffset) * antiquantScale\right) + bias + x3\right)
    $$

> [!NOTE]  
> When using this API, ensure that the driver firmware package and CANN package are both version 8.0.RC2 or a matching later version. Otherwise, errors such as a BUS ERROR will be raised.

## Prototype

```python
torch_npu.npu_mm_all_reduce_base(x1, x2, hcom, *, reduce_op='sum', bias=None, antiquant_scale=None, antiquant_offset=None, x3=None, dequant_scale=None, pertoken_scale=None, comm_quant_scale_1=None, comm_quant_scale_2=None, comm_turn=0, antiquant_group_size=0) -> Tensor
```

## Parameters

- **`x1`** (`Tensor`): Required. The data type can be `int8`, `float16`, or `bfloat16`. The data layout can be ND. This parameter must be a 2D or 3D tensor.
- **`x2`** (`Tensor`): Required. The data type can be `float16`, `int8`, or `bfloat16`. The data layout can be NZ (Ascend-optimized layout) or ND. In non-quantization scenarios, the data type must be identical to that of `x1`. The size of the 0th dimension must be identical to that of the last dimension of `x1`.
- **`hcom`** (`str`): Required. Communicator handle name obtained by calling the `get_hccl_comm_name` API.
- **`*`**: Required. Positional argument separator. Arguments before this symbol are positional-only and must be passed in sequence. Arguments after this symbol are keyword-only, position-independent options that require key-value assignments (default values are used if no value is assigned).
- **`reduce_op`** (`str`): Optional. Type of the reduce operation. Currently, only the default value `sum` is supported.
- **`bias`** (`Tensor`): Optional. The data type can be `int32`, `float16`, or `bfloat16`. The data layout can be ND. This parameter must be a 1D tensor, where the size must match the last dimension of `output` or `x2`.
- **`antiquant_scale`** (`Tensor`): Optional. Dequantization scale for `x2` in fake-quantization scenarios, $antiquantScale$ in the formula. The data type can be `float16` or `bfloat16`. The data layout can be ND. In fake-quantization scenarios, the data type must be identical to that of `x1`.
    - `pertensor` scenario: The shape is `[1]`.
    - `perchannel` scenario: The shape is `[1, n]` or `[n]`, where `n` indicates the size of the last dimension of `x2`.
    - `pergroup` scenario: The shape is `[ceil(k, antiquant_group_size), n]`, where `k` indicates the size of the first dimension of `x2`, `n` indicates the size of the last dimension of `x2`, and `antiquant_group_size` is the group size input for the dequantization computation of the input `x2`.

        > [!NOTE]  
        > The calculation logic for $ceil(k, antiquant\_group\_size)$ is $(k + antiquant\_group\_size - 1)/antiquant\_group\_size$, where only the integer part of the result is taken.

- **`antiquant_offset`** (`Tensor`): Optional. Dequantization offset for `x2` in fake-quantization scenarios, $antiquantOffset$ in the formula. The data type can be `float16` or `bfloat16`. The data layout can be ND. The data type and shape must be identical to those of `antiquant_scale`.
- **`x3`** (`Tensor`): Optional. Offset added after MatMul computation. The data type can be `float16` or `bfloat16`. The data layout can be ND. The data type and shape must be identical to those of `output`.
  
- **`dequant_scale`** (`Tensor`): Optional. Dequantization scale applied after MatMul computation. The data type can be `int64`, `uint64`, `bfloat16`, or `float32`. The data layout can be ND.
    - `pertensor` scenario: The shape is `[1]`.
    - `perchannel` scenario: The shape is `[n]` or `[1, n]`, where `n` indicates the size of the last dimension of `x2`.

- **`pertoken_scale`** (`Tensor`): Optional. Per-token dequantization scale applied after MatMul computation. The data type can be `float32`. When `x1` is `[m, k]`, the shape is `[m]`. When `x1` is `[b, s, k]`, the shape is `[b * s]`.
  
- **`comm_quant_scale_1`** (`Tensor`): Optional. Quantization and dequantization scale before and after AlltoAll communication. The data type can be `float16` or `bfloat16`. The data layout can be ND. When `x2` is `[k, n]`, the shape is `[1, n]` or `[n]`. Ensure that the data on each rank is consistent and correct.
- **`comm_quant_scale_2`** (`Tensor`): Optional. Quantization and dequantization scale before and after AllGather communication. The data type can be `float16` or `bfloat16`. The data layout can be ND. When `x2` is `[k, n]`, the shape is `[1, n]` or `[n]`. Ensure that the data on each rank is consistent and correct.
- **`comm_turn`** (`int`): Optional. Communication splitting granularity between ranks. The default value is `0`, indicating the default splitting mode. Currently, only the value `0` is supported.
- **`antiquant_group_size`** (`int`): Optional. Group size for dequantizing `x2` in `pergroup` fake-quantization. It describes the size of the data block to be dequantized along the $k$ axis corresponding to a single set of dequantization parameters. When the fake-quantization mode is not `pergroup`, the value of this parameter must be `0`. When the fake-quantization mode is `pergroup`, the value of this parameter must be a multiple of 32 and must be in the range [32, min(k-1, INT_MAX)], where `k` indicates the size of the first dimension of `x2`. The default value is `0`, indicating a non-`pergroup` scenario.

## Return Values

`Tensor`

In non-quantization and fake-quantization scenarios, the data type is identical to that of `x1`. In full quantization scenarios, the output data type can be `float16` or `bfloat16`. The size of the 0th dimension is identical to that of the 0th dimension of `x1`. When `x1` is 2D, the size of the 1st dimension is identical to that of the 1st dimension of `x2`. When `x1` is 3D, the size of the 1st dimension is identical to that of the 1st dimension of `x1`, and the size of the 2nd dimension is identical to that of the 1st dimension of `x2`.

## Constraints

- This API can be used in inference scenarios.
- This fused operator is disabled in incremental quantization scenarios and enabled in full quantization scenarios.
- This API supports graph mode.
- The input `x1` can be 2D or 3D with shape `(m, k)` or `(b, s, k)`, and `x2` must be 2D with shape `(k, n)`. The $k$ dimension must match and meet the requirements of the mm operator. The parameter `bias` supports only 1D tensors, where the size must be identical to that of the last dimension of `output`. The shape of `x3` must be identical to that of `output`.
- `x1` does not support transposed input. If `x2` is transposed, the size of its first dimension must match the last dimension of `x1`, satisfying the requirements of the MatMul operation.
- The value range of $k$ in `antiquant_group_size` is identical to that in the MatMul operation. `INT_MAX` must be greater than $(k-1)$.
- Atlas A2 training products/Atlas A2 inference products:
    - The data type can be `bfloat16`.
    - Empty tensors are not supported for `x1` and `x2`.
    - Configurations of 1, 2, 4, and 8 ranks are supported. Only all-mesh networking over HCCS links is supported.
    - In non-quantization scenarios, the value ranges of $m$, $k$, and $n$ are all [1, 2147483647].
    - The shapes of `comm_quant_scale_1` and `comm_quant_scale_2` must be identical, and their `dtype` must be identical to the output `dtype`. These parameters are supported only in full quantization scenarios.

- Full quantization scenarios: The value range of $m$ is [1, 2147483647], and the size of the last dimension for `x1` and `x2` is [1, 65535]. Therefore, the value range of $k$ is [1, 65535]. The value of $n$ can be greater than 65535 only when `x2` with shape `[n, k]` is transposed.
- Fake-quantization scenarios: The value range of $m$ is [1, 2147483647], and the value range of $k$ and $n$ is [1, 65535].
- Atlas A2 training products: Communication fusion operators (AllGatherMatmul, MatmulReduceScatter, and MatmulAllReduce) within a single model must share the identical communication domain.
- In long-sequence scenarios, memory insufficiency or calculation timeout errors can occur as the size of $b/s$ or $m$ increases.
- The data type varies in different scenarios.

    **Table 1** Non-quantization scenarios

    |Product|x1|x2|bias|x3|output|antiquant_scale|antiquant_offset|dequant_scale|
    |--------|--------|--------|--------|--------|--------|--------|--------|--------|
    |Atlas A2 training products/Atlas A2 inference products|`float16`|`float16`|`float16`|`float16`|`float16`|None|None|None|
    |Atlas A2 training products/Atlas A2 inference products|`bfloat16`|`bfloat16`|`bfloat16`|`bfloat16`|`bfloat16`|None|None|None|
    
    **Table 2** Fake-quantization scenarios

    |Product|x1|x2|bias|x3|output|antiquant_scale|antiquant_offset|dequant_scale|
    |--------|--------|--------|--------|--------|--------|--------|--------|--------|
    |Atlas A2 training products/Atlas A2 inference products|`float16`|`int8`|`float16`|`float16`|`float16`|`float16`|`float16`|None|
    |Atlas A2 training products/Atlas A2 inference products|`bfloat16`|`int8`|`bfloat16`|`bfloat16`|`bfloat16`|`bfloat16`|`bfloat16`|None|
    
    **Table 3** Full-quantization scenarios

    |Product|x1|x2|bias|x3|output|antiquant_scale|antiquant_offset|dequant_scale|pertoken_scale|
    |--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
    |Atlas A2 training products/Atlas A2 inference products|`int8`|`int8`|`int32`|`float16`|`float16`|None|None|`uint64` or `int64`|None|
    |Atlas A2 training products/Atlas A2 inference products|`int8`|`int8`|`int32`|`bfloat16`|`bfloat16`|None|None|`bfloat16`|None|
    |Atlas A2 training products/Atlas A2 inference products|`int8`|`int8`|`int32`|`float16`|`float16`|None|None|`float32`|`float32`|
    |Atlas A2 training products/Atlas A2 inference products|`int8`|`int8`|`int32`|`bfloat16`|`bfloat16`|None|None|`bfloat16`|`float32`|

    > [!NOTE]  
    > In full quantization scenarios, if `dequant_scale` is provided as a `float32` value, convert it into `int64` by using the `torch_npu.npu_trans_quant_param` API before calling `torch_npu.npu_mm_all_reduce_base`. For details about the conversion method, see the corresponding API documentation.

## Examples

- Single-operator call

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp
    
    def run_mm_all_reduce_base(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype):
        torch_npu.npu.set_device(rank)
        init_method = "tcp://" + master_ip + ":" + master_port
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
    
        default_pg = _get_default_group()
        if torch.__version__ > "2.0.1":
            hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcom_info = default_pg.get_hccl_comm_name(rank)
    
        input_ = torch.randn(x1_shape, dtype=dtype).npu()
        weight = torch.randn(x2_shape, dtype=dtype).npu()
        output = torch_npu.npu_mm_all_reduce_base(input_, weight, hcom_info, reduce_op="sum")
        print("output: ", output)
    
    if __name__ == "__main__":
        worksize = 8
        master_ip = "127.0.0.1"
        master_port = "50001"
        x1_shape = [128, 512]
        x2_shape = [512, 64]
        dtype = torch.float16
    
        mp.spawn(
            run_mm_all_reduce_base,
            args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype),
            nprocs=worksize)
    
    # Expected output of the preceding code sample:
    output:  tensor([[ 60.7500,  -0.8770, -32.7812,  ...,   6.9219,  45.1250,  -1.4062],
            [-32.4688,  -5.7734, -19.7500,  ...,   6.2227, -63.9688, -42.1250],
            [-10.0781,  70.0000, -40.5938,  ...,  16.0000,  28.5312,  34.9688],
            ...,
            [  6.4844,  33.2500, -12.0781,  ..., -57.5312, -37.0000, -14.3203],
            [ -9.2422, -41.1562,   4.7188,  ...,   6.2812, -12.9531, -64.6250],
            [-25.3750,  13.9141,   9.8281,  ..., -21.7188,  64.5625, -56.1562]],
        device='npu:1', dtype=torch.float16)
    output:  tensor([[ 60.7500,  -0.8770, -32.7812,  ...,   6.9219,  45.1250,  -1.4062],
            [-32.4688,  -5.7734, -19.7500,  ...,   6.2227, -63.9688, -42.1250],
            [-10.0781,  70.0000, -40.5938,  ...,  16.0000,  28.5312,  34.9688],
            ...,
            [  6.4844,  33.2500, -12.0781,  ..., -57.5312, -37.0000, -14.3203],
            [ -9.2422, -41.1562,   4.7188,  ...,   6.2812, -12.9531, -64.6250],
            [-25.3750,  13.9141,   9.8281,  ..., -21.7188,  64.5625, -56.1562]],
        device='npu:0', dtype=torch.float16)
    ```
    
- Graph mode call

     Call examples for the NZ data layout in non-quantization, fake-quantization, and full quantization scenarios:

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import numpy as np
    
    class MM_ALLREDUCE_GRAPH_Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
    
        def forward(
            self,
            x1,
            x2,
            hcom,
            reduce_op,
            bias,
            antiquant_scale,
            antiquant_offset,
            x3,
            dequant_scale,
        ):
            output_npu = torch_npu.npu_mm_all_reduce_base(
                x1=x1,
                x2=x2,
                hcom=hcom,
                reduce_op=reduce_op,
                bias=bias,
                antiquant_scale=antiquant_scale,
                antiquant_offset=antiquant_offset,
                x3=x3,
                dequant_scale=dequant_scale,
            )
            return output_npu
    
    class MM_ALLREDUCE_A8W8_GRAPH_Model(MM_ALLREDUCE_GRAPH_Model):
        def __init__(self):
            super().__init__()
    
        def forward(
            self,
            x1,
            x2,
            hcom,
            reduce_op,
            bias,
            antiquant_scale,
            antiquant_offset,
            x3,
            dequant_scale,
        ):
            output_npu = torch_npu.npu_mm_all_reduce_base(
                x1=x1,
                x2=x2.t(),
                hcom=hcom,
                reduce_op=reduce_op,
                bias=bias,
                antiquant_scale=antiquant_scale,
                antiquant_offset=antiquant_offset,
                x3=x3,
                dequant_scale=dequant_scale,
            )
            return output_npu
    
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
    
    def get_graph(
        input,
        weight,
        hcomm_info,
        dequant_scale,
        bias,
        antiquant_scale,
        antiquant_offset,
        x3,
    ):
        model = MM_ALLREDUCE_A8W8_GRAPH_Model()
        model = define_model(model, 2)  # 1: Static graph input; 2: Dynamic graph input
        output = model(
            x1=input,
            x2=weight,
            hcom=hcomm_info,
            reduce_op="sum",
            bias=bias,
            antiquant_scale=antiquant_scale,
            antiquant_offset=antiquant_offset,
            x3=x3,
            dequant_scale=dequant_scale,
        )
        return output

    def run_mc2_a16w16(x1_shape, x2_shape, hcom_info):
        np_input = np.random.uniform(float(-3), float(3), size=x1_shape).astype(np.float16)
        np_weight = np.random.uniform(float(-3), float(3), size=x2_shape).astype(np.float16)
        input = torch.tensor(np_input).npu()
        weight = torch.tensor(np_weight).npu()
        output_a16w16 = get_graph(input, weight, hcom_info, None, None, None, None, None)
        return output_a16w16

    def run_mc2_a8w8(x1_shape, x2_shape, hcom_info):
        np_input = np.random.uniform(float(-3), float(3), size=x1_shape).astype(np.int8)
        np_weight = np.random.uniform(float(-3), float(3), size=x2_shape).astype(np.int8)
        input = torch.tensor(np_input).npu()
        weight = torch.tensor(np_weight).npu()
        weight_nz = torch_npu.npu_format_cast(weight.contiguous(), 29)
        dequant_scale = (
            torch.randn(x2_shape[0], dtype=torch.float32)
            .uniform_(float(-10), float(10))
            .npu()
        )
        dequant_scale = torch_npu.npu_trans_quant_param(dequant_scale)
        output_a8w8 = get_graph(
            input, weight_nz, hcom_info, dequant_scale, None, None, None, None
        )
        return output_a8w8

    def run_mc2_a16w8(x1_shape, x2_shape, hcom_info):
        np_input = np.random.uniform(float(-3), float(3), size=x1_shape).astype(np.float16)
        np_weight = np.random.uniform(float(-3), float(3), size=x2_shape).astype(np.int8)
        input = torch.tensor(np_input).npu()
        weight = torch.tensor(np_weight).npu()
        weight_nz = torch_npu.npu_format_cast(weight.contiguous(), 29)
        antiquant_scale = (
            torch.randn(x2_shape[0], dtype=torch.float16)
            .uniform_(float(-1), float(1))
            .npu()
        )
        antiquant_offset = torch.ones(x2_shape[0], dtype=torch.float16).npu()
        output_a16w8 = get_graph(
            input, weight_nz, hcom_info, None, None, antiquant_scale, antiquant_offset, None
        )
        return output_a16w8

    def run_mm_all_reduce_base(
        rank, world_size, master_ip, master_port, x1_shape, x2_shape, op_type
    ):
        torch_npu.npu.set_device(rank)
        init_method = "tcp://" + master_ip + ":" + master_port
        dist.init_process_group(
            backend="hccl", rank=rank, world_size=world_size, init_method=init_method
        )
        from torch.distributed.distributed_c10d import _get_default_group
    
        default_pg = _get_default_group()
        if torch.__version__ > "2.0.1":
            hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(
                rank
            )
        else:
            hcom_info = default_pg.get_hccl_comm_name(rank)
        output = None
        # Non-quantization call
        if op_type == "a16w16":
            output = run_mc2_a16w16(x1_shape, x2_shape, hcom_info)
        # Fake-quantization call
        if op_type == "a16w8":
            output = run_mc2_a16w8(x1_shape, x2_shape, hcom_info)
        # Full-quantization call
        if op_type == "a8w8":
            output = run_mc2_a8w8(x1_shape, x2_shape, hcom_info)
        print("output:", output)
    
    if __name__ == "__main__":
        worksize = 2
        master_ip = "127.0.0.1"
        master_port = "50001"
        x1_shape = [1280, 5120]
        x2_shape = [640, 5120]
        op_type = "a16w8"  # Options: a16w16, a16w8, a8w8
        mp.spawn(
            run_mm_all_reduce_base,
            args=(worksize, master_ip, master_port, x1_shape, x2_shape, op_type),
            nprocs=worksize)
        
    # Expected output of the preceding code sample:
    output: tensor([[-3.6594e+01, -8.4219e+00, -7.3688e+01,  ..., -4.9531e+01,
            -4.4438e+01, -1.2300e+02],
            [ 1.3225e+02,  2.4175e+02, -1.6094e+01,  ...,  1.4062e+02,
            -1.5750e+01,  4.0375e+01],
            [ 1.1931e+02,  9.7000e+01, -1.4200e+02,  ..., -1.2912e+02,
            -3.6062e+01,  7.0750e+01],
            ...,
            [-1.1031e+02, -6.4750e+01, -1.6500e+01,  ...,  2.3675e+02,
            9.6750e+01, -1.2662e+02],
            [-1.2569e+02,  2.3288e+02,  6.6250e+01,  ...,  3.0812e+01,
            6.2500e-02, -2.0550e+02],
            [ 7.4062e+01, -6.0100e+02, -3.0750e+02,  ..., -2.1500e+02,
            -2.4450e+02,  3.2400e+02]], device='npu:1', dtype=torch.float16)
    output: tensor([[-3.6594e+01, -8.4219e+00, -7.3688e+01,  ..., -4.9531e+01,
            -4.4438e+01, -1.2300e+02],
            [ 1.3225e+02,  2.4175e+02, -1.6094e+01,  ...,  1.4062e+02,
            -1.5750e+01,  4.0375e+01],
            [ 1.1931e+02,  9.7000e+01, -1.4200e+02,  ..., -1.2912e+02,
            -3.6062e+01,  7.0750e+01],
            ...,
            [-1.1031e+02, -6.4750e+01, -1.6500e+01,  ...,  2.3675e+02,
            9.6750e+01, -1.2662e+02],
            [-1.2569e+02,  2.3288e+02,  6.6250e+01,  ...,  3.0812e+01,
            6.2500e-02, -2.0550e+02],
            [ 7.4062e+01, -6.0100e+02, -3.0750e+02,  ..., -2.1500e+02,
            -2.4450e+02,  3.2400e+02]], device='npu:0', dtype=torch.float16)
    ```

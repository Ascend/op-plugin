# torch_npu.npu_quant_scatter_

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id2 -->
<!-- npu="310p" id3 -->
- <term>Atlas inference products</term>: Supported
<!-- end id3 -->

## Function

First quantize `updates`, then update the values in `self` with the values in `updates` according to the specified `axis` and `indices`. The data in `self` is changed.

## Prototype

```python
torch_npu.npu_quant_scatter_(self, indices, updates, quant_scales, quant_zero_points=None, axis=-2, quant_axis=-1, reduce='update', dst_type=None, round_mode='rint') -> Tensor
```

## Parameters

- **`self`** (`Tensor`): Required input. Source data tensor. The data layout can be $ND$, non-contiguous tensors are supported, and the number of dimensions can only be 3 to 8.

    <!-- npu="310p" id4 -->
    - <term>Atlas inference products</term>: The data type can be `torch.int8`.
    <!-- end id4 -->
    <!-- npu="910b" id5 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.int8`.
    <!-- end id5 -->
    <!-- npu="950" id6 -->
    - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.int8`, `torch_npu.hifloat8`, `torch.float8_e5m2`, or `torch.float8_e4m3fn`.
    <!-- end id6 -->

- **`indices`** (`Tensor`): Required. Index tensor. The data type can be `torch.int32`. The data layout can be ND. Non-contiguous tensors are supported.
- **`updates`** (`Tensor`): Required. Update data tensor. The data layout can be ND. Non-contiguous tensors are supported.

    <!-- npu="310p" id7 -->
    - <term>Atlas inference products</term>: The data type can be `torch.float16`.
    <!-- end id7 -->
    <!-- npu="910b" id8 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.bfloat16` or `torch.float16`.
    <!-- end id8 -->
    <!-- npu="950" id9 -->
    - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.bfloat16` or `torch.float16`.
    <!-- end id9 -->

- **`quant_scales`** (`Tensor`): Required. Quantization scale tensor. The data layout can be ND. Non-contiguous tensors are supported.

    <!-- npu="310p" id10 -->
    - <term>Atlas inference products</term>: The data type can be `torch.float32`.
    <!-- end id10 -->
    <!-- npu="910b" id11 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.bfloat16` or `torch.float32`.
    <!-- end id11 -->
    <!-- npu="950" id12 -->
    - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.bfloat16` or `torch.float32`.
    <!-- end id12 -->

- **`quant_zero_points`** (`Tensor`): Optional. Quantization offset tensor. The data layout can be ND. Non-contiguous tensors are supported.

    <!-- npu="310p" id13 -->
    - <term>Atlas inference products</term>: The data type can be `torch.int32`.
    <!-- end id13 -->
    <!-- npu="910b" id14 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.bfloat16` or `torch.int32`.
    <!-- end id14 -->
    <!-- npu="950" id15 -->
    - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.bfloat16` or `torch.int32`.
    <!-- end id15 -->

- **`axis`** (`int`): Optional. Axis on `updates` used for updating. The default value is `-2`.
- **`quant_axis`** (`int`): Optional. Axis on `updates` used for quantization. The default value is `-1`.
- **`reduce`** (`str`): Optional. Data operation mode. The default value is `'update'`.
- **`dst_type`** (`int`): Optional. Data type of the output `self`. The default value is `None`, indicating `torch.int8`.

    <!-- npu="910b,310p" id16 -->
    - <term>Atlas inference products</term> and <term>Atlas A2 products</term>: this parameter is not supported currently, and `torch.int8` is used by default.
    <!-- end id16 -->
    <!-- npu="950" id17 -->
    - <term>Ascend 950PR&950DT products</term>: the supported values are `torch.int8`, `torch_npu.hifloat8`, `torch.float8_e5m2`, and `torch.float8_e4m3fn`.
    <!-- end id17 -->

- **`round_mode`** (`str`): Optional. Rounding mode used during quantization. The default value is `"rint"`.

    <!-- npu="910b,310p" id18 -->
    - <term>Atlas inference products</term> and <term>Atlas A2 products</term>: this parameter is not supported currently, and `"rint"` is used by default.
    <!-- end id18 -->
    <!-- npu="950" id19 -->
    - <term>Ascend 950PR&950DT products</term>: when `dst_type` is `torch.int8`, `torch.float8_e5m2`, or `torch.float8_e4m3fn`, `"rint"` is supported; when `dst_type` is `torch_npu.hifloat8`, `"round"` and `"hybrid"` are supported.
    <!-- end id19 -->

## Return Values

`Tensor`

Represents the result of updating `self`.

## Constraints

- This API supports single-operator mode and TorchAir graph mode.

- `indices` can be 1D or 2D. If it is 2D, the size of its 2nd dimension must be 2. Out-of-range indices are not supported. The user must ensure that the indices are valid; the framework does not perform bounds checking. The data segments of `self` mapped by `indices` must not overlap. If they overlap, the results of multiple executions differ due to multi-core concurrency.
- The number of dimensions of `updates` must be the same as that of `self`. The size of its 1st dimension is equal to the size of the 1st dimension of `indices` and is not greater than the size of the 1st dimension of `self`. The size of its `axis` dimension is not greater than the size of the `axis` dimension of `self`. The sizes of the other dimensions must be equal to those of the corresponding dimensions of `self`. The size of its last dimension must be 32B-aligned.
- The number of elements of `quant_scales` must be equal to the size of `updates` on the `quant_axis` axis.
- The number of elements of `quant_zero_points` must be equal to the size of `updates` on the `quant_axis` axis.
- `axis` cannot be the 1st or the last dimension of `updates`.
- `quant_axis` can only be the last dimension of `updates`.
<!-- npu="950,910b" id20 -->
- The data type input combinations of `updates`, `quant_scales`, and `quant_zero_points` include (applicable to the following products):
    - <term>Atlas A2 products</term>
    - <term>Ascend 950PR&950DT products</term>

    | updates | quant_scales | quant_zero_points |
    | --- | --- | --- |
    | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` |
    | `torch.float16` | `torch.float32` | `torch.int32` |
<!-- end id20 -->

## Example

- Single-operator call

   ```python
   >>> import torch
   >>> import torch_npu
   >>> import numpy as np
   >>>
   >>>
   >>> data_var = np.random.uniform(0, 1, [24, 4096, 128]).astype(np.int8)
   >>> var = torch.from_numpy(data_var).to(torch.int8).npu()
   >>>
   >>> data_indices = np.random.uniform(0, 1, [24]).astype(np.int32)
   >>> indices = torch.from_numpy(data_indices).to(torch.int32).npu()
   >>>
   >>> data_updates = np.random.uniform(1, 2, [24, 1, 128]).astype(np.float16)
   >>> updates = torch.from_numpy(data_updates).to(torch.bfloat16).npu()
   >>>
   >>> data_quant_scales = np.random.uniform(0, 1, [1, 1, 128]).astype(np.float16)
   >>> quant_scales = torch.from_numpy(data_quant_scales).to(torch.bfloat16).npu()
   >>>
   >>> data_quant_zero_points = np.random.uniform(0, 1, [1, 1, 128]).astype(np.float16)
   >>> quant_zero_points = torch.from_numpy(data_quant_zero_points).to(torch.bfloat16).npu()
   >>>
   >>> axis = -2
   >>> quant_axis = -1
   >>> reduce = "update"
   >>>
   >>> out=torch_npu.npu_quant_scatter_(var, indices, updates, quant_scales, quant_zero_points, axis=axis, quant_axis=quant_axis, reduce=reduce)
   >>> print(out.shape)
   torch.Size([24, 4096, 128])
   >>> print(out.dtype)
   torch.int8
   >>> print(out)
   tensor([[[2, 3, 2,  ..., 3, 6, 1],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            ...,
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0]],

         [[1, 2, 2,  ..., 4, 5, 1],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            ...,
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0]],

         [[2, 4, 3,  ..., 3, 6, 2],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            ...,
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0]],

         ...,

         [[2, 3, 4,  ..., 3, 7, 2],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            ...,
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0]],

         [[2, 3, 2,  ..., 4, 6, 1],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            ...,
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0]],

         [[1, 3, 3,  ..., 5, 5, 1],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            ...,
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0]]], device='npu:0', dtype=torch.int8)
   ```

- Graph mode call

    ```python
    # Graph mode approach
    import torch
    import torch_npu
    import math
    import torchair as tng
    import numpy as np
    from torchair.configs.compiler_config import CompilerConfig
    import torch._dynamo
    TORCHDYNAMO_VERBOSE=1
    TORCH_LOGS="+dynamo"
    
    # Print macro that supports graph mode
    import logging
    from torchair.core.utils import logger
    logger.setLevel(logging.DEBUG)
    config = CompilerConfig()
    config.debug.graph_dump.type = "pbtxt"
    npu_backend = tng.get_npu_backend(compiler_config=config)
    from torch.library import Library, impl
    
    # Data generation
    dtype_list2 =["fp16","int8","int32","fp32","fp16"]
    dtype_list =[np.float16,np.int8,np.int32,np.float32]
    updates_shape =[1,11,1,32]
    var_shape =[1,11,1,32]
    indices_shape =[1,2]
    quant_scales_shape =[1,1,1,32]
    quant_zero_points_shape =[1,1,1,32]
    
    axis =-2
    quant_axis=-1
    reduce = "update"

    updates_data = np.random.uniform(-1,1,size=updates_shape)
    var_data = np.random.uniform(1,2,size=var_shape).astype(dtype_list[1])
    quant_scales_data = np.random.uniform(1,2,size=quant_scales_shape)
    indices_data = np.random.uniform(0,1,size=indices_shape).astype(dtype_list[2])
    quant_zero_points_data = np.random.uniform(0,1,size=quant_zero_points_shape)

    updates_npu = torch.from_numpy(updates_data).npu().to(torch.bfloat16).npu()
    var_npu = torch.from_numpy(var_data).npu()
    quant_scales_npu = torch.from_numpy(quant_scales_data).npu().to(torch.bfloat16).npu()
    quant_zero_points_npu = torch.from_numpy(quant_zero_points_data).to(torch.bfloat16).npu()
    indices_npu = torch.from_numpy(indices_data).npu()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self):
            return torch_npu.npu_quant_scatter_(var_npu, indices_npu, updates_npu, quant_scales_npu, quant_zero_points_npu, axis=axis, quant_axis=quant_axis, reduce=reduce)

    def MetaInfershape():
        with torch.no_grad():
            model = Model()
            model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
            graph_output = model()
        single_op = torch_npu.npu_quant_scatter_(var_npu, indices_npu, updates_npu, quant_scales_npu, quant_zero_points_npu, axis=axis, quant_axis=quant_axis, reduce=reduce)
        print("single op output with mask:", single_op[0], single_op[0].shape)
        print("graph output with mask:", graph_output[0], graph_output[0].shape)

    if __name__ == "__main__":
        MetaInfershape()
    
    # Expected output of the preceding code sample:
    single op output with mask: tensor([[[ 0,  0,  1,  1,  1,  0,  1,  0,  1,  1,  0,  0,  0,  1,  0,  1,  0,
               1,  1,  1,  0,  0,  0,  0,  0,  1,  1,  1,  0,  1,  1,  1]],
            [[ 0,  0,  1,  0,  1,  0,  0,  1,  0,  0,  1,  0,  0,  1,  1,  1,  0,
               1,  1,  0,  1,  1,  0,  0, -1,  0,  1,  0,  1,  0,  1,  0]],
            [[ 0,  1,  1,  1,  1,  1,  1,  1,  0,  1,  0,  0,  1,  1,  1,  1,  0,
               1,  0,  1,  0,  1,  1,  0,  0,  0,  0,  0,  1,  1,  1,  1]],
            [[ 0,  0,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,
               1,  0,  1,  1,  1,  1,  1,  1,  1,  0,  0,  1,  0,  0,  1]],
            [[ 0,  0,  1,  1,  1,  0,  1,  1,  0,  0,  0,  0,  1,  1,  1,  2,  0,
               1,  1,  0,  1,  1,  1,  1, -1,  0,  0,  0,  1,  1,  1,  0]],
            [[ 0,  1,  1,  0,  1,  0,  0,  1,  0,  1,  0,  1,  1,  0,  1,  1,  0,
               1,  1,  1,  0,  0,  1,  0, -1,  0,  0,  0,  1,  1,  1,  0]],
            [[ 0, -1,  1,  1,  1,  0,  0,  1,  1,  0,  0,  1,  0,  1,  2,  1,  0,
               1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  0,  0,  0]],
            [[ 1,  0,  0,  1,  1,  0,  1,  0,  0,  1,  0,  0,  0,  2,  0,  1,  0,
               1,  1,  1,  0,  1,  0,  0,  1,  0,  0,  0,  1,  1,  1,  1]],
            [[ 0,  0,  1,  0,  1,  1,  0,  1,  0,  1,  0,  0,  1,  2,  1,  1,  0,
               0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  1,  0,  1,  0]],
            [[ 0,  0,  0,  1,  1,  0,  0,  1,  0,  0,  0, -1,  0,  1,  1,  0,  1,
               1,  1,  1,  1,  1,  0,  0,  0,  1,  0,  0,  1,  1,  1,  0]],
            [[ 0,  1,  0,  0,  1,  0,  1,  0,  0,  1,  1,  0,  1,  1,  1,  1,  0,
               1,  1,  0,  1,  1,  0,  0,  0,  0,  0,  0,  1,  0,  1,  1]]],
           device='npu:0', dtype=torch.int8) torch.Size([11, 1, 32])
    graph output with mask: tensor([[[ 0,  0,  1,  1,  1,  0,  1,  0,  1,  1,  0,  0,  0,  1,  0,  1,  0,
               1,  1,  1,  0,  0,  0,  0,  0,  1,  1,  1,  0,  1,  1,  1]],
            [[ 0,  0,  1,  0,  1,  0,  0,  1,  0,  0,  1,  0,  0,  1,  1,  1,  0,
               1,  1,  0,  1,  1,  0,  0, -1,  0,  1,  0,  1,  0,  1,  0]],
            [[ 0,  1,  1,  1,  1,  1,  1,  1,  0,  1,  0,  0,  1,  1,  1,  1,  0,
               1,  0,  1,  0,  1,  1,  0,  0,  0,  0,  0,  1,  1,  1,  1]],
            [[ 0,  0,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,
               1,  0,  1,  1,  1,  1,  1,  1,  1,  0,  0,  1,  0,  0,  1]],
            [[ 0,  0,  1,  1,  1,  0,  1,  1,  0,  0,  0,  0,  1,  1,  1,  2,  0,
               1,  1,  0,  1,  1,  1,  1, -1,  0,  0,  0,  1,  1,  1,  0]],
            [[ 0,  1,  1,  0,  1,  0,  0,  1,  0,  1,  0,  1,  1,  0,  1,  1,  0,
               1,  1,  1,  0,  0,  1,  0, -1,  0,  0,  0,  1,  1,  1,  0]],
            [[ 0, -1,  1,  1,  1,  0,  0,  1,  1,  0,  0,  1,  0,  1,  2,  1,  0,
               1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  0,  0,  0]],
            [[ 1,  0,  0,  1,  1,  0,  1,  0,  0,  1,  0,  0,  0,  2,  0,  1,  0,
               1,  1,  1,  0,  1,  0,  0,  1,  0,  0,  0,  1,  1,  1,  1]],
            [[ 0,  0,  1,  0,  1,  1,  0,  1,  0,  1,  0,  0,  1,  2,  1,  1,  0,
               0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  1,  0,  1,  0]],
            [[ 0,  0,  0,  1,  1,  0,  0,  1,  0,  0,  0, -1,  0,  1,  1,  0,  1,
               1,  1,  1,  1,  1,  0,  0,  0,  1,  0,  0,  1,  1,  1,  0]],
            [[ 0,  1,  0,  0,  1,  0,  1,  0,  0,  1,  1,  0,  1,  1,  1,  1,  0,
               1,  1,  0,  1,  1,  0,  0,  0,  0,  0,  0,  1,  0,  1,  1]]],
           device='npu:0', dtype=torch.int8) torch.Size([11, 1, 32])
    ```

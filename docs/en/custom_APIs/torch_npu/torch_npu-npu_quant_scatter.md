# torch_npu.npu_quant_scatter

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 training products/Atlas A2 inference products</term>| √   |
|<term>Atlas inference products</term>| √   |

## Function

Description: Quantizes `updates`, and then updates the values in `input` using the values in `updates` according to the specified `axis` and `indices`. The `input` remains unchanged.

## Prototype

```python
torch_npu.npu_quant_scatter(input, indices, updates, quant_scales, quant_zero_points=None, axis=0, quant_axis=1, reduce='update') -> Tensor
```

## Parameters

- **`input`** (`Tensor`): Required. Source data tensor. Non-contiguous tensors are supported. The data layout can be ND. The shape must have 3 to 8 dimensions.
    - Atlas inference products, Atlas A2 training products, and Atlas 800I A2 inference products: The data type can be `int8`.
- **`indices`** (`Tensor`): Required. Index tensor. The data type can be `int32`. The data layout can be ND. Non-contiguous tensors are supported.
- **`updates`** (`Tensor`): Required. Update data tensor. The data layout can be ND. Non-contiguous tensors are supported.
    - Atlas inference products: The data type can be `float16`.
    - Atlas A2 training products/Atlas A2 inference products: The data type can be `bfloat16` or `float16`.

- **`quant_scales`** (`Tensor`): Required. Quantization scale tensor. The data layout can be ND. Non-contiguous tensors are supported.
    - Atlas inference products: The data type can be `float32`.
    - Atlas A2 training products/Atlas A2 inference products: The data type can be `bfloat16` or `float32`.

- **`quant_zero_points`** (`Tensor`): Optional. Quantization offset tensor. The data layout can be ND. Non-contiguous tensors are supported.
    - Atlas inference products: The data type can be `int32`.
    - Atlas A2 training products/Atlas A2 inference products: The data type can be `bfloat16` or `int32`.

- **`axis`** (`int`): Optional. Axis on `updates` used for updating. The default value is `0`.
- **`quant_axis`** (`int`): Optional. Axis on `updates` used for quantization. The default value is `1`.
- **`reduce`** (`str`): Optional. Data operation mode. The default value is `'update'`.

## Return Values

`Tensor`

Output tensor representing the results after `input` is updated.

## Constraints

- This API supports graph mode.

- `indices` must be a 1D or 2D tensor. If `indices` is a 2D tensor, the size of its second dimension must be `2`. Out-of-bounds indices are not supported. Ensure that all indices are valid, and the framework does not perform bounds checking. The `input` data segments referenced by `indices` must not overlap. Otherwise, execution results may vary across runs because of multi-core concurrent execution.
- `updates` must have the same number of dimensions as `input`. The size of its first dimension must be equal to the size of the first dimension of `indices` and must not exceed the size of the first dimension of `input`. The size of its `axis` dimension must not exceed the size of the corresponding `axis` dimension of `input`. The sizes of all other dimensions must match those of the corresponding dimensions of `input`. The size of its last dimension must be 32-byte aligned.
- The number of elements in `quant_scales` must be equal to the size of the `quant_axis` dimension of `updates`.
- The number of elements in `quant_zero_points` must be equal to the size of the `quant_axis` dimension of `updates`.
- `axis` must not be the first or last dimension of `updates`.
- `quant_axis` must be equal to the last dimension of `updates`.

## Examples

- Single-operator call

   ```python
   >>> import torch
   >>> import torch_npu
   >>> import numpy as np
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
   >>> out = torch_npu.npu_quant_scatter(var, indices, updates, quant_scales, quant_zero_points, axis=axis, quant_axis=quant_axis, reduce=reduce)
   >>> print(out.shape)
   torch.Size([24, 4096, 128])
   >>> print(out.dtype)
   torch.int8
   >>> print(out)
   tensor([[[2, 2, 2,  ..., 4, 9, 2],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            ...,
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0]],

         [[2, 2, 2,  ..., 5, 6, 2],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            ...,
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0]],

         [[3, 2, 3,  ..., 6, 8, 2],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            ...,
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0]],

         ...,

         [[2, 2, 2,  ..., 4, 8, 2],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            ...,
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0]],

         [[2, 2, 2,  ..., 4, 8, 2],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            ...,
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0]],

         [[2, 2, 3,  ..., 4, 9, 2],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            ...,
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0]]], device='npu:0', dtype=torch.int8)
   ```

- Graph mode call

    ```python
    # Configure graph capture
    import torch
    import torch_npu
    import math
    import torchair as tng
    import numpy as np
    from torchair.configs.compiler_config import CompilerConfig
    import torch._dynamo
    TORCHDYNAMO_VERBOSE=1
    TORCH_LOGS="+dynamo"
    
    # Configure logging and debug settings for graph capture
    import logging
    from torchair.core.utils import logger
    logger.setLevel(logging.DEBUG)
    config = CompilerConfig()
    config.debug.graph_dump.type = "pbtxt"
    npu_backend = tng.get_npu_backend(compiler_config=config)
    from torch.library import Library, impl
    
    # Generate data
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
            return torch_npu.npu_quant_scatter(var_npu, indices_npu, updates_npu, quant_scales_npu, quant_zero_points_npu, axis=axis, quant_axis=quant_axis, reduce=reduce)

    def MetaInfershape():
        with torch.no_grad():
            model = Model()
            model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
            graph_output = model()
        single_op = torch_npu.npu_quant_scatter(var_npu, indices_npu, updates_npu, quant_scales_npu, quant_zero_points_npu, axis=axis, quant_axis=quant_axis, reduce=reduce)
        print("single op output with mask:", single_op[0], single_op[0].shape)
        print("graph output with mask:", graph_output[0], graph_output[0].shape)

    if __name__ == "__main__":
        MetaInfershape()
    
    # Expected output of the preceding code sample:
    single op output with mask: tensor([[[ 1,  1,  0,  1,  0, -1,  0,  0,  0,  1,  0,  1,  0, -1,  1,  0,  0,
               0,  0,  0,  0,  0,  1,  0,  1,  0,  1,  1,  2,  1,  0,  0]],
            [[ 1,  0,  0,  1,  0,  0,  1,  1,  1,  1,  0,  0,  0,  1,  1,  0,  1,
               1,  1,  1,  1,  1,  0,  1,  0,  0,  1,  1,  0,  1,  0,  0]],
            [[ 1,  0,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0, -1,  1,  1,  1,  1,
               0,  1,  0,  2,  0,  0,  0,  1,  0,  1,  1,  2,  0,  1,  1]],
            [[ 1,  1,  0,  1,  0, -1,  0,  1,  0,  1,  0,  0, -1,  0,  1,  0,  0,
               1,  0,  2,  2,  0,  0,  1,  0,  1,  0,  0,  1,  0,  1,  0]],
            [[ 1,  1,  0,  1,  1,  1,  0,  1,  1,  0,  1,  0,  1,  1,  1,  1,  1,
               0,  0,  1,  2,  0,  1,  1,  0,  0,  1,  0,  1,  0,  1,  1]],
            [[ 0,  1,  0,  1,  0,  1,  1,  0,  0,  1,  1,  0,  0,  0,  1,  1,  0,
               0,  1,  1,  0, -1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  0]],
            [[ 0,  0,  0,  1,  0,  0,  0,  1,  1,  1,  0,  1,  0, -1,  1,  0,  0,
               1,  1,  0,  1,  0,  1,  0,  1,  0,  1,  1,  1,  0,  1,  1]],
            [[ 1,  1,  1,  0,  0,  0,  0,  1,  0,  1,  1,  1,  0,  1,  1,  1,  1,
               0,  0,  1,  1,  0,  0,  1,  0,  0,  0,  1,  1,  0,  1,  1]],
            [[ 1,  1,  0,  0,  1,  0,  0,  1,  0,  1,  1,  1,  0,  0,  1, -1,  0,
               1,  1,  0,  0,  1,  0,  1,  1,  0,  0,  1,  0,  1,  1,  1]],
            [[ 1,  0,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  1,  1,  1,  0,  1,
               0,  1,  1,  1, -1,  0,  1,  0,  0,  0,  1,  1,  1,  0,  0]],
            [[ 1,  0, -1,  1,  0,  0,  1,  0,  1,  2,  0,  1,  0, -1,  1,  1,  1,
               1,  0,  0,  2,  1,  0,  1,  1,  0,  1,  0,  1,  0,  1,  0]]],
           device='npu:0', dtype=torch.int8) torch.Size([11, 1, 32])           
    graph output with mask: tensor([[[ 1,  1,  0,  1,  0, -1,  0,  0,  0,  1,  0,  1,  0, -1,  1,  0,  0,
               0,  0,  0,  0,  0,  1,  0,  1,  0,  1,  1,  2,  1,  0,  0]],
            [[ 1,  0,  0,  1,  0,  0,  1,  1,  1,  1,  0,  0,  0,  1,  1,  0,  1,
               1,  1,  1,  1,  1,  0,  1,  0,  0,  1,  1,  0,  1,  0,  0]],
            [[ 1,  0,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0, -1,  1,  1,  1,  1,
               0,  1,  0,  2,  0,  0,  0,  1,  0,  1,  1,  2,  0,  1,  1]],
            [[ 1,  1,  0,  1,  0, -1,  0,  1,  0,  1,  0,  0, -1,  0,  1,  0,  0,
               1,  0,  2,  2,  0,  0,  1,  0,  1,  0,  0,  1,  0,  1,  0]],
            [[ 1,  1,  0,  1,  1,  1,  0,  1,  1,  0,  1,  0,  1,  1,  1,  1,  1,
               0,  0,  1,  2,  0,  1,  1,  0,  0,  1,  0,  1,  0,  1,  1]],
            [[ 0,  1,  0,  1,  0,  1,  1,  0,  0,  1,  1,  0,  0,  0,  1,  1,  0,
               0,  1,  1,  0, -1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  0]],
            [[ 0,  0,  0,  1,  0,  0,  0,  1,  1,  1,  0,  1,  0, -1,  1,  0,  0,
               1,  1,  0,  1,  0,  1,  0,  1,  0,  1,  1,  1,  0,  1,  1]],
            [[ 1,  1,  1,  0,  0,  0,  0,  1,  0,  1,  1,  1,  0,  1,  1,  1,  1,
               0,  0,  1,  1,  0,  0,  1,  0,  0,  0,  1,  1,  0,  1,  1]],
            [[ 1,  1,  0,  0,  1,  0,  0,  1,  0,  1,  1,  1,  0,  0,  1, -1,  0,
               1,  1,  0,  0,  1,  0,  1,  1,  0,  0,  1,  0,  1,  1,  1]],
            [[ 1,  0,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  1,  1,  1,  0,  1,
               0,  1,  1,  1, -1,  0,  1,  0,  0,  0,  1,  1,  1,  0,  0]],
            [[ 1,  0, -1,  1,  0,  0,  1,  0,  1,  2,  0,  1,  0, -1,  1,  1,  1,
               1,  0,  0,  2,  1,  0,  1,  1,  0,  1,  0,  1,  0,  1,  0]]],
           device='npu:0', dtype=torch.int8) torch.Size([11, 1, 32])
    ```

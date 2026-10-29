# torch_npu.npu_convert_weight_to_int4pack

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

Packs an input Tensor of the `torch.int32` type for storage as `torch_npu.int4`. Every eight `torch_npu.int4` values are carried by one `torch.int32` value, and they are stored in an interleaved manner.

<!-- npu="950" id11 -->
<term>Ascend 950PR&950DT products</term>: In addition to the preceding capability, these products also support packing an input Tensor of the `torch.float32` type for storage as `torch_npu.float4_e2m1fn_x2`. Every eight `torch_npu.float4_e2m1fn_x2` values are carried by one `torch.float32` value, and they are stored in an interleaved manner.
<!-- end id11 -->

## Prototype

```python
torch_npu.npu_convert_weight_to_int4pack(weight, inner_k_tiles=0) -> Tensor
```

## Parameters

- **`weight`** (`Tensor`): Required. Input weight. The data layout can be ND or FRACTAL_NZ. The data type can be `torch.int32`. Non-contiguous tensors are not supported. This parameter must be 2D with shape `(k, n)` or `(n, k)`. The size of the last dimension must be a multiple of 8. Individual element values must be within the `torch_npu.int4` representation range of [-8, 7].

  <!-- npu="A3,910b" id4 -->
  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: the data type can be `torch.int32`, and the element values in `weight` must be within the representation range of `torch_npu.int4`, that is, $[-8, 7]$. Two dimensions are supported, the shape can be $(k, n)$ or $(n, k)$, and the last dimension must be 8-element aligned.
  <!-- end id4 -->
  <!-- npu="950" id5 -->
  - <term>Ascend 950PR&950DT products</term>: the data type can be `torch.int32` or `torch.float32`. When the input data type is `torch.int32`, the element values in `weight` must be within the representation range of `torch_npu.int4`, that is, $[-8, 7]$. When the input data type is `torch.float32`, the element values in `weight` must be within the representation range of `torch_npu.float4_e2m1fn_x2`, that is, $[-6.0, 6.0]$. Two or three dimensions are supported, the shape can be $(k, n)$, $(n, k)$, $(g, k, n)$, or $(g, n, k)$, and the last dimension must be 8-element aligned.

    > [!NOTE]
    >
    > - PyTorch 2.7 and earlier: within the `torch_npu.float4_e2m1fn_x2` range, the `float4_e2m1fn` type provided by the `ml_dtypes` library in Python is used. The `ml_dtypes` version must be 0.5.0 or later. For details, see [Example](#example).
    > - Starting from PyTorch 2.8, `torch.float4_e2m1fn_x2` can be used directly.
  <!-- end id5 -->

- **`inner_k_tiles`** (`int`): Optional. Number of K-tiles that are packed together in the internal packing format. The default value is `0`. **Reserved parameter, currently not used.**

## Return Values

**`out`** (`Tensor`): packed output tensor. The shape is $(k, n/8)$, $(n, k/8)$, or $(g, k, n/8)$, and the data layout is the same as that of `weight`.

<!-- npu="A3,910b" id6 -->
- <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: the data type can be `torch.int32`.
<!-- end id6 -->
<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT products</term>: the data type can be `torch.int32` (corresponding to `weight` of `torch.int32`) or `torch.float32` (corresponding to `weight` of `torch.float32`).
<!-- end id7 -->

## Constraints

- This API can be used in inference scenarios.
- This API supports single-operator mode and TorchAir graph mode.

## Example

- Single-operator call

  - Scenario where the input is `torch.int32`

    ```python
    import torch
    import torch_npu

    m = 128
    k = 64
    n = 128
    trans_weight = False

    cpu_x = torch.randn((m, k), dtype=torch.float16)
    if trans_weight:
        cpu_weight = torch.randint(low=-8, high=8, size=(n, k), dtype=torch.int32)
        cpu_antiquantscale = torch.randn((n, 1), dtype=torch.float16)
        cpu_antiquantoffset = torch.randn((n, 1), dtype=torch.float16)
    else:
        cpu_weight = torch.randint(low=-8, high=8, size=(k, n), dtype=torch.int32)
        cpu_antiquantscale = torch.randn((1, n), dtype=torch.float16)
        cpu_antiquantoffset = torch.randn((1, n), dtype=torch.float16)

    weight_int4 = torch_npu.npu_convert_weight_to_int4pack(cpu_weight.npu())

    if trans_weight:
        cpu_weight = cpu_weight.transpose(-1, -2)
        weight_int4 = weight_int4.transpose(-1, -2)
        cpu_antiquantscale = cpu_antiquantscale.transpose(-1, -2)
        cpu_antiquantoffset = cpu_antiquantoffset.transpose(-1, -2)

    npu_out = torch_npu.npu_weight_quant_batchmatmul(cpu_x.npu(), weight_int4, cpu_antiquantscale.npu(), cpu_antiquantoffset.npu())
    ```

  <!-- npu="950" id8 -->
  - Scenario where the input is `torch.float32`: This example is supported only on <term>Ascend 950PR&950DT products</term>.

    ```python
    import torch
    import torch_npu
    import numpy as np
    # The ml_dtypes version must be 0.5.0 or later.
    from ml_dtypes import float4_e2m1fn

    m = 128
    k = 64
    n = 128
    E2M1_MIN, E2M1_MAX = -6, 6
    antiquant_group_size = 32
    trans_weight = False

    cpu_x = torch.randn((m, k), dtype=torch.float16)
    if trans_weight:
        cpu_weight = (E2M1_MIN + (E2M1_MAX - E2M1_MIN) * np.random.random(k * n).reshape((n, k))).astype(float4_e2m1fn)
        cpu_weight = torch.from_numpy(cpu_weight.astype(np.float32))
        cpu_antiquantscale = torch.randint(127 - 5, 127 + 5, (n, k // antiquant_group_size), dtype=torch.uint8) # Use uint8 to carry scale data
    else:
        cpu_weight = (E2M1_MIN + (E2M1_MAX - E2M1_MIN) * np.random.random(k * n).reshape((k, n))).astype(float4_e2m1fn)
        cpu_weight = torch.from_numpy(cpu_weight.astype(np.float32))
        cpu_antiquantscale = torch.randint(127 - 5, 127 + 5, (k // antiquant_group_size, n), dtype=torch.uint8) # Use uint8 to carry scale data

    weight_packed = torch_npu.npu_convert_weight_to_int4pack(cpu_weight.npu())

    if trans_weight:
        weight_packed = weight_packed.transpose(-1, -2)
        cpu_antiquantscale = cpu_antiquantscale.transpose(-1, -2)

    npu_out = torch_npu.npu_weight_quant_batchmatmul(cpu_x.npu(), weight_packed, cpu_antiquantscale.npu(), None, None, None, None, antiquant_group_size)
    ```
  <!-- end id8 -->

- Graph mode call

  - $ND$ format, input type `torch.int32`

    ```python
    import torch
    import torch_npu
    import torchair
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)

    m = 16
    k = 64
    n = 128
    trans_weight = False
    is_weight_nz = False

    cpu_x = torch.randn((m, k),dtype=torch.float16)
    if trans_weight:
        cpu_weight = torch.randint(low=-8, high=8, size=(n, k), dtype=torch.int32)
        cpu_antiquantscale = torch.ones((n, 1),dtype=torch.float16)
        cpu_antiquantoffset = torch.zeros((n, 1),dtype=torch.float16)
    else:
        cpu_weight = torch.randint(low=-8, high=8, size=(k, n), dtype=torch.int32)
        cpu_antiquantscale = torch.ones((1, n),dtype=torch.float16)
        cpu_antiquantoffset = torch.zeros((1, n),dtype=torch.float16)

    npu_weight = cpu_weight.npu()

    # Determine whether to convert the ND format to the NZ format
    if is_weight_nz:
        npu_weight = torch_npu.npu_format_cast(npu_weight, 29)
    # int32 to int4pack
    weight_int4pack = torch_npu.npu_convert_weight_to_int4pack(npu_weight)

    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, weight, antiquant_scale, antiquant_offset, quant_scale,quant_offset, bias, antiquant_group_size):
            if trans_weight:
                weight  = weight.transpose(-1, -2)
                antiquant_scale = antiquant_scale.transpose(-1, -2)
                antiquant_offset = antiquant_offset.transpose(-1, -2)
            return torch_npu.npu_weight_quant_batchmatmul(x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias, antiquant_group_size)

    cpu_model = MyModel()
    model = cpu_model.npu()
    model = torch.compile(cpu_model, backend=npu_backend, dynamic=True, fullgraph=True)

    npu_out = model(cpu_x.npu(), weight_int4pack, cpu_antiquantscale.npu(), cpu_antiquantoffset.npu(), None, None, None, 0)
    ```

  - $FRACTAL\_NZ$ format, input type `torch.int32`

    ```python
    import torch
    import torch_npu
    import torchair
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)

    m = 16
    k = 64
    n = 64
    trans_weight = False
    is_weight_nz = True
    cpu_x = torch.randn((m, k),dtype=torch.float16)

    if trans_weight:
        cpu_weight = torch.randint(low=-8, high=8, size=(n, k), dtype=torch.int32)
        cpu_antiquantscale = torch.ones((n, 1),dtype=torch.float16)
        cpu_antiquantoffset = torch.zeros((n, 1),dtype=torch.float16)
    else:
        cpu_weight = torch.randint(low=-8, high=8, size=(k, n), dtype=torch.int32)
        cpu_antiquantscale = torch.ones((1, n),dtype=torch.float16)
        cpu_antiquantoffset = torch.zeros((1, n),dtype=torch.float16)
    npu_weight = cpu_weight.npu()

    # Determine whether to convert the ND format to the NZ format
    if is_weight_nz:
        # nd to fractal_nz
            npu_weight = torch_npu.npu_format_cast(npu_weight.npu(), 29, customize_dtype=cpu_x.dtype)

    # int32 to int4pack
    weight_int4pack = torch_npu.npu_convert_weight_to_int4pack(npu_weight)
    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, weight, antiquant_scale, antiquant_offset, quant_scale,quant_offset, bias, antiquant_group_size):
            if trans_weight:
                weight  = weight.transpose(-1, -2)
                antiquant_scale = antiquant_scale.transpose(-1, -2)
                antiquant_offset = antiquant_offset.transpose(-1, -2)
            return torch_npu.npu_weight_quant_batchmatmul(x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias, antiquant_group_size)
    cpu_model = MyModel()
    model = cpu_model.npu()
    model = torch.compile(cpu_model, backend=npu_backend, dynamic=True, fullgraph=True)
    npu_out = model(cpu_x.npu(), weight_int4pack, cpu_antiquantscale.npu(), cpu_antiquantoffset.npu(), None, None, None, 0)
    ```

  <!-- npu="950" id9 -->
  - $ND$ format, input type `torch.float32`, supported only on <term>Ascend 950PR&950DT products</term>

    ```python
    import torch
    import torch_npu
    import torchair
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    import numpy as np
    from ml_dtypes import float4_e2m1fn

    m = 128
    k = 64
    n = 128
    E2M1_MIN, E2M1_MAX = -6, 6
    antiquant_group_size = 32
    trans_weight = False
    is_weight_nz = False

    cpu_x = torch.randn((m, k),dtype=torch.float16)
    if trans_weight:
        cpu_weight = (E2M1_MIN + (E2M1_MAX - E2M1_MIN) * np.random.random(k * n).reshape((n, k))).astype(float4_e2m1fn)
        cpu_weight = torch.from_numpy(cpu_weight.astype(np.float32))
        cpu_antiquantscale = torch.randint(127 - 5, 127 + 5, (n, k // antiquant_group_size), dtype=torch.uint8) # Use uint8 to carry scale data
    else:
        cpu_weight = (E2M1_MIN + (E2M1_MAX - E2M1_MIN) * np.random.random(k * n).reshape((k, n))).astype(float4_e2m1fn)
        cpu_weight = torch.from_numpy(cpu_weight.astype(np.float32))
        cpu_antiquantscale = torch.randint(127 - 5, 127 + 5, (k // antiquant_group_size, n), dtype=torch.uint8) # Use uint8 to carry scale data
    npu_weight = cpu_weight.npu()

    # Determine whether to convert the ND format to the NZ format
    if is_weight_nz:
        npu_weight = torch_npu.npu_format_cast(npu_weight, 29, customize_dtype=torch.float16)
    weight_packed = torch_npu.npu_convert_weight_to_int4pack(npu_weight)

    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, weight, antiquant_scale, antiquant_offset, quant_scale,quant_offset, bias, antiquant_group_size):
            if trans_weight:
                weight  = weight.transpose(-1, -2)
                antiquant_scale = antiquant_scale.transpose(-1, -2)
                antiquant_offset = antiquant_offset.transpose(-1, -2)
            return torch_npu.npu_weight_quant_batchmatmul(x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias, antiquant_group_size)

    cpu_model = MyModel()
    model = cpu_model.npu()
    model = torch.compile(cpu_model, backend=npu_backend, dynamic=True, fullgraph=True)
    npu_out = model(cpu_x.npu(), weight_packed, cpu_antiquantscale.npu(), None, None, None, None, antiquant_group_size)
    ```
  <!-- end id9 -->
  <!-- npu="950" id10 -->
  - $FRACTAL\_NZ$ format, input type `torch.float32`, supported only on <term>Ascend 950PR&950DT products</term>

    ```python
    import torch
    import torch_npu
    import torchair
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    import numpy as np
    from ml_dtypes import float4_e2m1fn

    m = 128
    k = 64
    n = 128
    E2M1_MIN, E2M1_MAX = -6, 6
    antiquant_group_size = 32
    trans_weight = False
    is_weight_nz = True

    cpu_x = torch.randn((m, k),dtype=torch.float16)
    if trans_weight:
        cpu_weight = (E2M1_MIN + (E2M1_MAX - E2M1_MIN) * np.random.random(k * n).reshape((n, k))).astype(float4_e2m1fn)
        cpu_weight = torch.from_numpy(cpu_weight.astype(np.float32))
        cpu_antiquantscale = torch.randint(127 - 5, 127 + 5, (n, k // antiquant_group_size), dtype=torch.uint8) # Use uint8 to carry scale data
    else:
        cpu_weight = (E2M1_MIN + (E2M1_MAX - E2M1_MIN) * np.random.random(k * n).reshape((k, n))).astype(float4_e2m1fn)
        cpu_weight = torch.from_numpy(cpu_weight.astype(np.float32))
        cpu_antiquantscale = torch.randint(127 - 5, 127 + 5, (k // antiquant_group_size, n), dtype=torch.uint8) # Use uint8 to carry scale data
    npu_weight = cpu_weight.npu()

    # Determine whether to convert the ND format to the NZ format
    if is_weight_nz:
        npu_weight = torch_npu.npu_format_cast(npu_weight, 29, customize_dtype=torch.float16)
    weight_packed = torch_npu.npu_convert_weight_to_int4pack(npu_weight)

    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, weight, antiquant_scale, antiquant_offset, quant_scale,quant_offset, bias, antiquant_group_size):
            if trans_weight:
                weight  = weight.transpose(-1, -2)
                antiquant_scale = antiquant_scale.transpose(-1, -2)
                antiquant_offset = antiquant_offset.transpose(-1, -2)
            return torch_npu.npu_weight_quant_batchmatmul(x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias, antiquant_group_size)

    cpu_model = MyModel()
    model = cpu_model.npu()
    model = torch.compile(cpu_model, backend=npu_backend, dynamic=True, fullgraph=True)
    npu_out = model(cpu_x.npu(), weight_packed, cpu_antiquantscale.npu(), None, None, None, None, antiquant_group_size)
    ```
  <!-- end id10 -->

# torch_npu.npu_convert_weight_to_int4pack

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>           |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term> | √    |

## Function

Packs an `int32` input tensor into the `int4` data type. Every eight `int4` elements are carried by a single `int32` element and stored in an interleaved format.

## Prototype

```python
torch_npu.npu_convert_weight_to_int4pack(weight,inner_k_tiles=0) -> Tensor
```

## Parameters

- **`weight`** (`Tensor`): Required. Input weight. The data layout can be ND or FRACTAL_NZ. The data type can be `int32`. Non-contiguous tensors are not supported. This parameter must be 2D with shape `(k, n)` or `(n, k)`. The last dimension must be aligned to 8 elements. Individual element values must be within the `int4` representation range of [-8, 7].
- **`inner_k_tiles`** (`int`): Optional. Number of K-tiles that are packed together in the internal packing format. The default value is `0`. **Reserved parameter, currently not used.**

## Return Values

`Tensor`

Packed `int4` output tensor. The data type is `int32`. The shape must be `(k, n / 8)` or `(n, k / 8)`. The data layout can be ND.

## Constraints

- This API can be used in inference scenarios.
- This API supports graph mode.

## Examples

- Single-operator call

    ```python
    import torch
    import torch_npu
    
    m = 128
    k = 64
    n = 32
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

    npu_out = torch_npu.npu_weight_quant_batchmatmul(cpu_x.npu(), weight_int4.npu(), cpu_antiquantscale.npu(), cpu_antiquantoffset.npu())
    print(npu_out)
    # Expected output of the preceding code sample:   
    tensor([[  27.2344,  -33.7500,   11.8047,  ...,   40.2188,  -15.7188,
                6.2070],
            [  30.0625,  -66.8750,    7.6758,  ...,  -24.9531,  -11.2031,
            -52.0312],
            [   1.2246, -102.3125,  -13.2734,  ...,  -31.2344,   16.0938,
            -48.9375],
            ...,
            [  38.8750,  -56.6562,  -15.9219,  ...,    1.0771,   -3.8047,
            13.3359],
            [  17.1719,  -26.8594,   -7.6016,  ..., -251.2500,    2.5684,
            -38.1562],
            [  32.5938,  -64.6250,    8.0938,  ...,  -28.5312,  -18.7031,
            18.0781]], device='npu:0', dtype=torch.float16)
    ```

- Graph mode call

    ```python
    import torch
    import torch_npu
    import  torchair
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    
    m = 16
    k = 17
    n = 72
    
    trans_weight = False
    is_weight_nz = False
    
    cpu_x = torch.randn((m, k),dtype=torch.float16)
    if trans_weight:
        cpu_weight = torch.randint(low=-8, high=8, size=(n, k) ,dtype=torch.int32)
        cpu_antiquantscale = torch.ones((n, 1),dtype=torch.float16)
        cpu_antiquantoffset = torch.zeros((n, 1),dtype=torch.float16)
    else:
        cpu_weight = torch.randint(low=-8, high=8, size=(k, n) ,dtype=torch.int32)
        cpu_antiquantscale = torch.ones((1, n),dtype=torch.float16)
        cpu_antiquantoffset = torch.zeros((1, n),dtype=torch.float16)
    
    npu_weight = cpu_weight.npu()
    if is_weight_nz:
       # nd to fractal_nz
       npu_weight = torch_npu.npu_format_cast(npu_weight.npu(), 29)
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
            return torch_npu.npu_weight_quant_batchmatmul(x, weight, antiquant_scale, antiquant_offset, quant_scale ,quant_offset, bias, antiquant_group_size)
    
    cpu_model = MyModel()
    model = cpu_model.npu()
    model = torch.compile(model, backend=npu_backend, dynamic=True, fullgraph=True)

    npu_out = model(cpu_x.npu(), weight_int4pack, cpu_antiquantscale.npu(), cpu_antiquantoffset.npu(), None, None, None, 0)
    print(npu_out)

    # Expected output of the preceding code sample: 
    tensor([[  4.6055,   6.0078,  -9.5078,  ...,  29.7188,   9.4688,  -7.6797],
            [-23.0312, -24.9062,  32.2500,  ...,  -7.8789,  14.3359, -19.7812],
            [-10.6172,  -0.8887,   7.7344,  ...,  -2.7676, -26.4531,   8.3906],
            ...,
            [ -3.8086, -21.3125, -17.8594,  ...,  20.3750,   0.5649,  32.9062],
            [ 16.1094,  13.8203,   3.7461,  ..., -22.6875,  19.0000,   4.4375],
            [ 34.9375, -15.1797, -23.1094,  ..., -13.6797,   8.7734,   6.8750]],
        device='npu:0', dtype=torch.float16)
    ```

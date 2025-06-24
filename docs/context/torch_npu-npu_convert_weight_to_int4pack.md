# torch_npu.npu_convert_weight_to_int4pack

## 功能说明

将`int32`类型的输入tensor打包为`int4`存放，每8个`int4`数据通过一个`int32`数据承载，并进行交叠排放。

## 函数原型

```
torch_npu.npu_convert_weight_to_int4pack(weight,inner_k_tiles=0) -> Tensor
```

## 参数说明

- **weight** (`Tensor`) ：输入的weight，数据格式支持$ND$、$FRACTAL\_NZ$，数据类型支持`int32`，不支持非连续的`Tensor`；维度支持2维，shape支持$（k, n）$、 $(n, k)$，最后一维度需要8个元素对齐，元素的值需要在`int4`的表示范围内，即[-8, 7]。
- **inner_k_tiles** (`int`)：用于制定内部打包格式中，多少个K-tiles被打包在一起，默认值为`0`。**预留参数，暂未使用**。

## 返回值
`Tensor`

代表`int4`打包后的输出，数据类型为`int32`，shape为$（k, n/8）$, $(n, k/8)$，数据格式支持$ND$。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持图模式（PyTorch 2.1版本）。

## 支持的型号

- <term> Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> 
- <term> Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> 

## 调用示例

- 单算子模式调用

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
    # 执行上述代码的输出类似如下    
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

- 图模式调用

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
    model = torch.compile(cpu_model, backend=npu_backend, dynamic=True, fullgraph=True)
    
    npu_out = model(cpu_x.npu(), weight_int4pack, cpu_antiquantscale.npu(), cpu_antiquantoffset.npu(), None, None, None, 0)
    print(npu_out)

    # 执行上述代码的输出类似如下  
    tensor([[  4.6055,   6.0078,  -9.5078,  ...,  29.7188,   9.4688,  -7.6797],
            [-23.0312, -24.9062,  32.2500,  ...,  -7.8789,  14.3359, -19.7812],
            [-10.6172,  -0.8887,   7.7344,  ...,  -2.7676, -26.4531,   8.3906],
            ...,
            [ -3.8086, -21.3125, -17.8594,  ...,  20.3750,   0.5649,  32.9062],
            [ 16.1094,  13.8203,   3.7461,  ..., -22.6875,  19.0000,   4.4375],
            [ 34.9375, -15.1797, -23.1094,  ..., -13.6797,   8.7734,   6.8750]],
        device='npu:0', dtype=torch.float16)
    ```


# torch_npu.npu_dynamic_quant

## 功能说明

- 算子功能：对输入的张量进行per-token对称动态量化。

    如果是MoE（Mixture of Experts，混合专家模型）场景，会引入`group_index`，`smooth_scales`中包含多组smooth向量，按`group_index`中的数值作用到`x`的不同行上。具体地，假如`x`包含m个token，`smooth_scales`有n行，`smooth_scales[0]`会作用到`x[0:group_index[0]]`上，`smooth_scales[i]`会作用到`x[group_index[i-1]: group_index[i]]`上，`i=1, 2, ..., n-1`。

- 计算公式：
    - 如果`smooth_scales`不存在：

        ![](figures/zh-cn_formulaimage_0000002224834166.png)

    - 如果`smooth_scales`存在：

        ![](figures/zh-cn_formulaimage_0000002224834302.png)

        rowMax表示求一行的最大值，DTYPE_MAX表示常量，是y输出对应的数据类型的最大值。

## 函数原型

```
torch_npu.npu_dynamic_quant(x, *, smooth_scales=None, group_index=None, dst_type=None) ->(Tensor, Tensor)
```

## 参数说明

- **x** (`Tensor`)：需要进行量化的源数据张量，必选输入，数据类型支持`float16`、`bfloat16`，数据格式支持$ND$，支持非连续的Tensor。输入`x`的维度必须大于1。进行`int4`量化时，要求x形状的最后一维是8的整数倍。
- **smooth_scales** (`Tensor`)：对`x`进行scales的张量，可选输入，数据类型支持`float16`、`bfloat16`，数据格式支持$ND$，支持非连续的Tensor。shape必须是1维，和`x`的最后一维相等。
- **group_index** (`Tensor`)：对`smooth_scales`进行分组的下标，可选输入，仅在MoE场景下生效。数据类型支持`int32`，数据格式支持$ND$，支持非连续的Tensor。

- **dst_type** (`ScalarType`)：指定量化输出的类型，可选输入，传`None`时当作`int8`处理。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：支持取值`int8`、`quint4x2`。
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持取值`int8`、`quint4x2`。

## 返回值

- **y** (`Tensor`)：量化后的输出Tensor，数据类型由`dst_type`指定。当`dst_type`是`quint4x2`时，`y`的数据类型为`int32`，形状最后一维为`x`最后一维除以8，其余维度与`x`一致，每个`int32`元素包含8个`int4`结果。其他场景下`y`形状与输入`x`一致，数据类型由`dst_type`指定。
- **scale** (`Tensor`)：非对称动态量化过程中计算出的缩放系数，数据类型为`float32`，形状为`x`的形状剔除最后一维。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持图模式（PyTorch 2.1版本）。
- 该接口仅在如下产品支持MoE场景。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> 
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>

- 使用`smooth_scales`时：
    - 若不使用`group_index`，`smooth_scales`必须是一维Tensor，元素数量与`x`的最后一维大小一致。
    - 若使用`group_index`，`smooth_scales`必须是二维Tensor，第二维元素数量与`x`的最后一维大小一致，`group_index`必须是一维数组，元素数量与`smooth_scales`第一维一致。`group_index`中的元素必须是单调递增的，其最后一个元素的值，应等于`x`的元素数量除以`x`的最后一个维度。

## 支持的型号

- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>

## 调用示例

- 单算子模式调用
    - 只有一个输入x

        ```python
        >>> import torch
        >>> import torch_npu
        >>>
        >>> x = torch.rand((3, 3), dtype=torch.float16).to("npu")
        >>> x
        tensor([[0.7261, 0.3726, 0.9126],
                [0.9023, 0.9990, 0.1279],
                [0.8628, 0.6240, 0.9028]], device='npu:0', dtype=torch.float16)
        >>>
        >>> output, scale = torch_npu.npu_dynamic_quant(x)
        >>> output
        tensor([[101,  52, 127],
                [115, 127,  16],
                [121,  88, 127]], device='npu:0', dtype=torch.int8)
        >>> scale
        tensor([0.0072, 0.0079, 0.0071], device='npu:0')
        ```

    - 使用smooth_scales输入

        ```python
        >>> import torch
        >>> import torch_npu
        >>>
        >>> x = torch.rand((3, 3), dtype=torch.float16).to("npu")
        >>> x
        tensor([[0.6680, 0.9492, 0.0845],
                [0.1924, 0.5278, 0.1484],
                [0.6631, 0.9497, 0.0957]], device='npu:0', dtype=torch.float16)
        >>>
        >>> smooth_scales = torch.rand((3,), dtype=torch.float16).to("npu")
        >>> smooth_scales
        tensor([0.8042, 0.0884, 0.8901], device='npu:0', dtype=torch.float16)
        >>>
        >>> output, scale = torch_npu.npu_dynamic_quant(x, smooth_scales=smooth_scales)
        >>> output
        tensor([[127,  20,  18],
                [127,  38, 108],
                [127,  20,  20]], device='npu:0', dtype=torch.int8)
        >>> scale
        tensor([0.0042, 0.0012, 0.0042], device='npu:0')
        ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    torch_npu.npu.set_compile_mode(jit_compile=True)
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    device = torch.device(f'npu:0')
    torch_npu.npu.set_device(device)


    class DynamicQuantModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input_tensor, smooth_scales=None, group_index=None, dst_type=None):
            out, scale = torch_npu.npu_dynamic_quant(input_tensor, smooth_scales=smooth_scales, group_index=group_index,dst_type=dst_type)
            return out, scale


    x = torch.randn((2, 4, 6), device='npu', dtype=torch.float16).npu()
    smooth_scales = torch.randn((6), device='npu', dtype=torch.float16).npu()
    dynamic_quant_model = DynamicQuantModel().npu()
    dynamic_quant_model = torch.compile(dynamic_quant_model, backend=npu_backend, dynamic=True)
    out, scale = dynamic_quant_model(x, smooth_scales=smooth_scales)
    print(out)
    print(scale)

    # 执行上述代码的输出类似如下
    tensor([[[-116,  127,   14, -105,   12,  -44],
            [   7, -127,  -49,  -27,   -4,   -7],
            [ -49,   18,  127,   39,   14,   13],
            [  12,  -47,  127,   73,   28,    1]],

            [[  62,  127,  -61,  -15,   -9,   -8],
            [ 127,  -74,  -66,  117,   27,   27],
            [   3,   65,   29,  127,  -27,   20],
            [  -4, -127,   13,  -40,  -21,  -11]]], device='npu:0',
        dtype=torch.int8)
    [W compiler_depend.ts:133] Warning: Warning: Device do not support double dtype now,
    dtype cast replace with float. (function operator())
    tensor([[0.0080, 0.0422, 0.0219, 0.0132],
            [0.0176, 0.0069, 0.0093, 0.0368]], device='npu:0')
    ```


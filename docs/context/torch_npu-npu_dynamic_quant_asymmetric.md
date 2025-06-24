# torch_npu.npu_dynamic_quant_asymmetric

## 功能说明

- 算子功能：对输入的张量进行per-token非对称动态量化。其中输入的最后一个维度对应一个token，每个token作为一组进行量化。
- 计算公式：假设待量化张量为x，

    ![](figures/zh-cn_formulaimage_0000002128390261.png)

    - $rowMax$、$rowMin$代表按行取最大值、按行取最小值，此处的“行”对应`x`最后一个维度的数据，即一个token。
    - $DST\_MAX$、$DST\_MIN$分别对应量化后的最大值和最小值，在进行`int8`量化时，二者分别对应+127、-128，进行`int4`量化时，分别对应+7、-8
    - 若使用smooth quant，会引入`smooth_scales`输入，其形状与`x`最后一个维度大小一致，在进行量化前，会先令`x`乘以`smooth_scales`，再按上述公式进行量化
    - 若使用smooth quant，MoE（Mixture of Experts，混合专家模型）场景下会引入`smooth_scales`和`group_index`，此时`smooth_scales`中包含多组smooth向量，按`group_index`中的数值作用到`x`的不同行上。具体的，假如`x`包含m个token，`smooth_scales`有n行，`smooth_scales[0]`会作用到`x[0:group_index[0]]`上，`smooth_scales[i]`会作用到`x[group_index[i-1]: group_index[i]]`上，`i=[1, 2, ..., n-1]`。

## 函数原型

```
torch_npu.npu_dynamic_quant_asymmetric(x, *, smooth_scales=None, group_index=None, dst_type=None) -> (Tensor, Tensor, Tensor)
```

## 参数说明

- **x** (`Tensor`)：需要进行量化的源数据张量，必选输入，数据类型支持`float16`、`bfloat16`，数据格式支持$ND$，支持非连续的Tensor。输入`x`的维度必须大于1。进行`int4`量化时，要求`x`形状的最后一维是8的整数倍。
- **smooth_scales** (`Tensor`)：对`x`进行平滑缩放的张量，可选输入，数据类型需要与`x`保持一致，数据格式支持$ND$，支持非连续的Tensor。
- **group_index** (`Tensor`)：在MoE场景下，对`smooth_scales`进行分组的下标，可选输入，数据类型支持`int32`，数据格式支持$ND$，支持非连续的Tensor。
- **dst_type** (`ScalarType`)：用于选择进行`int8/int4`量化，可选输入，输入值只能是`torch.int8`和`torch.quint4x2`，默认为`int8`量化。

## 返回值

- **y** (`Tensor`)：量化后的输出Tensor，在进行`int8`量化时，`y`的数据类型为`int8`，形状与`x`一致；在进行`int4`量化时，`y`的数据类型为`int32`，形状最后一维为`x`最后一维除以8，其余维度与`x`一致，每个`int32`元素包含8个`int4`结果。
- **scale** (`Tensor`)：非对称动态量化过程中计算出的缩放系数，数据类型为`float32`，形状为`x`的形状剔除最后一维。
- **offset** (`Tensor`)：非对称动态量化过程中计算出的偏移系数，数据类型为`float32`，形状为`x`的形状剔除最后一维。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持图模式（PyTorch 2.1版本）。
- 使用可选输入`smooth_scales`、`group_index`、`dst_type`时，必须使用关键字传参。
- 使用`smooth_scales`时：
    - 若不使用`group_index`，`smooth_scales`必须是一维Tensor，元素数量与`x`的最后一维大小一致。
    - 若使用`group_index`，`smooth_scales`必须是二维Tensor，第二维元素数量与`x`的最后一维大小一致，`group_index`必须是一维数组，元素数量与`smooth_scales`第一维一致。`group_index`中的元素必须是单调递增的，其最后一个元素的值，应等于`x`的元素数量除以`x`的最后一个维度。

## 支持的型号

- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> 
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> 

## 调用示例

- 单算子模式调用
    - 只有一个输入`x`，进行`int8`量化

        ```python
        >>> import torch
        >>> import torch_npu
        >>>
        >>> x = torch.rand((3, 8), dtype=torch.half).npu()
        >>> y, scale, offset = torch_npu.npu_dynamic_quant_asymmetric(x)
        >>> y
        tensor([[ -19,   38,   81,   35,    5,  127, -128,  -35],
                [ -20, -128,   49,   41,   65,   96,  -26,  127],
                [  33,  -24, -119,   36, -110, -128, -120,  127]], device='npu:0',
            dtype=torch.int8)
        >>>
        >>> scale
        tensor([0.0038, 0.0034, 0.0023], device='npu:0')
        >>>
        >>> offset
        tensor([-134.1538, -142.3600, -278.4672], device='npu:0')
        ```

    - 只有一个输入`x`，进行`int4`量化

        ```python
        >>> import torch
        >>> import torch_npu
        >>>
        >>> x = torch.rand((3, 8), dtype=torch.half).npu()
        >>> y, scale, offset = torch_npu.npu_dynamic_quant_asymmetric(x, dst_type=torch.quint4x2)
        >>>
        >>> y
        tensor([[  126435331],
                [ 2095038856],
                [-1489413144]], device='npu:0', dtype=torch.int32)
        >>>
        >>> scale
        tensor([0.0435, 0.0579, 0.0610], device='npu:0')
        >>>
        >>> offset
        tensor([-10.7191, -10.2500,  -8.4562], device='npu:0')
        ```

    - 使用`smooth_scales`输入，非MoE场景（不使用`group_index`），进行`int8`量化

        ```python
        >>> import torch
        >>> import torch_npu
        >>>
        >>> x = torch.rand((3, 8), dtype=torch.half).npu()
        >>> smooth_scales = torch.rand((8,), dtype=torch.half).npu()
        >>> y, scale, offset = torch_npu.npu_dynamic_quant_asymmetric(x, smooth_scales=smooth_scales)
        >>>
        >>> y
        tensor([[-112, -110, -128,   29,  127,   12,  -11,  -51],
                [ -82,   52, -127,   14,   17, -128,  -63,  127],
                [-102,  -47, -123, -128,  127,   51, -101,  -60]], device='npu:0',
            dtype=torch.int8)
        >>>
        >>> scale
        tensor([0.0019, 0.0023, 0.0027], device='npu:0')
        >>>
        >>> offset
        tensor([-158.2731, -137.6059, -160.5875], device='npu:0')
        ```

    - 使用`smooth_scales`输入，MoE场景（使用`group_index`），进行`int8`量化

        ```python
        >>> import torch
        >>> import torch_npu
        >>>
        >>> x = torch.rand((3, 8), dtype=torch.half).npu()
        >>> smooth_scales = torch.rand((2, 8), dtype=torch.half).npu()
        >>> group_index = torch.Tensor([1, 3]).to(torch.int32).npu()
        >>> y, scale, offset = torch_npu.npu_dynamic_quant_asymmetric(x, smooth_scales=smooth_scales, group_index=group_index)
        >>>
        >>> y
        tensor([[  -6, -127,   70,  -92,    7,  127,   58, -128],
                [-128,  127,  -61,   20,  -86,  -62, -128, -124],
                [-116,  -82, -128,    3,   12,   64,  127, -120]], device='npu:0',
            dtype=torch.int8)
        >>>
        >>> scale
        tensor([0.0030, 0.0028, 0.0030], device='npu:0')
        >>>
        >>> offset
        tensor([-133.4975, -149.8293, -135.2519], device='npu:0')
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
    
    device=torch.device(f'npu:4')    
    torch_npu.npu.set_device(device)
    
    class DynamicQuantModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
    
        def forward(self, input_tensor, smooth_scales=None, group_index=None, dst_type=None):
            out, scale, offset = torch_npu.npu_dynamic_quant_asymmetric(input_tensor, smooth_scales=smooth_scales, group_index=group_index, dst_type=dst_type)
            return out, scale, offset
    
    x = torch.randn((2, 4, 6),device='npu',dtype=torch.float16).npu()
    smooth_scales = torch.randn((6),device='npu',dtype=torch.float16).npu()
    dynamic_quant_model = DynamicQuantModel().npu()
    dynamic_quant_model = torch.compile(dynamic_quant_model, backend=npu_backend, dynamic=True)
    out, scale, offset = dynamic_quant_model(x, smooth_scales=smooth_scales)
    print(out)
    print(scale)
    print(offset)

    # 执行上述代码的输出类似如下    
    tensor([[[  96,   34, -128,  127,   93,    7],
            [  20,  127, -128,    7,  -91,  118],
            [ -23, -128,  -88,   40,  127,  114],
            [  -5, -128,  -21,  -52,   25,  127]],

            [[  -7,  -45, -128,  -38,  127,  -52],
            [  21,   86,  -26,   10,  127, -128],
            [  14,  -41,  127,  -13, -128, -114],
            [  -3,  -84,   52, -128,  -29,  127]]], device='npu:4',
        dtype=torch.int8)
    .[W compiler_depend.ts:133] Warning: Warning: Device do not support double dtype now
    .tensor([[0.0036, 0.0035, 0.0050, 0.0025],
            [0.0028, 0.0066, 0.0035, 0.0039]], device='npu:4')
    tensor([[ 73.0180,  12.5112, -22.1802, -32.9387],
            [ 31.0871,  28.8000,  -2.9755,  -3.0922]], device='npu:4')

    ```


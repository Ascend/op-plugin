# torch_npu.npu_quantize

## 功能说明

- 算子功能：对输入的张量进行量化处理。
- 计算公式：
    - 如果**div_mode**为`True`：

        ![](figures/zh-cn_formulaimage_0000001888362410.png)

    - 如果**div_mode**为`False`：

        ![](figures/zh-cn_formulaimage_0000001888362854.png)

## 函数原型

```
torch_npu.npu_quantize(input, scales, zero_points, dtype, axis=1, div_mode=True) -> Tensor
```

## 参数说明

- **input** (`Tensor`)：需要进行量化的源数据张量，必选输入，数据格式支持$ND$，支持非连续的Tensor。`div_mode`为`False`且`dtype`为`quint4x2`时，最后一维需要能被8整除。
    - <term>Atlas 推理系列产品</term>：数据类型支持`float`、`float16`。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持`float`、`float16`、`bfloat16`。

- **scales** (`Tensor`)：对`input`进行scales的张量，必选输入：
    - **`div_mode`** 为`True`时
        - <term>Atlas 推理系列产品</term>：数据类型支持`float`。
        - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持`float`、`bfloat16`。

    - **`div_mode`** 为`False`时，数据格式支持$ND$，支持非连续的Tensor。支持1维或多维（1维时，对应轴的大小需要与`input`中第`axis`维相等或等于1；多维时，`scales`的shape需要与`input`的shape维度相等，除`axis`指定的维度，其他维度为1，`axis`指定的维度必须和`input`对应的维度相等或等于1）。
        - <term>Atlas 推理系列产品</term>：数据类型支持`float`、`float16`。
        - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持`float`、`float16`、`bfloat16`。

- **zero_points** (`Tensor`)：对`input`进行offset的张量，可选输入。
    - **`div_mode`** 为`True`时
        - <term>Atlas 推理系列产品</term>：数据类型支持`int8`、`uint8`、`int32`。
        - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持`int8`、`uint8`、`int32`、`bfloat16`。

    - **`div_mode`** 为`False`时，数据格式支持$ND$，支持非连续的Tensor。支持1维或多维（1维时，对应轴的大小需要与`input`中第`axis`维相等或等于1；多维时，`scales`的shape需要与`input`维度相等，除`axis`指定的维度，其他维度为1，`axis`指定的维度必须和`input`对应的维度相等）。`zero_points`的shape和`dtype`需要和`scales`一致。
        - <term>Atlas 推理系列产品</term>：数据类型支持`float`、`float16`。
        - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持`float`、`float16`、`bfloat16`。

- **dtype** (`int`)：`int`类型，指定输出参数的类型。
    - **`div_mode`** 为`True`时，
        - <term>Atlas 推理系列产品</term>：类型支持`qint8`、`quint8`、`int32`。
        - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：类型支持`qint8`、`quint8`、`int32`。

    - **`div_mode`** 为`False`时，类型支持`qint8`、`quint4x2`。如果`dtype`为`quint4x2`时，输出tensor类型为`int32`，由8个`int4`拼接。

- **axis** (`int`)：`int`类型，量化的element-wise轴，其他的轴做broadcast，默认值为`1`。

    **`div_mode`** 为`False`时，`axis`取值范围是[-2, +∞）且指定的轴不能超过输入`input`的维度数。如果`axis=-2`，代表量化的element-wise轴是输入`input`的倒数第二根轴；如果`axis`大于-2，量化的element-wise轴是输入的最后一根轴。

- **div_mode** (`bool`)：布尔类型，表示计算`scales`模式。当`div_mode`为`True`时，表示用除法计算`scales`；`div_mode`为`False`时，表示用乘法计算`scales`，默认值为`True`。

## 返回值
`Tensor`

公式中的输出，输出大小与`input`一致。数据类型由参数`dtype`指定，如果参数`dtype`为`quint4x2`，输出的`dtype`是`int32`，shape的最后一维是输入shape最后一维的1/8，shape其他维度和输入一致。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持图模式（PyTorch 2.1版本）。
- `div_mode`为`False`时：
    - 支持<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>。
    - 当`dtype`为`quint4x2`或者`axis`为-2时，不支持<term>Atlas 推理系列产品</term>。

## 支持的型号

- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> 
- <term>Atlas 推理系列产品</term> 

## 调用示例

- 单算子模式调用
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>

        ```python
        >>> import torch
        >>> import torch_npu
        >>>
        >>> x = torch.randn(1, 1, 12).bfloat16().npu()
        >>> scale = torch.tensor([0.1] * 12).bfloat16().npu()
        >>> out = torch_npu.npu_quantize(x, scale, None, torch.qint8, -1, False)
        >>> x
        tensor([[[ 0.9609,  1.3281, -0.6172,  0.5469, -1.1797, -1.1719, -0.7422,
                0.9727, -0.9062, -0.0815, -0.8047,  1.0703]]], device='npu:0',
            dtype=torch.bfloat16)
        >>> out
        tensor([[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], device='npu:0',
            dtype=torch.int8)
        ```

    - <term>Atlas 推理系列产品</term>

        ```python
        >>> import torch
        >>> import torch_npu
        >>>
        >>> x = torch.randn((2, 3, 12), dtype=torch.float).npu()
        >>> scale = torch.tensor(([3] * 12),dtype=torch.float).npu()
        >>> out = torch_npu.npu_quantize(x, scale, None, torch.qint8, -1, False)
        >>> x
        tensor([[[-7.7834e-01, -1.0473e+00, -1.1155e+00,  1.2233e+00, -1.2271e+00,
                -2.5612e+00, -1.8274e-01,  2.8293e+00,  1.9029e-01, -1.9333e+00,
                -4.9270e-01, -1.0650e+00],
                [-8.9416e-01,  3.1869e-02, -5.8144e-01, -4.9477e-01,  9.7323e-02,
                -3.8681e-01,  2.1969e-03, -6.3244e-01,  7.1591e-01, -1.8587e-01,
                -1.3381e+00, -2.6253e-01],
                [ 1.8462e-02,  1.2397e-01, -9.0656e-01, -9.9280e-01, -4.4235e-02,
                1.0623e+00, -9.8437e-02,  1.2941e+00,  1.0805e+00, -1.7269e-01,
                -9.9205e-02, -6.1429e-01]],

                [[ 1.3678e+00, -2.7348e-01, -4.1354e-01, -9.4638e-01,  4.2792e-01,
                8.0462e-01,  9.3584e-01,  6.3704e-01,  1.1269e+00, -1.5329e+00,
                5.8572e-01, -1.3966e+00],
                [ 3.5882e-01,  8.7029e-01, -1.3176e+00,  1.1601e+00, -3.6984e-01,
                7.3642e-01, -1.0755e+00,  6.6557e-01,  3.1149e+00, -6.8776e-01,
                -1.0913e+00,  4.4962e-01],
                [-1.2505e+00,  1.5474e+00, -7.4332e-02, -1.6657e+00,  1.3275e+00,
                5.8914e-02,  8.4287e-01, -1.7109e+00,  1.8256e-01,  3.2937e-01,
                2.4875e+00,  1.3921e+00]]], device='npu:0')
        >>> out
        tensor([[[-2, -3, -3,  4, -4, -8, -1,  8,  1, -6, -1, -3],
                [-3,  0, -2, -1,  0, -1,  0, -2,  2, -1, -4, -1],
                [ 0,  0, -3, -3,  0,  3,  0,  4,  3, -1,  0, -2]],

                [[ 4, -1, -1, -3,  1,  2,  3,  2,  3, -5,  2, -4],
                [ 1,  3, -4,  3, -1,  2, -3,  2,  9, -2, -3,  1],
                [-4,  5,  0, -5,  4,  0,  3, -5,  1,  1,  7,  4]]], device='npu:0',
            dtype=torch.int8)
        ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig

    x = torch.randn((2, 3, 12), dtype=torch.float16).npu()
    scale = torch.tensor(([3] * 12), dtype=torch.float16).npu()
    axis = 1
    div_mode = False


    class Network(torch.nn.Module):
        def __init__(self):
            super(Network, self).__init__()

        def forward(self, x, scale, zero_points, dst_type, div_mode):
            return torch_npu.npu_quantize(x, scale, zero_points=zero_points, dtype=dst_type, div_mode=div_mode)


    model = Network()
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    config.debug.graph_dump.type = 'pbtxt'
    model = torch.compile(model, fullgraph=True, backend=npu_backend, dynamic=True)
    output_data = model(x, scale, None, dst_type=torch.qint8, div_mode=div_mode)
    print("shape of x:", x.shape)
    print("shape of output_data:", output_data.shape)
    print("x:", x)
    print("output_data:", output_data)



    # 执行上述代码的输出类似如下
    shape of x: torch.Size([2, 3, 12])
    shape of output_data: torch.Size([2, 3, 12])
    x: tensor([[[ 0.2600,  0.6782,  0.5024,  0.9492,  0.6089,  0.7461,  1.5332,
            -0.2123,  0.6558, -0.8354, -0.5366, -0.6821],
            [-0.2522,  0.2415, -0.0269, -0.1497,  0.2256, -0.5239,  0.7363,
            -0.2468,  1.6064,  1.4170, -0.2213,  1.5947],
            [-0.6328,  0.8105,  0.2532,  1.0684, -1.2119, -0.6865,  0.7451,
            -0.8120,  0.6401, -2.1270, -0.9482, -1.1973]],

            [[-1.7461, -1.1758, -0.5352,  1.5938,  1.8945, -2.2500, -0.5073,
            -0.8164,  0.8267, -0.4377,  1.2490,  0.2415],
            [ 0.8062, -1.0498, -0.8345,  1.1465, -0.7349,  0.1317,  0.2280,
            -0.8145,  0.2673,  1.4756, -1.6768,  1.1572],
            [-0.3147, -0.4446, -1.0508,  0.8325,  1.4590,  0.2096, -0.9961,
            0.6089, -0.2460,  1.1543,  0.9277,  0.1079]]], device='npu:0',
        dtype=torch.float16)
        
    output_data: tensor([[[ 1,  2,  2,  3,  2,  2,  5, -1,  2, -3, -2, -2],
            [-1,  1,  0,  0,  1, -2,  2, -1,  5,  4, -1,  5],
            [-2,  2,  1,  3, -4, -2,  2, -2,  2, -6, -3, -4]],

            [[-5, -4, -2,  5,  6, -7, -2, -2,  2, -1,  4,  1],
            [ 2, -3, -3,  3, -2,  0,  1, -2,  1,  4, -5,  3],
            [-1, -1, -3,  2,  4,  1, -3,  2, -1,  3,  3,  0]]], device='npu:0',
        dtype=torch.int8)

    ```


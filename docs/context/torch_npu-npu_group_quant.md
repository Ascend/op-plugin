# torch_npu.npu_group_quant

## 功能说明

- 算子功能：对输入的张量进行分组量化操作。
- 计算公式为：

    ![](figures/zh-cn_formulaimage_0000002077236353.png)

## 函数原型

```
torch_npu.npu_group_quant(x, scale, group_index, *, offset=None, dst_dtype=None) -> Tensor
```

## 参数说明

- **x** (`Tensor`)：需要做量化的源数据张量，必选输入。数据类型支持`float32`、`float16`、`bfloat16`。数据格式支持$ND$。`x`为2维张量，如果`dst_dtype`为`quint4x2`，shape的最后一维需要能被8整除。
- **scale** (`Tensor`)：量化中的scale值，必选输入。数据类型支持`float32`、`float16`、`bfloat16`。数据格式支持$ND$。`scale`为2维张量，第0维大小不支持为0，并且`scale`的第1维与x的第1维相等。
- **group_index** (`Tensor`)：分组量化中的group编号值，必选输入。数据类型支持`int32`、`int64`。数据格式支持$ND$。`group_index`为1维张量，并且`group_index`的第0维与`scale`的第0维相等。
- **offset** (`Tensor`)：量化中的offset值，可选输入。数据类型支持`float32`、`float16`、`bfloat16`，且数据类型与`scale`的数据类型一致。数据格式支持$ND$。`offset`为一个数。
- **dst_dtype** (`ScalarType`)：可选参数，输入值允许为`int8`或`quint4x2`，默认值为`int8`。

## 返回值
`Tensor`

一个`Tensor`类型的输出，代表`npu_group_quant`的计算结果。如果参数`dst_dtype`为`int8`，输出大小与输入`x`的大小一致。如果参数`dst_dtype`为`quint4x2`，输出的数据类型是`int32`，shape的第0维大小与输入`x`的第0维大小一致，最后一维是输入`x`的最后一维的1/8。

## 约束说明

- 如果属性`dst_dtype`为`quint4x2`，则输入`x`的shape的最后一维需要能被8整除。
- 输入`group_index`必须是非递减序列，最小值不能小于0，最大值必须与输入`x`的shape的第0维大小相等。
- 该接口支持图模式（PyTorch 2.1版本）。

## 支持的型号

- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> 
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> 

## 调用示例

- 单算子模式调用

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> x = torch.randn(6, 4).to(torch.float16).npu()
    >>> x
    tensor([[ 1.0029, -1.2373,  1.0107, -0.2681],
            [ 0.5791,  0.1101,  1.0059, -0.9658],
            [-1.7637,  1.7588, -1.3193,  0.3989],
            [ 1.3262,  0.4854,  1.9551,  0.9697],
            [-0.8770, -1.8828,  2.1777, -0.0050],
            [ 0.4722,  0.5605,  0.8267, -0.9810]], device='npu:0',
        dtype=torch.float16)
    >>>
    >>> scale = torch.randn(4, 4).to(torch.float32).npu()
    >>> scale
    tensor([[-0.2710, -0.9381,  0.2850, -1.1230],
            [ 0.5217, -0.7233, -0.1730, -0.1245],
            [-1.5433, -0.9129, -2.2095,  1.7371],
            [-0.8253,  0.3973,  0.1430,  0.3885]], device='npu:0')
    >>>
    >>> group_index = torch.tensor([1, 4, 6, 6], dtype=torch.int32).npu()
    >>> group_index
    tensor([1, 4, 6, 6], device='npu:0', dtype=torch.int32)
    >>>
    >>> offset = torch.randn(1).to(torch.float32).npu()
    >>> offset
    tensor([-1.1658], device='npu:0')
    >>>
    >>> y = torch_npu.npu_group_quant(x, scale, group_index, offset=offset, dst_dtype=torch.qint8)
    >>> y
    tensor([[-1,  0, -1, -1],
            [-1, -1, -1, -1],
            [-2, -2, -1, -1],
            [ 0, -2, -2, -1],
            [ 0,  1, -6, -1],
            [-2, -2, -3, -3]], device='npu:0', dtype=torch.int8)    
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig

    attr_dst_type = 2
    attr_dst_type_torch = torch.qint8 if attr_dst_type == 2 else torch.quint4x2

    x = torch.randn(6, 4).to(torch.float16).npu()
    scale = torch.randn(4, 4).to(torch.float32).npu()
    group_index = torch.tensor([1, 4, 6, 6], dtype=torch.int32).npu()
    offset = torch.randn(1).to(torch.float32).npu()


    class Network(torch.nn.Module):
        def __init__(self):
            super(Network, self).__init__()

        def forward(self, x, scale, group_index, offset, dst_type):
            return torch_npu.npu_group_quant(x, scale, group_index, offset=offset, dst_dtype=dst_type)


    model = Network()
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    config.debug.graph_dump.type = 'pbtxt'
    model = torch.compile(model, fullgraph=True, backend=npu_backend, dynamic=True)

    output_data = model(x, scale, group_index, offset=offset, dst_type=attr_dst_type_torch)
    print(output_data)

    # 执行上述代码的输出类似如下
    tensor([[ 2,  0,  0,  1],
            [-1,  1,  1,  0],
            [-1, -1,  0,  0],
            [ 2,  1,  1,  0],
            [ 1, -1,  0,  1],
            [ 0,  0, -1, -1]], device='npu:0', dtype=torch.int8)
    ```


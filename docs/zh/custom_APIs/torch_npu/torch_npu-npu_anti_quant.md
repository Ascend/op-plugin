# torch_npu.npu_anti_quant

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>           |    √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √   |
|<term>Atlas 推理系列产品</term>| √   |

## 功能说明

- API功能：对张量`x`进行反量化操作，即将整数恢复为浮点数。
- 计算公式：

  其中*out*是输出，*quant*是指定的输出类型`dst_dtype`。

  $$
  out = \text{quant}((x + \text{offset}) * \text{scale}) 
  $$

## 函数原型

```
torch_npu.npu_anti_quant(x, scale, *, offset=None, dst_dtype=None, src_dtype=None) -> Tensor
```

## 参数说明

- **x** (`Tensor`)：必选参数，需要做反量化的输入，数据格式支持$ND$，支持非连续的Tensor，支持空Tensor。最大支持8维。
  - <term>Atlas 推理系列产品</term>：数据类型支持`int8`。
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`int8`、`int32`，其中`int32`类型数据的每个值是由8个`int4`数值拼成。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`int8`、`int32`，其中`int32`类型数据的每个值是由8个`int4`数值拼成。

- **scale** (`Tensor`)：必选参数，反量化中的`scale`值，仅支持1维Tensor，shape为$(n,)$。其中n可以为1，如果n不为1，当`x`为`int8`类型时，必须与输入`x`的尾轴维度大小相同；当`x`为`int32`类型时，必须为输入`x`的尾轴维度大小的8倍。数据格式支持$ND$，支持非连续的Tensor，支持空Tensor。
  - <term>Atlas 推理系列产品</term>：数据类型支持`float32`。
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`float32`、`bfloat16`。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float32`、`bfloat16`。

- <strong>*</strong>：必选参数，代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。

- **offset** (`Tensor`)：可选参数，反量化中的`offset`值。仅支持1维Tensor，数据类型和shape必须与`scale`一致。数据格式支持$ND$，支持非连续的Tensor，支持空Tensor。

- **dst_dtype** (`ScalarType`)：可选参数，指定输出的数据类型，默认值为`float16`。
  - <term>Atlas 推理系列产品</term>：数据类型支持`float16`。
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`float16`、`bfloat16`。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float16`、`bfloat16`。

- **src_dtype** (`ScalarType`)：可选参数，指定源输入的数据类型，默认值为`int8`。
  - <term>Atlas 推理系列产品</term>：数据类型支持`int8`。
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`quint4x2`或`int8`。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`quint4x2`或`int8`。

## 返回值说明
`Tensor`

代表`npu_anti_quant`的计算结果，对应公式中的*out*。支持非连续的Tensor，支持空Tensor。

## 约束说明

- 该接口支持推理、训练场景下使用。
- 该接口支持图模式。
- `x`、`scale`这两个输入中不能为None。

## 调用示例

- 单算子模式调用

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> x_tensor = torch.tensor([1, 2, 3, 4], dtype=torch.int8).npu()
    >>> scale = torch.tensor([2.0], dtype=torch.float).npu()
    >>> offset = torch.tensor([2.0], dtype=torch.float).npu()
    >>> out = torch_npu.npu_anti_quant(x_tensor, scale, offset=offset, dst_dtype=torch.float16)
    >>> out
    tensor([ 6.,  8., 10., 12.], device='npu:0', dtype=torch.float16)
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig
    
    config = CompilerConfig()
    config.debug.graph_dump.type = 'pbtxt'
    npu_backend = tng.get_npu_backend(compiler_config=config)
    x_tensor = torch.tensor([1,2,3,4], dtype=torch.int8).npu()
    scale = torch.tensor([2.0], dtype=torch.float).npu()
    offset = torch.tensor([2.0], dtype=torch.float).npu()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self,x,scale,offset):
            return torch_npu.npu_anti_quant(x, scale, offset=offset, dst_dtype=torch.float16)

    cpu_model = Model()
    model = cpu_model.npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
    output = model(x_tensor,scale,offset)
    print(output)

    # 执行上述代码的输出类似如下
    tensor([ 6.,  8., 10., 12.], device='npu:0', dtype=torch.float16)
    ```

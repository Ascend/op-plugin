# torch\_npu.npu\_grouped\_dynamic\_block\_quant

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| <term>Atlas 350 加速卡</term> | √ |

## 功能说明

- API功能：根据传入的分组索引的起始值（group\_list）对各个group以基本块的粒度进行量化，量化为（FP8/HiFP8），并输出量化参数scale（FP32）。
- 计算公式：

$$
input\_max=block\_reduce\_max(abs(input)) \\
scale=min(input\_max/FP8\_MAX, 1/min\_scale) \\
y=cast\_to\_[HiF8/FP8](input/scale)
$$

## 函数原型

```python
torch_npu.npu_grouped_dynamic_block_quant(input, group_list, *, min_scale=0.0, round_mode="rint", dst_type=291, row_block_size=1, col_block_size=128, group_list_type=0) -> (Tensor, Tensor)
```

## 参数说明

- **input**（`Tensor`）：必选参数，表示算子输入的Tensor，公式中的$input$。维度为2-3维（形状为\[M, N\]或\[B, M, N\]），数据格式支持ND，数据类型支持`bfloat16`、`float16`。支持非连续Tensor，支持空Tensor。
- **group\_list**（`Tensor`）：必选参数，表示量化分组的起始索引，要求大于等于0，且非递减，并且最后一个数需要与`x`的-2轴大小相等。维度仅支持1维，数据格式支持ND，数据类型支持`int32`，支持非连续Tensor，支持空Tensor。
- \*：代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **min\_scale**（`float`）：可选参数，表示参与scale计算的最小值，对应公式中的$min\_scale$。取值要求大于等于0，默认值为0.0。数据类型为`float32`。
- **round\_mode**（`str`）：可选参数，表示最后由高bit数据cast到目标数据类型的近似模式。默认采用"rint"。
  - 当dst\_type为float8\_e5m2/float8\_e4m3fn时，模式支持"rint"。
  - 当dst\_type为hifloat8时，模式支持"round"、"hybrid"。

- **dst\_type**（`int`）：可选参数，表示数据转换后y的数据类型，支持取值为290（hifloat8）、291（float8\_e5m2）、292（float8\_e4m3fn）/36，默认类型为float8\_e5m2。
- **row\_block\_size**（`int`）：可选参数，表示指定M轴上的量化粒度。当前支持取值为1、128、256、512，默认值为1。
- **col\_block\_size**（`int`）：可选参数，表示指定N轴上的量化粒度。当前支持取值64、128、192、256，默认值为128。
- **group\_list\_type**（`int`）：可选参数，表示group\_list功能类型，默认值为0，表示group\_list为cumsum模式。

## 返回值说明

- **y**（`Tensor`）：表示量化后的输出Tensor，公式中的$y$。shape的维度与`input`保持一致。数据类型支持`hifloat8`、`float8_e5m2`、`float8_e4m3fn`，支持非连续Tensor，支持空Tensor。
- **scale**（`Tensor`）：表示每个分组对应的量化尺度，公式中的$scale$。数据类型支持`float32`，支持非连续Tensor，支持空Tensor。如果输入`input` shape为\[M, N\]，`group_list`的shape为\[g\]，则`scale` shape为\[\(M//row\_block\_size+g\), \(N/col\_block\_size\)\]。如果输入`input` shape为\[B, M, N\]，`group_list`的shape为\[g\]，则`scale` shape为\[B, \(M//row\_block\_size+g\), \(N/col\_block\_size\)\]。

## 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口支持单算子模式和图模式调用。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    import numpy as np
    
    def grouped_dynamic_block_quant_test(x_dtype, dst_type):
        # 构造x tensor
        x = torch.randn((1, 128), dtype=x_dtype).npu()
        group_list = torch.ones((1,), dtype=torch.int32).npu()
        y_tmp, scale_tmp = torch_npu.npu_grouped_dynamic_block_quant(x, group_list, dst_type=dst_type)
        y = y_tmp.cpu()
        scale = scale_tmp.cpu()
        print("GroupedDynamicBlockQuant result:")
        print("x:\n", x)
        print("group_list:\n", group_list)
        print("y:\n", y)
        print("scale:\n", scale)
    
    if __name__ == "__main__":
        grouped_dynamic_block_quant_test(torch.float16, torch.float8_e5m2)
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair
    import numpy as np
    
    class GroupedDynamicBlockQuantModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, group_list, min_scale=0.0, dst_type=torch_npu.float8_e5m2, row_block_size=1, col_block_size=128, group_list_type=0):
            return torch_npu.npu_grouped_dynamic_block_quant(x, group_list, min_scale=min_scale, dst_type=dst_type, row_block_size=row_block_size, col_block_size=col_block_size, group_list_type=group_list_type)
    
    def grouped_dynamic_block_quant_test(x_dtype, dst_type):
        # 构造x tensor
        x = torch.randn((1, 128), dtype=x_dtype).npu()
        group_list = torch.ones((1,), dtype=torch.int32).npu()
        model = GroupedDynamicBlockQuantModel()
        model.to('npu')
        config = torchair.CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        model = torch.compile(model, backend=npu_backend, dynamic=False)
        y_tmp, scale_tmp = model(x, group_list, dst_type=dst_type)
        y = y_tmp.cpu()
        scale = scale_tmp.cpu()
        print("GroupedDynamicBlockQuant result:")
        print("x:\n", x)
        print("y:\n", y)
        print("scale:\n", scale)
    
    if __name__ == "__main__":
        grouped_dynamic_block_quant_test(torch.float16, torch.float8_e5m2)
    ```

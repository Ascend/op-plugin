# torch_npu.npu_dynamic_block_quant

## 产品支持情况

| 产品                                                      | 是否支持 |
| --------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品</term>                        |    √     |
| <term>Atlas A3 推理系列产品</term>                        |    √     |
| <term>Atlas A2 训练系列产品</term>                        |    √     |
| <term>Atlas A2 推理系列产品</term> |    √     |

## 功能说明

-   API功能：对输入张量，通过给定的`row_block_size`和`col_block_size`将输入划分成多个数据块，以数据块为基本粒度进行量化。在每个块中，先计算出当前块对应的量化参数`scale`，并根据`scale`对输入进行量化。输出最终的量化结果，以及每个块的量化参数`scale`。

- 计算公式：
  $$
  input\_max = block\_reduce\_max(abs(x))
  $$

  $$
  scale = min(FP8\_MAX(HiF8\_MAX / INT8\_MAX) / input\_max, 1/min\_scale)
  $$

  $$
  y = cast\_to\_[FP8/HiF8/INT8](x / scale)
  $$

  其中$block\_reduce\_max$代表求每个`block`中的最大值。

## 函数原型

```
torch_npu.npu_dynamic_block_quant(x, *, min_scale=0.0, round_mode="rint", dst_type=1, row_block_size=1, col_block_size=128) -> (Tensor, Tensor)
```

## 参数说明

- **x** (`Tensor`)：必选参数，输入张量，数据类型支持`float16`、`bfloat16`，支持非连续的Tensor，数据格式支持ND。当前shape支持2维和3维。
- **min_scale** (`float`)：可选参数，参与`scale`计算的最小`scale`值。当前支持取值大于等于0。
- **round_mode** (`str`)：可选参数，指定类型转换到输出的转换方式。当前仅支持取值`rint`。
- **dst_type** (`int`)：可选参数，指定输出`y`的数据类型。当前仅支持取值1，表示输出`y`的数据类型为`int8`。
- **row_block_size** (`int`)：可选参数，指定单个量化的数据块的行大小。当前仅支持取值1。
- **col_block_size** (`int`)：可选参数，指定单个量化的数据块的列大小，当前仅支持取值128。

## 返回值说明

- **y** (`Tensor`)：量化结果。
- **scale** (`Tensor`)：量化时使用的量化参数。

## 调用示例

  ```python
  >>> import torch
  >>> import torch_npu
  
  >>> x = torch.rand(3, 4).to("npu").to(torch.float16)
  >>> min_scale = 0
  >>> dst_type = 1
  >>> row_block_size = 1
  >>> col_block_size = 128
  
  >>> y, scale = torch_npu.npu_dynamic_block_quant(x, min_scale=min_scale, dst_type=dst_type, row_block_size=row_block_size, col_block_size=col_block_size)
  >>> y
  tensor([[ 92,  65,  15, 127],
          [100, 127, 116,  64],
          [ 95,  15,  87, 127]], device='npu:0', dtype=torch.int8)
  >>> scale
  tensor([[0.0063],
          [0.0076],
          [0.0073]], device='npu:0')
  ```

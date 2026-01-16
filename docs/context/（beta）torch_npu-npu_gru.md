# （beta）torch_npu.npu_gru

> [!NOTICE]  
> 该接口计划废弃，可以使用`torch.gru`接口进行替换。

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>           |    √     |
|<term>Atlas A2 训练系列产品</term> | √   |
|<term>Atlas 训练系列产品</term> | √   |
|<term>Atlas 推理系列产品</term>| √   |

## 功能说明

计算DynamicGRUV2。

## 函数原型

```
torch_npu.npu_gru(input, hx, weight_input, weight_hidden, bias_input, bias_hidden, seq_length, has_biases, num_layers, dropout, train, bidirectional, batch_first) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

- **input**（`Tensor`）：数据类型支持`float16`；格式支持FRACTAL_NZ。
- **hx**（`Tensor`）：数据类型支持`float16`，`float32`；格式支持FRACTAL_NZ。
- **weight_input**（`Tensor`）：数据类型支持`float16`；格式支持FRACTAL_Z。
- **weight_hidden**（`Tensor`）：数据类型支持`float16`；格式支持FRACTAL_Z。
- **bias_input**（`Tensor`）：数据类型支持`float16`，`float32`；格式支持ND。
- **bias_hidden**（`Tensor`）：数据类型支持`float16`，`float32`；格式支持ND。
- **seq_length**（`Tensor`）：数据类型支持`int32`；格式支持ND。
- **has_biases**（`bool`）：默认值为`True`。
- **num_layers**（`int`）：层数。
- **dropout**（`float`）：丢弃概率。
- **train**（`bool`）：训练是否在op进行，默认值为`True`。
- **bidirectional**（`bool`）：默认值为`True`。
- **batch_first**（`bool`）：默认值为`True`。

## 返回值说明

- **y**（`Tensor`）：数据类型支持`float16`，`float32`；格式支持FRACTAL_NZ。
- **output_h**（`Tensor`）：数据类型支持`float16`，`float32`；格式支持FRACTAL_NZ。
- **update**（`Tensor`）：数据类型支持`float16`，`float32`；格式支持FRACTAL_NZ。
- **reset**（`Tensor`）：数据类型支持`float16`，`float32`；格式支持FRACTAL_NZ。
- **new**（`Tensor`）：数据类型支持`float16`，`float32`；格式支持FRACTAL_NZ。
- **hidden_new**（`Tensor`）：数据类型支持`float16`，`float32`；格式支持FRACTAL_NZ。

## 约束说明

接口暂不支持jit_compile=False，需要在该模式下使用时请将"DynamicGRUV2"添加至"NPU_FUZZY_COMPILE_BLACKLIST"选项内，具体操作可参考[添加二进制黑名单示例](blacklist.md)。

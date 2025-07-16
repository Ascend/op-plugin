# （beta）torch_npu.npu_gru

>**须知：**<br>
>该接口计划废弃，可以使用torch.gru接口进行替换。

## 函数原型

```
torch_npu.npu_gru(input, hx, weight_input, weight_hidden, bias_input, bias_hidden, seq_length, has_biases, num_layers, dropout, train, bidirectional, batch_first) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)
```

## 功能说明

计算DynamicGRUV2。

## 参数说明

- input (Tensor) - 数据类型：float16；格式：FRACTAL_NZ。
- hx (Tensor) - 数据类型：float16，float32；格式：FRACTAL_NZ。
- weight_input (Tensor) - 数据类型：float16；格式：FRACTAL_Z。
- weight_hidden (Tensor) - 数据类型：float16；格式：FRACTAL_Z。
- bias_input (Tensor) - 数据类型：float16，float32；格式：ND。
- bias_hidden (Tensor) - 数据类型：float16，float32；格式：ND。
- seq_length (Tensor) - 数据类型：int32；格式：ND。
- has_biases (Bool，默认值为True)。
- num_layers (Int)。
- dropout (Float)。
- train (Bool，默认值为True) - 标识训练是否在op进行的bool参数。
- bidirectional (Bool，默认值为True)。
- batch_first (Bool，默认值为True)。

## 输出说明

- y (Tensor) - 数据类型：float16，float32；格式：FRACTAL_NZ。
- output_h (Tensor) - 数据类型：float16，float32；格式：FRACTAL_NZ。
- update (Tensor) - 数据类型：float16，float32；格式：FRACTAL_NZ。
- reset (Tensor) - 数据类型：float16，float32；格式：FRACTAL_NZ。
- new (Tensor) - 数据类型：float16，float32；格式：FRACTAL_NZ。
- hidden_new (Tensor) - 数据类型：float16，float32；格式：FRACTAL_NZ。

## 约束说明

接口暂不支持jit_compile=False，需要在该模式下使用时请将"DynamicGRUV2"添加至"NPU_FUZZY_COMPILE_BLACKLIST"选项内，具体操作可参考[添加二进制黑名单示例](添加二进制黑名单示例.md)。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>


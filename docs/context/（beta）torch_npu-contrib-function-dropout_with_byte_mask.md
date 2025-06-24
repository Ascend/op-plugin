# （beta）torch_npu.contrib.function.dropout_with_byte_mask

## 函数原型

```
torch_npu.contrib.function.dropout_with_byte_mask(input1, p=0.5, training=True, inplace=False)
```

## 功能说明

应用NPU兼容的dropout_with_byte_mask操作，仅支持NPU设备。这个dropout_with_byte_mask方法生成无状态随机uint8掩码，并根据掩码做dropout。

## 参数说明

- p：dropout概率，默认值为0.5。
- training：是否启动dropout，当设置为True时启动，False时不启动。默认值为True。
- inplace：是否原地生效，当设置为True时将原地修改入参包含的值。默认值为False。

## 约束说明

仅在设备32核场景下性能提升。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>


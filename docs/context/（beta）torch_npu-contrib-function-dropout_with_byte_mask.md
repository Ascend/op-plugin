# （beta）torch_npu.contrib.function.dropout_with_byte_mask

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

应用NPU兼容的dropout_with_byte_mask操作，仅支持NPU设备。此方法生成无状态随机uint8掩码，并根据该掩码执行dropout。

## 函数原型

```
torch_npu.contrib.function.dropout_with_byte_mask(input1, p=0.5, training=True, inplace=False)
```

## 参数说明

- **p** (`int`)：dropout概率，默认值为0.5。
- **training** (`bool`)：是否启动dropout，当设置为True时启动，False时不启动。默认值为True。
- **inplace** (`bool`)：是否原地生效，当设置为True时将原地修改入参包含的值。默认值为False。

## 约束说明

仅在设备32核场景下性能提升。


# （beta）torch_npu.npu.get_autocast_dtype

## 函数原型

```
torch_npu.npu.get_autocast_dtype()
```

## 功能说明

在amp场景获取设备支持的数据类型，该dtype由torch_npu.npu.set_autocast_dtype设置或者默认数据类型torch.float16。

## 输出说明

torch.dtype

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>


# （beta）torch_npu.npu.set_autocast_dtype

## 函数原型

```
torch_npu.npu.set_autocast_dtype(dtype)
```

## 功能说明

设置设备在AMP场景支持的数据类型。

## 参数说明

dtype：数据类型。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
torch_npu.npu.set_autocast_dtype(torch.float16)
```


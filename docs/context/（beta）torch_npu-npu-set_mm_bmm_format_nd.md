# （beta）torch_npu.npu.set_mm_bmm_format_nd

## 函数原型

```
torch_npu.npu.set_mm_bmm_format_nd(bool)
```

## 功能说明

设置线性module里面的mm和bmm算子是否用ND格式。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>>torch_npu.npu.set_mm_bmm_format_nd(True)
```


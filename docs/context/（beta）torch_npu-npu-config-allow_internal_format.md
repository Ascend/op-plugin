# （beta）torch_npu.npu.config.allow_internal_format

## 函数原型

```
torch_npu.npu.config.allow_internal_format = bool
```

## 功能说明

是否使用私有格式，设置为True时允许使用私有格式，设置为False时，不允许申请任何私有格式的tensor，避免了适配层出现私有格式流通。

## 参数说明

输入`bool`值。
- <term>Atlas 训练系列产品</term>/<term>Atlas A2 训练系列产品</term>/<term>Atlas 推理系列产品</term>默认值为`True`。
- <term>Atlas A3 训练系列产品</term>默认值为`False`。

## 返回值说明
无

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>>torch_npu.npu.config.allow_internal_format = False
```


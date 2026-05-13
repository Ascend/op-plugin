# get_trace_id

## 产品支持情况

| 产品                               | 是否支持 |
| ---------------------------------- | :------: |
| <term>Atlas A3 训练系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品</term> |    √     |
| <term>Atlas 训练系列产品</term>    |    √     |

## 功能说明

获取当前的trace_id。

## 函数原型

```python
get_trace_id(self)
```

## 参数说明

无

## 返回值说明

返回trace_id。

## 调用示例

该接口不直接调用，用于为set_custom_trace_id_callback接口提供trace_id，具体示例请参见[set_custom_trace_id_callback](set_custom_trace_id_callback.md)。

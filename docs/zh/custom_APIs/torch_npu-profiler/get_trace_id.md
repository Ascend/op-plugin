# get_trace_id

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910" id3 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id3 -->

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

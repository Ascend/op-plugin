# （beta）torch_npu.npu.Event().recorded_time()

## 函数原型

```
torch_npu.npu.Event().recorded_time() -> int
```

## 功能说明

获取NPU Event对象在设备上被记录的时间。

## 输入说明

成员函数使用时必须是NPU Event对象才能调用。

## 输出说明

输出被记录的时间，是一个无符号的整数（uint64）。

## 异常说明

“INTERNALError”- Event对象必须在获取记录时间戳之前被记录。

## 约束说明

Event对象在创建的时候，需要传入参数enable_timing=True。

## 支持的型号

- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>

## 调用示例

```python
import torch
import torch_npu
 
event = torch_npu.npu.Event(enable_timing=True)
event.record()
res = event.recorded_time()
```


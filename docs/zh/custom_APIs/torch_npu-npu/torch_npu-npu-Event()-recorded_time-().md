# （beta）torch_npu.npu.Event().recorded_time()

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |

## 功能说明

获取NPU Event对象在设备上被记录的时间。

## 函数原型

```
torch_npu.npu.Event().recorded_time() -> int
```

## 参数说明

该成员函数只能在NPU Event对象上调用。

## 返回值说明


-  **int**：输出被记录的时间，是一个无符号的整数（uint64），单位为微秒。

-  若返回“INTERNALError”，则表示Event对象必须在获取记录时间戳之前被记录。
## 约束说明

Event对象在创建的时候，需要传入参数enable_timing=True。
 
## 调用示例

```python
import torch
import torch_npu
 
event = torch_npu.npu.Event(enable_timing=True)
event.record()
res = event.recorded_time()
```


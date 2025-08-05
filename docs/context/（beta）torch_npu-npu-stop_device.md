# （beta）torch_npu.npu.stop_device

>**须知：**<br>
>本接口为预留接口，暂不支持。

## 函数原型

```
torch_npu.npu.stop_device(device_id: int) -> int 
```

## 功能说明

停止对应device上的计算，对于没有执行的计算进行清除，后续在此device上执行计算会报错。

## 参数说明

“device_id”(int)需要处理的device id。

## 输入说明

要确保是一个有效的device。

## 输出说明

返回值为int，代表执行结果，0表示执行成功，1表示执行失败。

## 支持的型号

- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>

## 调用示例

```python
import torch
import torch_npu  
torch.npu.set_device(0) 
torch_npu.npu.stop_device(0)
```


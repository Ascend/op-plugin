# （beta）torch_npu.npu.check_uce_in_memory

>**须知：**<br>
>本接口为预留接口，暂不支持。

## 函数原型

```
torch_npu.npu.check_uce_in_memory(device_id:int)
```

## 功能说明

提供故障内存地址类型检测接口，供MindCluster进行故障恢复策略的决策。其功能是在出现UCE片上内存故障时，判断故障内存地址类型。

>**注意：**<br>
>此API的功能实现依赖于PyTorch的内存管理机制，仅在PYTORCH_NO_NPU_MEMORY_CACHING未配置，即开启内存复用机制时，才可使用此API，若export PYTORCH_NO_NPU_MEMORY_CACHING=1，即关闭内存复用机制时，此API无法使用。

## 参数说明

device_id(int) ：需要处理的device id。

## 输入说明

要确保是一个有效的device。

## 输出说明

- 0：无UCE故障地址。
- 1：UCE故障地址为非Ascend Extension for PyTorch使用的内存地址。
- 2：UCE故障地址为Ascend Extension for PyTorch使用的临时内存地址。
- 3：UCE故障地址为Ascend Extension for PyTorch使用的常驻内存地址。

## 支持的型号

- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>

## 调用示例

```python
import torch,torch_npu
torch.npu.set_device(0)
torch_npu.npu.check_uce_in_memory(0)
```


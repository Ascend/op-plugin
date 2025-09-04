# torch_npu.npu.SyncLaunchStream

## 函数原型

```
torch_npu.npu.SyncLaunchStream(device)
```

## 功能说明

创建一条同步下发NPUStream，在该流上下发的任务不再使用taskqueue异步下发。在集群场景某一设备出现故障，其他设备保存checkpoint时，可使用此同步下发流保存。

## 参数说明

“device”(Any) – 可以为设备数字id或者字符串“npu:0”，默认值为“none”（即当前线程对应的设备id）。

## 输出说明

一条创建好的NPUStream，在该流上下发任务不再使用taskqueue异步下发。

## 约束说明

- 由于不再下发到taskqueue，因此该流的下发性能相比普通流有所降低，建议在集群训练时某些节点出现故障，其他节点保存ckpt时使用。
- 同步下发流资源池只有4条，创建超过4条时将会循环从资源池中获取。

## 支持的型号

- <term> Atlas 训练系列产品</term> 
- <term> Atlas A2 训练系列产品</term> 
- <term> Atlas A3 训练系列产品</term>

## 调用示例

```python
import torch
import torch_npu
s = torch_npu.npu.SyncLaunchStream()
with torch.npu.stream(s):
    tensor1 =torch.randn(4).npu()
    tensor2 = tensor1 + tensor1
    s.synchronize()
```


# （beta）torch_npu.contrib.Prefetcher

## 函数原型

```
torch_npu.contrib.Prefetcher(loader, stream=None)
```

## 功能说明

NPU设备上使用的预取程序。

## 参数说明

- loader (torch.utils.data.DataLoader or DataLoader like iterator)：用于在预处理后生成输入数据。
- stream (torch.npu.Stream)：默认值为None。由于NPU内存逻辑限制，如果要在训练中重复初始化prefetcher就需要指定一个stream来防止内存泄漏；如果prefetcher仅在训练中被初始化一次则无需指定stream。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>


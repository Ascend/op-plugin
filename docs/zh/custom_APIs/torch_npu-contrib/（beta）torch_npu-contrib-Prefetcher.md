# （beta）torch_npu.contrib.Prefetcher

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 函数原型

```python
torch_npu.contrib.Prefetcher(loader, stream=None)
```

## 功能说明

NPU设备上的数据预取器，主要用于优化数据加载流程，提升训练效率。

## 参数说明

- **loader** (torch.utils.data.DataLoader or DataLoader like iterator)：必选参数。预处理后的输入数据。
- **stream** (torch.npu.Stream)：可选参数，默认值为None。由于NPU内存逻辑限制，如果要在训练中重复初始化Prefetcher，就需要指定一个stream来防止内存泄漏；如果Prefetcher仅在训练中被初始化一次，则无需指定stream，会自动创建一个stream。

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> from torch_npu.contrib import Prefetcher
>>> # 创建DataLoader
>>> dataset = torch.utils.data.TensorDataset(torch.randn(100, 3, 224, 224), torch.randint(0, 10, (100,)))
>>> loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
>>> # 初始化Prefetcher（仅初始化一次，无需指定stream）
>>> prefetcher = Prefetcher(loader)
>>> # 迭代获取数据
>>> input, target = prefetcher.next()
>>> while input is not None:
...     # 对input和target进行训练操作
...     input, target = prefetcher.next()
>>> # 重复初始化Prefetcher时（如多epoch训练），需指定stream防止内存泄漏
>>> stream = torch.npu.Stream()
>>> for epoch in range(10):
...     prefetcher = Prefetcher(loader, stream=stream)
...     input, target = prefetcher.next()
...     while input is not None:
...         # 对input和target进行训练操作
...         input, target = prefetcher.next()

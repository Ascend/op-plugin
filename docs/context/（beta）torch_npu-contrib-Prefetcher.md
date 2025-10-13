# （beta）torch_npu.contrib.Prefetcher

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 函数原型

```
torch_npu.contrib.Prefetcher(loader, stream=None)
```

## 功能说明

NPU设备上的数据预取器，主要用于优化数据加载流程，提升训练效率。

## 参数说明

- **loader** (torch.utils.data.DataLoader or DataLoader like iterator)：必选参数。预处理后的输入数据。
- **stream** (torch.npu.Stream)：可选参数，默认值为None。由于NPU内存逻辑限制，如果要在训练中重复初始化prefetcher，就需要指定一个stream来防止内存泄漏；如果prefetcher仅在训练中被初始化一次，则无需指定stream，会自动创建一个stream。


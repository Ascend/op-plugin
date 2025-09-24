# （beta）torch_npu.contrib.module.NpuCachedDropout
## 产品支持情况


| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √    |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |
## 功能说明

在NPU设备上使用FairseqDropout。

## 函数原型

```
torch_npu.contrib.module.NpuCachedDropout(torch.nn.Dropout)
```

## 参数说明

- **p** (`float`)：元素归零的概率。
- **module_name** (`string`)：模型名称。


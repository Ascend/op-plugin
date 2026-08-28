# （beta）torch_npu.contrib.NpuFairseqDropout

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

在NPU设备上使用FairseqDropout。

## 函数原型

```python
torch_npu.contrib.NpuFairseqDropout(p, module_name=None)
```

## 参数说明

- **p** (`float`)：元素归零的概率。
- **module_name** (`str`)：可选参数，模块名称，用于标识当前Dropout所属的模块，默认为`None`。该参数仅作标识用途，不参与实际计算逻辑。

## 约束说明

不支持动态shape。

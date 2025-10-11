# （beta）torch_npu.contrib.module.NpuDropPath

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

使用NPU亲和写法替换swin_transformer.py中的原生Drop路径。丢弃每个样本（应用于residual blocks的主路径）的路径（随机深度）。

## 函数原型

```
torch_npu.contrib.module.NpuDropPath(drop_prob=None)
```

## 参数说明

### 计算参数
- **drop_prob** (`float`)：dropout概率。

### 计算输入
- **x** (`Tensor`)：应用dropout的输入张量。

## 返回值说明
`Tensor`

dropout的计算结果。

## 调用示例

```python
>>> import torch, torch_npu
>>> from torch_npu.contrib.module import NpuDropPath
>>> input1 = torch.randn(68, 5).npu()
>>> input1.requires_grad_(True)
>>> input2 = torch.randn(68, 5).npu()
>>> input2.requires_grad_(True)
>>> fast_drop_path = NpuDropPath(0).npu()
>>> output = input1 + fast_drop_path(input2)
>>> output.sum().backward()
```


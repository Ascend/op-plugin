# （beta）torch_npu.contrib.module.npu_modules.DropoutWithByteMask
## 产品支持情况


| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √    |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |
## 功能说明

应用NPU兼容的DropoutWithByteMask操作。

## 函数原型

```
torch_npu.contrib.module.npu_modules.DropoutWithByteMask(p=0.5, inplace=False, max_seed=2 ** 10 - 1)
```

## 参数说明

### 计算参数
- **p** (`float`)：元素归零的概率。默认值为0.5。
- **inplace** (`bool`)：如果设置为True，原地执行此操作。默认值为False。
- **max_seed**：预留参数，暂未使用。

### 计算输入
- **Input** (`Tensor`)：输入张量，可为任何shape。

## 返回值说明

`Tensor`

输出张量与输入张量的shape相同。


## 调用示例

```python
>>> import torch, torch_npu
>>> from torch_npu.contrib.module.npu_modules import DropoutWithByteMask
>>> m = DropoutWithByteMask(p=0.5)
>>> input = torch.randn(16, 16).npu()
>>> output = m(input)
>>> output.shape
torch.Size([16, 16])
```


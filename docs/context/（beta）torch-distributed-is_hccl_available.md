# （beta）torch.distributed.is_hccl_available
## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

判断HCCL通信后端是否可用，与torch.distributed.is_nccl_available类似，具体请参考[https://pytorch.org/docs/stable/distributed.html\#torch.distributed.is_nccl_available](https://pytorch.org/docs/stable/distributed.html#torch.distributed.is_nccl_available)。

## 函数原型
```
torch.distributed.is_hccl_available()
```

## 返回值说明
`bool`：True为可用，False为不可用。

## 调用示例

```python
import torch
import torch_npu

torch.distributed.is_hccl_available()

True
```
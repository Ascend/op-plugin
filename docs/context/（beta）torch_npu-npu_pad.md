# （beta）torch_npu.npu_pad
## 产品支持情况


| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √    |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |
## 功能说明

填充张量。

## 函数原型

```
torch_npu.npu_pad(input, paddings) -> Tensor
```


## 参数说明

- **input** (`Tensor`)：输入张量。
- **paddings** (`List[int]`)：数据类型为`int32`、`int64`。


## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> input = torch.tensor([[20, 20, 10, 10]], dtype=torch.float16).to("npu")
>>> paddings = [1, 1, 1, 1]
>>> output = torch_npu.npu_pad(input, paddings)
>>> output
tensor([[ 0.,  0.,  0.,  0.,  0.,  0.],
        [ 0., 20., 20., 10., 10.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.]], device='npu:0', dtype=torch.float16)
```


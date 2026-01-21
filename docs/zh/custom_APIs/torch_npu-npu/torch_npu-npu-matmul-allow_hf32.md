# torch_npu.npu.matmul.allow_hf32
## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |


## 功能说明

matmul类算子开启支持hf32类型能力。

`torch_npu.npu.matmul.allow_hf32`功能和调用方式与`torch.backends.cuda.matmul.allow_tf32`类似，`torch.backends.cuda.matmul.allow_tf32`的功能具体请参考[https://pytorch.org/docs/stable/backends.html\#torch.backends.cuda.matmul.allow_tf32](https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.matmul.allow_tf32)。

## 函数原型

```
torch_npu.npu.matmul.allow_hf32 = bool
```


## 参数说明

输入bool值，默认值False。

## 返回值说明
`bool`


## 调用示例

```python
>>>import torch
>>>import torch_npu
>>>torch_npu.npu.matmul.allow_hf32
False
>>>torch_npu.npu.matmul.allow_hf32=True
>>>torch_npu.npu.matmul.allow_hf32
True
>>>torch_npu.npu.matmul.allow_hf32=False
>>>torch_npu.npu.matmul.allow_hf32
False
```


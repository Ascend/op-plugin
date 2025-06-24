# torch_npu.npu.conv.allow_hf32

torch_npu.npu.conv.allow_hf32功能和调用方式与torch.backends.cudnn.allow_tf32类似，torch.backends.cudnn.allow_tf32的功能具体请参考[https://pytorch.org/docs/stable/backends.html\#torch.backends.cudnn.allow_tf32](https://pytorch.org/docs/stable/backends.html#torch.backends.cudnn.allow_tf32)。

torch_npu.npu.conv.allow_hf32的调用方式如下所示：

## 函数原型

```
torch_npu.npu.conv.allow_hf32 = bool
```

## 功能说明

conv类算子开启支持hf32类型能力。

## 参数说明

输入bool值，默认值True。

## 输出说明

bool类型。

## 支持的型号

- <term> Atlas 训练系列产品</term> 
- <term> Atlas A2 训练系列产品</term> 
- <term> Atlas A3 训练系列产品</term> 
- <term> Atlas 推理系列产品</term> 

## 调用示例

```python
>>>import torch
>>>import torch_npu
>>>torch_npu.npu.conv.allow_hf32
True
>>>torch_npu.npu.conv.allow_hf32=False
>>>torch_npu.npu.conv.allow_hf32
False
>>>torch_npu.npu.conv.allow_hf32=True
>>>torch_npu.npu.conv.allow_hf32
True
```


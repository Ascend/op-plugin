# （beta）torch\_npu.npu\_swiglu

##  产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 训练系列产品</term> | √   |

## 功能说明

-   API功能：Swish门控线性单元激活函数，实现张量`input`的swiglu计算。

-   计算公式：

     公式中$x$是输入参数`input`的Tensor。$dim$是切分维度，默认为-1。$A$和$B$是`input`沿`dim`维度切分的Tensor。A表示前半部分张量，B表示后半部分张量。
    $$
    outputs=swiglu(x,dim=-1)=swish(A)*B=A*sigmoid(A)*B
    $$

## 函数原型

```
torch_npu.npu_swiglu(Tensor input, int dim=-1) -> (Tensor)
```

## 参数说明

**input** (`Tensor`)：必选参数，表示待计算的数据，对应公式中的$x$。shape支持1-8维，且shape必须在入参`dim`对应维度上可以整除2。不支持非连续的Tensor，不支持空Tensor。数据类型支持`float32`、`float16`或`bfloat16`类型。

**dim** (`int`)：可选参数，默认为-1。需要进行切分的维度序号，对`input`相应轴进行对半切。取值范围为\[-input.dim\(\), input.dim\(\)-1\]。

## 返回值说明
`Tensor`

对应公式中的$outputs$。数据类型与计算输入`input`的类型一致，不支持非连续的Tensor。

## 调用示例

```python
import torch
import torch_npu
input_tensor = torch.randn(2, 32, 6, 6)
output = torch_npu.npu_swiglu(input_tensor.npu(), dim = -1)
```


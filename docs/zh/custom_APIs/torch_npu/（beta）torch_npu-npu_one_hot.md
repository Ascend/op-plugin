# （beta）torch_npu.npu_one_hot
## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √    |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |
## 功能说明

返回一个one-hot张量。input中index表示的位置采用on_value值，而其他所有位置采用off_value的值。

## 函数原型

```
torch_npu.npu_one_hot(input, num_classes=-1, depth=1, on_value=1, off_value=0) -> Tensor
```


## 参数说明

- **input** (`Tensor`)：任何shape的class值。
- **num_classes** (`int`)：待填充的轴，默认值为-1。
- **depth** (`int`)：one_hot维度的深度，默认值为1。
- **on_value** (`Scalar`)：当indices[j] == i时输出中的填充值，默认值为1。
- **off_value** (`Scalar`)：当indices[j] != i时输出中的填充值，默认值为0。


## 调用示例

```python
>>> a=torch.IntTensor([5, 3, 2, 1]).npu()
>>> b=torch_npu.npu_one_hot(a, depth=5)
>>> b
tensor([[0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0.],
        [0., 0., 1., 0., 0.],
        [0., 1., 0., 0., 0.]], device='npu:0')
```

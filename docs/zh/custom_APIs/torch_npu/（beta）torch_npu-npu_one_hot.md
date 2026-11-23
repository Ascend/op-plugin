# （beta）torch_npu.npu_one_hot

> [!NOTICE]  
> 该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="310p" id3 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id4 -->

## 功能说明

返回一个one-hot张量。input中每个元素的值v作为类别索引：输出在one_hot维（深度为`depth`）索引v处填充`on_value`，其余位置填充`off_value`。input应为整数索引张量，元素取值建议在$[0, depth)$范围内，越界的元素对应的输出全部填充`off_value`。

## 函数原型

```python
torch_npu.npu_one_hot(input, num_classes=-1, depth=1, on_value=1, off_value=0) -> Tensor
```

## 参数说明

- **input** (`Tensor`)：必选参数，类别索引张量，建议传入整数类型。接口会先将其转换为`torch.int32`。转换后的索引落在`[0, depth)`内时，对应位置填`on_value`，其余位置填`off_value`。索引越界时，对应的整段编码均填`off_value`。
- **num_classes** (`int`)：编码轴参数，默认值为`-1`，表示在输入末尾追加一维。使用本页所列产品时建议保留默认值。编码维度的长度由`depth`指定。
- **depth** (`int`)：one_hot编码的长度，默认值为`1`。需要其他长度时请直接传入类别数量。
- **on_value** (`Scalar`)：当输出one_hot维的索引i等于input对应位置的值时，输出在该位置填充的值，默认值为1。
- **off_value** (`Scalar`)：当输出one_hot维的索引i不等于input对应位置的值时，输出在该位置填充的值，默认值为0。

## 调用示例

```python
>>> a=torch.IntTensor([5, 3, 2, 1]).npu()
>>> b=torch_npu.npu_one_hot(a, depth=5)
>>> print(b)
tensor([[0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0.],
        [0., 0., 1., 0., 0.],
        [0., 1., 0., 0., 0.]], device='npu:0')
```

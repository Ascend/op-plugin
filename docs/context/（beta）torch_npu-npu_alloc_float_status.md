# （beta）torch_npu.npu_alloc_float_status

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √    |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |
## 功能说明

申请一个专门用于存储浮点运算状态标志的Tensor。该Tensor用于后续记录计算过程中的溢出状态。


## 函数原型

```
torch_npu.npu_alloc_float_status(input) -> Tensor
```


## 参数说明

**input** (`Tensor`)：必选参数，任意构建的一个NPU张量（主要用于确定device信息）。

## 返回值说明

`Tensor`

一个包含8个float32类型全零值的Tensor。

## 调用示例

```python
>>> input = torch.randn([1,2,3]).npu()
## 分配状态空间
>>> output = torch_npu.npu_alloc_float_status(input)
>>> input
tensor([[[ 2.2324,  0.2478, -0.1056],
        [ 1.1273, -0.2573,  1.0558]]], device='npu:0')
>>> output
tensor([0., 0., 0., 0., 0., 0., 0., 0.], device='npu:0')

## 清除状态
>>> torch_npu.npu_clear_float_status(output)

## 执行可能溢出的计算操作
## ...模型前向/反向传播...

## 获取检测结果
>>> result = torch_npu.npu_get_float_status(status)

```


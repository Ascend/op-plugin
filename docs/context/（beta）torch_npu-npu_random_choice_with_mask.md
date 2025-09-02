# （beta）torch_npu.npu_random_choice_with_mask
## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √    |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |
## 功能说明

混洗非零元素的index。

## 函数原型

```
torch_npu.npu_random_choice_with_mask(x, count=256, seed=0, seed2=0) -> (Tensor, Tensor)
```


## 参数说明

- **x** (`Tensor`) - 输入张量。
- **count** (`int`，默认值为256) - 输出计数。如果值为0，则输出所有非零元素。
- **seed** (`int`，默认值为0) - 数据类型：int32，int64。
- **seed2** (`int`，默认值为2) - 数据类型：int32，int64。

## 返回值说明

- **y** (`Tensor`) - 2D张量，非零元素的index。
- **mask** (`Tensor`) - 1D张量，确定对应index是否有效。


## 调用示例

```python
>>> x = torch.tensor([1, 0, 1, 0], dtype=torch.bool).to("npu")
>>> result, mask = torch_npu.npu_random_choice_with_mask(x, 2, 1, 0)
>>> result
tensor([[0],[2]], device='npu:0', dtype=torch.int32)
>>> mask
tensor([True, True], device='npu:0')
```

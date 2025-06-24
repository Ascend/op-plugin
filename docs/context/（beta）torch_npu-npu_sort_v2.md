# （beta）torch_npu.npu_sort_v2

>**须知：**<br>
>该接口计划废弃，可以使用torch.sort接口进行替换。

## 函数原型

```
torch_npu.npu_sort_v2(self, dim=-1, descending=False, out=None) -> Tensor
```

## 功能说明

沿给定维度，按无index值对输入张量元素进行升序排序。若dim未设置，则选择输入的最后一个维度。如果descending为True，则元素将按值降序排序。

## 参数说明

- self (Tensor) - 输入张量。
- dim (Int，可选，默认值为-1) - 进行排序的维度。
- descending (Bool，可选，默认值为None) - 排序顺序控制（升序或降序）。

## 约束说明

目前仅支持输入的最后一个维度（dim=-1）。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>
## 调用示例

```python
>>> x = torch.randn(3, 4).npu()
>>> x
tensor([[-0.0067,  1.7790,  0.5031, -1.7217],
        [ 1.1685, -1.0486, -0.2938,  1.3241],
        [ 0.1880, -2.7447,  1.3976,  0.7380]], device='npu:0')
>>> sorted_x = torch_npu.npu_sort_v2(x)
>>> sorted_x
tensor([[-1.7217, -0.0067,  0.5029,  1.7793],
        [-1.0488, -0.2937,  1.1689,  1.3242],
        [-2.7441,  0.1880,  0.7378,  1.3975]], device='npu:0')
```


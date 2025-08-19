# （beta）torch_npu.npu_bmmV2

>**须知：**<br>
>该接口计划废弃，可以使用`torch.bmm`和`torch.view`接口进行替换。

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>           |    √     |
|<term>Atlas A2 训练系列产品</term> | √   |
|<term>Atlas 训练系列产品</term> | √   |
|<term>Atlas 推理系列产品</term>| √   |

## 功能说明

将矩阵“a”乘以矩阵“b”，生成“a*b”。支持FakeTensor模式。

## 函数原型

```
torch_npu.npu_bmmV2(self, mat2, output_sizes) -> Tensor
```

## 参数说明

- **self**（`Tensor`）：2D或更高维度矩阵张量。数据类型支持`float16`、`float32`、`int32`。格式支持$[ND, NHWC, FRACTAL_NZ]$。
- **mat2**（`Tensor`）：2D或更高维度矩阵张量。数据类型支持`float16`、`float32`、`int32`。格式支持$[ND, NHWC, FRACTAL_NZ]$。
- **output_sizes**（`List[int]`）：默认值为[]，输出的shape，用于matmul的反向传播。

## 调用示例

示例一：

```python
>>> mat1 = torch.randn(10, 3, 4).npu()
>>> mat2 = torch.randn(10, 4, 5).npu()
>>> res = torch_npu.npu_bmmV2(mat1, mat2, [])
>>> res.shape
torch.Size([10, 3, 5])
```

示例二：

```python
//FakeTensor模式
>>> from torch._subclasses.fake_tensor import FakeTensorMode
>>> with FakeTensorMode():
...     mat1 = torch.randn(10, 3, 4).npu()
...     mat2 = torch.randn(10, 4, 5).npu()
...     result = torch_npu.npu_bmmV2(mat1, mat2, [])
...
>>> result
FakeTensor(..., device='npu:0', size=(10, 3, 5))
```


# （beta）torch_npu.npu_slice

## 函数原型

```
torch_npu.npu_slice(self, offsets, size) -> Tensor
```

## 功能说明

从张量中提取切片。

>**注意：**<br>
>该接口不支持反向计算。

## 参数说明

- self (Tensor) - 输入张量。
- offsets (ListInt) - 数据类型：int32，int64。
- size (ListInt) - 数据类型：int32，int64。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> input = torch.tensor([[1,2,3,4,5], [6,7,8,9,10]], dtype=torch.float16).to("npu")
>>> offsets = [0, 0]
>>> size = [2, 2]
>>> output = torch_npu.npu_slice(input, offsets, size)
>>> output
tensor([[1., 2.],
        [6., 7.]], device='npu:0', dtype=torch.float16)
```


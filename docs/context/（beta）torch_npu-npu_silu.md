# （beta）torch_npu.npu_silu

>**须知：**<br>
>该接口计划废弃，可以使用torch.nn.functional.silu接口进行替换。

## 函数原型

```
torch_npu.npu_silu(self) -> Tensor
```

## 功能说明

计算self的Swish。

## 参数说明

self (Tensor) - 数据类型：float16、float32。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> a=torch.rand(2,8).npu()
>>> output = torch_npu.npu_silu(a)
>>> output
tensor([[0.4397, 0.7178, 0.5190, 0.2654, 0.2230, 0.2674, 0.6051, 0.3522],
        [0.4679, 0.1764, 0.6650, 0.3175, 0.0530, 0.4787, 0.5621, 0.4026]],
       device='npu:0')
```


# （beta）torch_npu.npu_apply_adam
>**须知：**<br>
>该接口计划废弃，可以使用`torch.optim.Adam`接口进行替换。

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>           |    √     |
|<term>Atlas A2 训练系列产品</term> | √   |
|<term>Atlas 训练系列产品</term> | √   |
|<term>Atlas 推理系列产品</term>| √   |

## 功能说明

获取adam优化器的计算结果。

## 函数原型

```
torch_npu.npu_apply_adam(beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, use_locking, use_nesterov, out = (var, m, v))
```

## 参数说明

- **beta1_power**（`Scalar`）：beta1的幂。
- **beta2_power**（`Scalar`）：beta2的幂。
- **lr**（`Scalar`）：学习率。
- **beta1**（`Scalar`）：一阶矩估计值的指数衰减率。
- **beta2**（`Scalar`）：二阶矩估计值的指数衰减率。
- **epsilon**（`Scalar`）：添加到分母中以提高数值稳定性的项数。
- **grad**（`Tensor`）：梯度。
- **use_locking**（`bool`）：可选参数，设置为True时使用lock进行更新操作。
- **use_nesterov**（`bool`）：可选参数，设置为True时采用nesterov更新。
- **var**（`Tensor`）：待优化变量。
- **m**（`Tensor`）：变量平均值。
- **v**（`Tensor`）：变量方差。

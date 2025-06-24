# （beta）torch\_npu.npu\_apply\_adam
>**须知：**<br>
>该接口计划废弃，可以使用torch.optim.Adam或torch.optim.adam接口进行替换。

## 函数原型

```
torch_npu.npu_apply_adam(beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, use_locking, use_nesterov, out = (var, m, v))
```

## 功能说明

获取adam优化器的计算结果。

## 参数说明

-   beta1\_power \(Scalar\) - beta1的幂。
-   beta2\_power \(Scalar\) - beta2的幂。
-   lr \(Scalar\)  - 学习率。
-   beta1 \(Scalar\) - 一阶矩估计值的指数衰减率。
-   beta2 \(Scalar\) - 二阶矩估计值的指数衰减率。
-   epsilon \(Scalar\) - 添加到分母中以提高数值稳定性的项数。
-   grad \(Tensor\) - 梯度。
-   use\_locking \(Bool，可选\) - 设置为True时使用lock进行更新操作。
-   use\_nesterov \(Bool，可选\) - 设置为True时采用nesterov更新。
-   var \(Tensor\) - 待优化变量。
-   m \(Tensor\) - 变量平均值。
-   v \(Tensor\) - 变量方差。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>


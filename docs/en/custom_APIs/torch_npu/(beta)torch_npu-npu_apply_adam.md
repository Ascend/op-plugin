# (beta) torch_npu.npu_apply_adam
>
> [!NOTICE]  
> This API is planned for deprecation. Use `torch.optim.Adam` instead.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>          |    √     |
|<term>Atlas A2 training products</term>| √   |
|<term>Atlas training products</term>| √   |
|<term>Atlas inference products</term>| √   |

## Function

Obtains the computation results of the Adam optimizer.

## Prototype

```python
torch_npu.npu_apply_adam(beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, use_locking, use_nesterov, out = (var, m, v))
```

## Parameters

- **`beta1_power`** (`Scalar`): Power of `beta1`.
- **`beta2_power`** (`Scalar`): Power of `beta2`.
- **`lr`** (`Scalar`): Learning rate.
- **`beta1`** (`Scalar`): Exponential decay rate for the first-moment estimates.
- **`beta2`** (`Scalar`): Exponential decay rate for the second-moment estimates.
- **`epsilon`** (`Scalar`): Term added to the denominator to improve numerical stability.
- **`grad`** (`Tensor`): Gradient tensor.
- **`use_locking`** (`bool`): Optional. If set to `True`, locking is used during update operations.
- **`use_nesterov`** (`bool`): Optional. If set to `True`, Nesterov updates are used.
- **`var`** (`Tensor`): Variable to be optimized.
- **`m`** (`Tensor`): First-moment estimate of the variable.
- **`v`** (`Tensor`): Second-moment estimate of the variable.

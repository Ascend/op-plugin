# (beta) torch_npu.npu_bert_apply_adam

> [!NOTICE]  
> This API is planned for deprecation. The underlying operator kernel is no longer maintained, and performance and accuracy are not guaranteed. This API is not recommended.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √    |
|<term>Atlas training products</term>                                      |    √     |

## Function

Obtains the computation results of the Adam optimizer for the BERT model.

## Prototype

```python
torch_npu.npu_bert_apply_adam(lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm, weight_decay, step_size=None, adam_mode=0, *, var,m,v)
```

## Parameters

- **`var`** (`Tensor`): The data type can be `float16` or `float32`.
- **`m`** (`Tensor`): The data type and shape must be identical to those of `var`.
- **`v`** (`Tensor`): The data type and shape must be identical to those of `var`.
- **`lr`** (`Scalar`): The data type must be identical to that of `var`. The shape is `(1,)`.
- **`beta1`** (`Scalar`): The data type must be identical to that of `var`. The shape is `(1,)`.
- **`beta2`** (`Scalar`): The data type must be identical to that of `var`. The shape is `(1,)`.
- **`epsilon`** (`Scalar`): The data type must be identical to that of `var`. The shape is `(1,)`.
- **`grad`** (`Tensor`): The data type and shape must be identical to those of `var`.
- **`max_grad_norm`** (`Scalar`): The data type must be identical to that of `var`. The shape is `(1,)`.
- **`global_grad_norm`** (`Scalar`): The data type must be identical to that of `var`. The shape is `(1,)`.
- **`weight_decay`** (`Scalar`): The data type must be identical to that of `var`. The shape is `(1,)`.
- **`step_size`** (`Tensor`): Optional. The data type must be identical to that of `var`. The shape is `(1,)`. The default value is `None`.
- **`adam_mode`** (`int`): Selects the Adam mode. Valid values are `0` (`"adam"`) or `1` (`"mbert_adam"`). The default value is `0`.

## Example

```python
>>> import torch
>>> import torch_npu
>>> var_in = torch.rand(321538).uniform_(-32., 21.).npu()
>>> m_in = torch.zeros(321538).npu()
>>> v_in = torch.zeros(321538).npu()
>>> grad = torch.rand(321538).uniform_(-0.05, 0.03).npu()
>>> max_grad_norm = -1.
>>> beta1 = 0.9
>>> beta2 = 0.99
>>> weight_decay = 0.
>>> lr = 0.
>>> epsilon = 1e-06
>>> global_grad_norm = 0.
>>> var_out, m_out, v_out = torch_npu.npu_bert_apply_adam(lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm, weight_decay, out=(var_in, m_in, v_in))
>>> print(var_out)
tensor([ 14.7733, -30.1218,  -1.3647,  ..., -16.6840,   7.1518,   8.4872],
      device='npu:0')
```

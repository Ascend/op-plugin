# torch_npu.npu_apply_adam_w

> [!NOTICE]  
> This API is new in this version. For details about the dependency requirements, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->

## Function

- API function: Implements parameter update computation for the AdamW optimizer. AdamW decouples weight decay from the gradient update on top of Adam and applies it directly to the parameters to be optimized. It is commonly used for parameter optimization in large-model training. This API updates the first moment `m` and the second moment `v` while updating the parameters `var` to be optimized in place. It optionally supports AMSGrad correction and maximize (negating the gradient to maximize the objective).
- Formulas:

  Let the parameters to be optimized be $var$, the first moment be $m$, the second moment be $v$, and the current gradient be $grad$. The intermediate gradient $g_t$ is:

  $$
  g_t = \begin{cases} -grad, & maximize = True \\ grad, & maximize = False \end{cases}
  $$

  Update of the first and second moments:

  $$
  m_{out} = \beta_1 \cdot m + (1 - \beta_1) \cdot g_t
  $$

  $$
  v_{out} = \beta_2 \cdot v + (1 - \beta_2) \cdot g_t^2
  $$

  Denominator $denom$ (whether the `max_grad_norm` correction is introduced is determined by `amsgrad`):

  $$
  denom = \begin{cases} \sqrt{\dfrac{\max(max\_grad\_norm,\ v_{out})}{1 - \beta_2^{power} \cdot \beta_2}} + \epsilon, & amsgrad = True \\[2ex] \sqrt{\dfrac{v_{out}}{1 - \beta_2^{power} \cdot \beta_2}} + \epsilon, & amsgrad = False \end{cases}
  $$

  Update of the parameters to be optimized:

  $$
  var_{out} = var \cdot (1 - lr \cdot weight\_decay) - lr \cdot \dfrac{m_{out} / (1 - \beta_1^{power} \cdot \beta_1)}{denom}
  $$

## Prototype

```python
torch_npu.npu_apply_adam_w(beta1_power, beta2_power, lr, weight_decay, beta1, beta2, epsilon, grad, max_grad_norm=None, amsgrad=None, maximize=None, out=(var, m, v)) -> (Tensor, Tensor, Tensor)
```

## Parameters

- **`beta1_power`** (`Scalar`): Required. Power of the exponential decay rate `beta1` of the first moment, corresponding to $\beta_1^{power}$ in the formulas. The data type can be `torch.float32`, `torch.float16`, or `torch.bfloat16`.
- **`beta2_power`** (`Scalar`): Required. Power of the exponential decay rate `beta2` of the second moment, corresponding to $\beta_2^{power}$ in the formulas. The data type can be `torch.float32`, `torch.float16`, or `torch.bfloat16`.
- **`lr`** (`Scalar`): Required. Learning rate, corresponding to $lr$ in the formulas. The data type can be `torch.float32`, `torch.float16`, or `torch.bfloat16`.
- **`weight_decay`** (`Scalar`): Required. Weight decay coefficient, corresponding to $weight\_decay$ in the formulas. The data type can be `torch.float32`, `torch.float16`, or `torch.bfloat16`.
- **`beta1`** (`Scalar`): Required. Exponential decay rate of the first moment estimate, corresponding to $\beta_1$ in the formulas. The data type can be `torch.float32`, `torch.float16`, or `torch.bfloat16`.
- **`beta2`** (`Scalar`): Required. Exponential decay rate of the second moment estimate, corresponding to $\beta_2$ in the formulas. The data type can be `torch.float32`, `torch.float16`, or `torch.bfloat16`.
- **`epsilon`** (`Scalar`): Required. Term added to the denominator to improve numerical stability, corresponding to $\epsilon$ in the formulas. The data type can be `torch.float32`, `torch.float16`, or `torch.bfloat16`.
- **`grad`** (`Tensor`): Required. Current gradient, corresponding to $grad$ in the formulas. The data layout can be ND. The data type can be `torch.float32`, `torch.float16`, or `torch.bfloat16`.
- **`max_grad_norm`** (`Tensor`): Optional. Maximum historical second moment required for AMSGrad correction, corresponding to $max\_grad\_norm$ in the formulas. The data layout can be ND. The shape and data type must be the same as those of `grad`. The data type can be `torch.float32`, `torch.float16`, or `torch.bfloat16`. The default value is `None`. When `amsgrad` is `True`, this parameter must be passed.
- **`amsgrad`** (`bool`): Optional. Whether to use AMSGrad correction. The default value is `None` (equivalent to `False`). The current version supports only `False`.
- **`maximize`** (`bool`): Optional. Whether to maximize the optimization objective (equivalent to negating the gradient `grad`). The default value is `None` (equivalent to `False`).
- **`out`** (`tuple(Tensor, Tensor, Tensor)`): Required. Specifies the output Tensors `(var, m, v)`. The computation results are written back to these Tensors in place:
  - **`var`** (`Tensor`): Parameters to be optimized, corresponding to $var$ in the formulas. The data layout can be ND. The data type can be `torch.float32`, `torch.float16`, or `torch.bfloat16`.
  - **`m`** (`Tensor`): First moment, corresponding to $m$ in the formulas. The shape and data type are the same as those of `var`.
  - **`v`** (`Tensor`): Second moment, corresponding to $v$ in the formulas. The shape and data type are the same as those of `var`.

## Return Values

- **`var`** (`Tensor`): Updated parameters to be optimized, corresponding to $var_{out}$ in the formulas. The data layout can be ND. The shape and data type are the same as those of the input `var`.
- **`m`** (`Tensor`): Updated first moment, corresponding to $m_{out}$ in the formulas. The shape and data type are the same as those of the input `m`.
- **`v`** (`Tensor`): Updated second moment, corresponding to $v_{out}$ in the formulas. The shape and data type are the same as those of the input `v`.

> [!NOTICE]
> The returned `(var, m, v)` are the same objects as the three Tensors passed in `out`. The computation results are updated in place.

## Constraints

- The shapes of `var`, `m`, `v`, and `grad` must be the same.
- When `amsgrad` is `True`, `max_grad_norm` must be passed; otherwise, the API reports an error. In the current version, `amsgrad` supports only `False`.
- This API supports only training scenarios.
- This API supports only Eager mode.

## Examples

```python
import torch
import torch_npu

shape = (21130, 512)
var = torch.rand(shape, dtype=torch.float32).npu() * 10 + 10
m = torch.rand(shape, dtype=torch.float32).npu() * 5 + 5
v = torch.rand(shape, dtype=torch.float32).npu() * 5 + 0.1
grad = (torch.rand(shape, dtype=torch.float32).npu() - 0.5) * 10

beta1_power = 0.9
beta2_power = 0.999
lr = 0.01
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
max_grad_norm = None
amsgrad = False
maximize = False

var_out, m_out, v_out = torch_npu.npu_apply_adam_w(
    beta1_power, beta2_power, lr, weight_decay, beta1, beta2, epsilon,
    grad, max_grad_norm, amsgrad, maximize,
    out=(var, m, v),
)

print("var_out:", var_out)
print("m_out:", m_out)
print("v_out:", v_out)
```

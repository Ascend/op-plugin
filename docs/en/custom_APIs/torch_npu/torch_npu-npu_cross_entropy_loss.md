# torch_npu.npu_cross_entropy_loss

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |

## Function

- Description: Computes the cross-entropy loss between the input `input` and label `target`. This API fuses the `log_softmax` and `nll_loss` operations from the native `CrossEntropyLoss` framework to reduce memory utilization during computation.
- Formulas:

    $x$ represents the input `input`, $y$ represents the label `target`, `weight` represents the weight, $C$ represents the total number of labels, and $N$ represents the batch size.

    Formula for cross entropy loss calculation:
    $$
    loss=\begin{cases}\sum_{n=1}^N\frac{1}{\sum_{n=1}^Nweight_{y_n}*1\{y_n\ !=\ ignoreIndex \}}l_n,&\text{if reduction = 'mean'} \\\sum_{n=1}^Nl_n,&\text {if reduction = 'sum' }\\\{l_0,l_1,...,l_n\},&\text{if reduction = 'None' }\end{cases}
    $$
    
    Formula for $l_n$ calculation:
    $$
    l_n = -weight_{y_n}*log\frac{exp(x_{n,y_n})}{\sum_{c=1}^Cexp(x_{n,c})}*1\{y_n\ !=\ ignoreIndex \}
    $$
  
    Formula for the log probability `log_prob` of the n-th sample for the c-th class:
    $$
    lse_n = log*\sum_{c=1}^{C}exp(x_{n,c})
    $$

    $$
    logProb_{n,c} = x_{n,c} - lse_n
    $$

## Prototype

```python
torch_npu.npu_cross_entropy_loss(input, target, weight=None, reduction="mean", ignore_index=-100, label_smoothing=0.0, lse_square_scale_for_zloss=0.0, return_zloss=False) -> (Tensor, Tensor, Tensor, Tensor)
```

## Parameters

- **`input`** (`Tensor`): Required. Input tensor, $x$ in the formulas. The data type can be `float16`, `float32`, or `bfloat16`. The shape of this parameter is `[N, C]`, where $N$ is the batch size and $C$ is the number of labels ($C > 0$).
- **`target`** (`Tensor`): Required. Label tensor, $y$ in the formulas. The data type can be `int64`. The shape of this parameter is `[N]`, which must match the zeroth dimension of `input`. The value range is [0, C).
- **`weight`** (`Tensor`): Optional. Scaling weight assigned to each class. The data type can be `float32`. The shape of this parameter is `[C]`, which must match the second dimension of `input`. The value range is (0, 1]. If no value is provided, it defaults to all ones.
- **`reduction`** (`str`): Optional. Reduction method for loss calculation. Valid values are `"mean"` (enables mean reduction), `"sum"` (enables sum reduction), or `"none"` (applies no reduction). The default value is `"mean"`.
- **`ignore_index`** (`int`): Optional. Label to be ignored during computation. The value must be less than $C$. A value less than 0 indicates that no ignore label is specified. The default value is `-100`.
- **`label_smoothing`** (`float`): Optional. Smoothing amount used when calculating the loss. The value range is [0.0, 1.0). The default value is `0.0`.
- **`lse_square_scale_for_zloss`** (`float`): Optional. Scale factor required for z-loss computation. The value range is [0.0, 1.0). The default value is `0.0`. Currently, this parameter is not supported.
- **`return_zloss`** (`bool`): Optional. Controls whether to return the auxiliary z-loss. Valid values are `True` (returns z-loss) or `False` (does not return z-loss). The default value is `False`. Currently, this parameter is not supported.

## Return Values

- **`loss`** (`Tensor`): Output loss tensor. The data type must be identical to that of `input`. When `reduction` is set to `"none"`, the shape must be `[N]`, matching the zeroth dimension of `input`. Otherwise, the shape must be `[1]`.
- **`log_prob`** (`Tensor`): Output tensor passed to backward computation. The data type must be identical to that of `input`. The shape of this parameter is `[N, C]`, which must be identical to that of `input`.
- **`zloss`** (`Tensor`): Auxiliary loss tensor. The data type must be identical to that of `input`. The shape must be identical to that of `loss`. This tensor is output only when `return_zloss` is set to `True`. Otherwise, an empty tensor is returned. Currently, this parameter is not supported.
- **`lse_for_zloss`** (`Tensor`): Output tensor passed to backward computation in z-loss scenarios. The data type must be identical to that of `input`. The shape of this parameter is `[N]`, matching the zeroth dimension of `input`. This tensor is returned only when `lse_square_scale_for_zloss` is not `0.0`. Otherwise, an empty tensor is returned. Currently, this parameter is not supported.

## Constraints

- The value range of `N` in the input shape is (0, 200000].
- When `input.requires_grad=True`, modifying the default value of `label_smoothing` is not supported in `"sum"` or `"none"` mode. In `"mean"` mode, only the default values of the optional parameters (including `weight`, `ignore_index`, and `label_smoothing`) are supported.
- The input parameters `lse_square_scale_for_zloss` and `return_zloss` are currently not enabled.
- The output tensors `zloss` and `lse_for_zloss` are currently not enabled.
- Only the `loss` output tensor supports gradient computation.

## Examples

- When reduction is set to `mean`:

    ```python
    import torch
    import torch_npu

    N = 4096 # Batch size
    C = 8080 # Number of labels

    # Construct inputs and labels
    input = torch.randn(N, C, dtype=torch.float32, requires_grad=True).npu()
    target = torch.arange(0, N, dtype=torch.int64).npu()

    # Call the NPU cross entropy loss function. When input.requires_grad=True is set to true, only the default values of the optional parameters can be transferred in mean (default) mode.
    loss, log_prob,_ , _ = torch_npu.npu_cross_entropy_loss(input, target)

    loss.backward()
    ```
    
- When reduction is set to `sum`:

    ```python
    import torch
    import torch_npu

    N = 4096 # Batch size
    C = 8080 # Number of labels

    # Construct inputs and labels
    input = torch.randn(N, C, dtype=torch.float32, requires_grad=True).npu()
    target = torch.arange(0, N, dtype=torch.int64).npu()

    # Call the NPU cross entropy loss function. When input.requires_grad=True is set to true, the default value of label_smoothing cannot be changed in sum mode.
    loss, log_prob,_ , _ = torch_npu.npu_cross_entropy_loss(input, target, reduction="sum", ignore_index=100)

    loss.backward()
    ```

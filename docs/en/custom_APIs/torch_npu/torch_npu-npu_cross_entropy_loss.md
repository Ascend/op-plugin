# torch_npu.npu_cross_entropy_loss

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id3 -->

## Function

- Function: computes the cross-entropy loss between the input `input` and the labels `target`. This API fuses log_softmax and nll_loss in the native `CrossEntropyLoss`, reducing the memory used during computation.
- Formulas:

    In the formulas, *x* is the input `input`, *y* is the label `target`, *weight* is the weight, *C* is the number of classes, and *N* is the batch size.

    The formula for computing the cross-entropy loss `loss` is as follows:
    $$
    loss=\begin{cases}\sum_{n=1}^N\frac{1}{\sum_{n=1}^Nweight_{y_n}*1\{y_n\ !=\ ignoreIndex \}}l_n,&\text{if reduction = 'mean'} \\\sum_{n=1}^Nl_n,&\text {if reduction = 'sum' }\\\{l_0,l_1,...,l_n\},&\text{if reduction = 'none' }\end{cases}
    $$
    
    The formula for computing $l_n$ is as follows:
    $$
    l_n = -weight_{y_n}*log\frac{exp(x_{n,y_n})}{\sum_{c=1}^Cexp(x_{n,c})}*1\{y_n\ !=\ ignoreIndex \}
    $$
  
    The formula for computing the log probability `log_prob` of the *n*-th sample for the *c*-th class is as follows:
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

- **`input`** (`Tensor`): Required. Input tensor, $x$ in the formulas. The data type can be `torch.float16`, `torch.float32`, or `torch.bfloat16`. The shape of this parameter is `[N, C]`, where $N$ is the batch size and $C$ is the number of labels ($C > 0$).
- **`target`** (`Tensor`): Required. Label tensor, $y$ in the formulas. The data type can be `torch.int64`. The shape of this parameter is `[N]`, which must match the zeroth dimension of `input`. The value range is [0, C).
- **`weight`** (`Tensor`): Optional. Scaling weight assigned to each class. The data type can be `torch.float32`. The shape of this parameter is `[C]`, which must match the second dimension of `input`. The value range is (0, 1]. If no value is provided, it defaults to all ones.
- **`reduction`** (`str`): Optional. Reduction method for loss calculation. Valid values are `"mean"` (enables mean reduction), `"sum"` (enables sum reduction), or `"none"` (applies no reduction). The default value is `"mean"`.
- **`ignore_index`** (`int`): Optional. Label to be ignored during computation. The value must be less than $C$. A value less than 0 indicates that no ignore label is specified. The default value is `-100`.
- **`label_smoothing`** (`float`): Optional. Smoothing amount used when calculating the loss. The value range is [0.0, 1.0). The default value is `0.0`.
- **`lse_square_scale_for_zloss`** (`float`): Optional. Scale factor required for z-loss computation. The value range is [0.0, 1.0). The default value is `0.0`. Currently, this parameter is not supported.
- **`return_zloss`** (`bool`): Optional. Controls whether to return the auxiliary z-loss. Valid values are `True` (returns z-loss) or `False` (does not return z-loss). The default value is `False`. Currently, this parameter is not supported.

## Return Values

- **`loss`** (`Tensor`): Output loss tensor. The data type must be identical to that of `input`. When `reduction` is set to `"none"`, the shape must be `[N]`, matching the zeroth dimension of `input`. Otherwise, the shape must be `[1]`.
- **`log_prob`** (`Tensor`): Output tensor used for backward computation. The data type must be identical to that of `input`. The shape of this parameter is `[N, C]`, which must be identical to that of `input`.
- **`zloss`** (`Tensor`): Auxiliary loss tensor. The data type must be identical to that of `input`. The shape must be identical to that of `loss`. This tensor is output only when `return_zloss` is set to `True`. Otherwise, an empty tensor is returned. Currently, this parameter is not supported.
- **`lse_for_zloss`** (`Tensor`): Output tensor passed to backward computation in z-loss scenarios. The data type must be identical to that of `input`. The shape of this parameter is `[N]`, matching the zeroth dimension of `input`. This tensor is returned only when `lse_square_scale_for_zloss` is not `0.0`. Otherwise, an empty tensor is returned. Currently, this parameter is not supported.

## Constraints

- The value range of N in the input shape is (0, 200000].
- When input.requires_grad=True, the default value of `label_smoothing` cannot be modified in `sum`/`none` mode. In `mean` mode, only the default values of the optional parameters can be passed, including `weight`, `ignore_index`, and `label_smoothing`.
- The inputs `lse_square_scale_for_zloss` and `return_zloss` are not supported currently.
- The outputs `zloss` and `lse_for_zloss` are not supported currently.
- Only `loss` in the outputs supports gradient computation.
- This API can be used only in training scenarios.
- This API supports eager mode and graph mode.

## Example

- When reduction is set to `mean`, the example is as follows:

    ```python
    import torch
    import torch_npu

    N = 4096 # batch size
    C = 8080 # number of classes

    # Construct inputs and labels
    input = torch.randn(N, C, dtype=torch.float32, requires_grad=True).npu()
    target = torch.arange(0, N, dtype=torch.int64).npu()

    # Call the NPU cross-entropy loss function. When input.requires_grad=True, only the default values of the optional parameters can be passed in the default mean mode.
    loss, log_prob,_ , _ = torch_npu.npu_cross_entropy_loss(input, target)

    loss.backward()
    ```
    
- When reduction is set to `sum`, the example is as follows:

    ```python
    import torch
    import torch_npu

    N = 4096 # batch size
    C = 8080 # number of classes

    # Construct inputs and labels
    input = torch.randn(N, C, dtype=torch.float32, requires_grad=True).npu()
    target = torch.arange(0, N, dtype=torch.int64).npu()

    # Call the NPU cross-entropy loss function. When input.requires_grad=True, modifying the default value of label_smoothing is not supported in sum mode.
    loss, log_prob,_ , _ = torch_npu.npu_cross_entropy_loss(input, target, reduction="sum", ignore_index=100)

    loss.backward()
    ```

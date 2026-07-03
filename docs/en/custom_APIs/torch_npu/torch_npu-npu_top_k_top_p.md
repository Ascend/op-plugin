# torch\_npu.npu\_top\_k\_top\_p

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>                            |    √     |
|<term>Atlas A3 inference products</term>                             | √  |
|<term>Atlas A2 training products</term>                             | √   |
|<term>Atlas A2 inference products</term>       |    √     |

## Function

- Description: Performs `top-k` and `top-p` sampling and filtering on the original input `logits`.

- Formulas:
  - Sort the input `logits` in ascending order along the last axis to obtain the corresponding sorting results `sortedValue` and `sortedIndices`.
  $$sortedValue, sortedIndices = sort(logits, dim=-1, descend=false, stable=true)$$
  - Calculate the retention threshold (the `k`-th largest value).
  $$topKValue[b][v] = sortedValue[b][sortedValue.size(1) - k[b]]$$
  - Generate the mask required for `top-k` filtering.
  $$topKMask = sortedValue < topKValue$$
  - Set the elements smaller than the threshold to `-inf` based on `topKMask`.
  $$
  sortedValue[b][v] = 
  \begin{cases}
  -inf & \text{topKMask[b][v]=true}\\
  sortedValue[b][v] & \text{topKMask[b][v]=false}
  \end{cases}
  $$
  - Convert the `top-k` filtered data into a probability distribution along the last axis using softmax.
  $$probsValue = softmax(sortedValue, dim=-1)$$
  - Calculate the cumulative probability along the last axis (accumulating from the smallest probability).
  $$probsSum = cumsum(probsValue, dim=-1)$$
  - Generate the `top-p` mask. Positions with a cumulative probability less than or equal to `1 - p` are filtered out, while ensuring that at least one element is retained in each batch.
  $$topPMask[b][v] = probsSum[b][v] <= 1-p[b]$$
  $$topPMask[b][-1] = false$$
  - Set the elements smaller than the threshold to `-inf` based on `topPMask`.
  $$
  sortedValue[b][v] = 
  \begin{cases}
  -inf & \text{topPMask[b][v]=true}\\
  sortedValue[b][v] & \text{topPMask[b][v]=false}
  \end{cases}
  $$
  - Restore the filtered results to the original order according to `sortedIndices`.
  $$out[b][v] = sortedValue[b][sortedIndices[b][v]]$$
  Where, $0 \le b \lt logits.size(0), 0 \le v \lt logits.size(1)$.

## Prototype

```python
torch_npu.npu_top_k_top_p(logits, p, k) -> torch.Tensor
```

## Parameters

- **`logits`** (`Tensor`): Required. Data to be processed. The data type can be `float16`, `bfloat16`, or `float32`. Non-contiguous tensors are supported. The data layout is ND. The number of dimensions must be 2.
- **`p`** (`Tensor`): Required. `top-p` tensor. The value range is `[0, 1]`. The data type can be `float16`, `bfloat16`, or `float32`, and must match that of `logits`. The shape must be 1D and must be identical to the first dimension of `logits`. The data layout is ND. Non-contiguous tensors are supported.
- **`k`** (`Tensor`): Required. `top-k` threshold tensor. The value range is `[1, 1024]`, and the maximum value must be less than or equal to `logits.size(1)`. The data type can be `int32`. The shape must be 1D and must be identical to the first dimension of `logits`. The data layout is ND. Non-contiguous tensors are supported.

## Return Values

`Tensor`

Filtered data. The data type can be `float16`, `bfloat16`, or `float32`, and matches that of `logits`. The shape must be 2D and must be identical to that of `logits`. Non-contiguous tensors are supported. The data layout is ND.

## Constraints

The average performance of this API is superior to that of small-operator implementations when the second dimension of `logits` is greater than 1024. You are advised to use this API when the second dimension of `logits` is greater than 1024.

## Examples

Single-operator call

  ```python
   >>> import torch
   >>> import torch_npu
   >>>
   >>> logits = torch.randn(16, 2048).npu()
   >>> p = torch.rand(16).npu()
   >>>
   >>> k = torch.randint(10, 1024, (16,)).npu().to(torch.int32)
   >>> out = torch_npu.npu_top_k_top_p(logits, p, k)
   >>>
   >>> print(out)
   tensor([[0.0000, 0.0000, 0.0000,  ...,   -inf,   -inf,   -inf],
        [0.0000, 0.0000, 0.0000,  ...,   -inf,   -inf,   -inf],
        [0.0000, 0.0000, 0.0000,  ...,   -inf,   -inf,   -inf],
        ...,
        [0.0000, 0.0000, 1.4379,  ...,   -inf,   -inf,   -inf],
        [0.0000, 0.0000, 0.0000,  ...,   -inf,   -inf,   -inf],
        [1.5425, 0.0000, 0.0000,  ...,   -inf, 1.5491,   -inf]],
       device='npu:0')
  ```

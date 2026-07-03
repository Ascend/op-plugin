# torch_npu.npu_gather_sparse_index

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |

## Function

- Description: Extracts elements from the specified dimension of the input tensor based on the indices in `index` and saves them to the output tensor.

- Formula: Assume that $x$ is the input `input` and $idx$ is the input `index`.
  $$
  y_{i,j} = x_{idx[i],j}
  $$

- Example:

  The input $x$ is as follows:
  $$
  \begin{bmatrix}
    1& 2 & 3\\
    4& 5 & 6\\
    7& 8 &9
  \end{bmatrix}
  $$
  The index $idx$ is as follows:
  $$
  \begin{bmatrix}
  1 & 0
  \end{bmatrix}
  $$
  In this case, the shape of the input $x$ is `(3, 3)`. The index $idx$ is a 1D vector, with shape `(2)`. This configuration extracts row `1` and row `0` from $x$. The output shape is `(2, 3)`. The output is as follows:
  $$
  \begin{bmatrix}
  4 & 5 & 6\\
  1 & 2 & 3
  \end{bmatrix}
  $$

## Prototype

```python
torch_npu.npu_gather_sparse_index(input, index) -> Tensor
```

## Parameters

**`input`** (`Tensor`): Required. Input tensor. The data layout can be ND. The data type can be `float32`, `float16`, `bfloat16`, `int64`, `int32`, `int16`, `int8`, `uint8`, `bool`, `float64`, `complex64`, or `complex128`.

**`index`** (`Tensor`): Required. Tensor representing the indices of target elements. This parameter can be up to 7D. The data type can be `int64` or `int32`. Value range: [0, input.shape[0] - 1]. Negative indices are not supported.

## Return Values

`Tensor`
Result obtained through the API computation, containing the elements extracted based on the indices in `index`. The data type is identical to `input`. The output dimension is $index.dim + input.dim - 1$. For example, when `input.shape` is `(16, 32)` and `index.shape` is `(2, 3)`, the output tensor `out.shape` is `(2, 3, 32)`.

## Constraints

- The sum of the dimensions of `input` and `index` minus `1` must be less than or equal to `8`. That is, $index.dim + input.dim - 1 \le 8$.
- To obtain performance benefits, `input` and `index` must meet the following constraints:
     1. The product of the dimensions in the shape of `input` must be greater than $150 * 1024/itemsize$. The `itemsize` variable indicates the element size of the `input` data type, which can be queried using `torch.dtype.itemsize`.
     2. The product of the dimensions in the shape of `index` must be greater than 960.
     3. The data must be aggregated. Zero and non-zero elements in `input` must be concentrated in contiguous regions. Frequent interleaving is not allowed. Expand `input != 0` into a 1D Boolean sequence `mask` in row-major order. Compute the state switching ratio between adjacent elements:
     $$
     \mathrm{switch\_ratio} = \frac{\sum_{i=1}^{N-1} \mathbf{1}(\mathrm{mask}_i \ne \mathrm{mask}_{i-1})}{N - 1}
     $$
     $N$ indicates the total number of elements in `input`, and $\mathrm{mask}_i$ indicates the `i`-th element in $\mathrm{mask}$. A smaller $\mathrm{switch\_ratio}$ indicates stronger aggregation. A larger $\mathrm{switch\_ratio}$ indicates more dispersed values. In this case, expected performance benefits may not be achieved.

## Examples

```python
import torch
import torch_npu

inputs = torch.randn(16, 32).npu()
index = torch.randint(0, 16, [2, 3]).npu()
out = torch_npu.npu_gather_sparse_index(inputs, index)
```

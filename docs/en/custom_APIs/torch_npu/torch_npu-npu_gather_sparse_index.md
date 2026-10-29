# torch_npu.npu_gather_sparse_index

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id3 -->

## Function

- Function: extracts elements from the specified dimension of the input tensor according to the indices in `index` and saves them to the output tensor.

- Formulas: Assume that $x$ is the input `input` and $idx$ is the input `index`.
  $$
  y_{i,j} = x_{idx[i],j}
  $$

- Examples:

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
  In this case, the shape of the input $x$ is $[3, 3]$ and the shape of the index $idx$ is $[2]$ (a 1D vector). The preceding indices mean that the 1st row and the 0th row of $x$ are extracted respectively. Therefore, the output shape is $[2, 3]$ and the output result is as follows:
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

**`input`** (`Tensor`): Required. Input tensor. The data layout can be ND. The data type can be `torch.float32`, `torch.float16`, `torch.bfloat16`, `torch.int64`, `torch.int32`, `torch.int16`, `torch.int8`, `torch.uint8`, `torch.bool`, `torch.float64`, `torch.complex64`, or `torch.complex128`.

**`index`** (`Tensor`): Required. Tensor representing the subscript indices of the target elements. The number of dimensions must not exceed 7. The data type can be `torch.int64` or `torch.int32`. The value range is $[0, input.shape[0] - 1]$. Negative indices are not supported.

## Return Values

`Tensor`

The result obtained by the API, including the elements extracted according to the indices in `index`. The data type is the same as that of `input`. The number of dimensions of the output tensor is $index.dim + input.dim - 1$. For example, if `input.shape = [16, 32]` and `index.shape = [2, 3]`, then the output tensor `out.shape = [2, 3, 32]`.

## Constraints

- The sum of the number of dimensions of `input` and the number of dimensions of `index` minus 1 must not exceed 8, that is, $index.dim + input.dim - 1 <= 8$.
- To obtain performance benefits, `input` and `index` must meet the following constraints:
     1. The product of the dimensions of the `input` shape must be greater than $150 * 1024 / itemsize$, where itemsize is the element size of the `input` dtype, which can be queried using `torch.dtype.itemsize`.
     2. The product of the dimensions of the `index` shape must be greater than 960.
     3. The data must be aggregated, that is, the zero and non-zero elements in `input` are concentrated in contiguous regions instead of being frequently interleaved. Expand `input != 0` into a 1D boolean sequence $mask$ in row-major order and calculate the ratio of state switches between adjacent elements:
     $$
     \mathrm{switch\_ratio} = \frac{\sum_{i=1}^{N-1} \mathbf{1}(\mathrm{mask}_i \ne \mathrm{mask}_{i-1})}{N - 1}
     $$
     In the formula, $N$ is the total number of elements of `input`, and $\mathrm{mask}_i$ indicates the $i$-th element in the boolean sequence $\mathrm{mask}$. A smaller $\mathrm{switch\_ratio}$ indicates that the zero and non-zero values are distributed more contiguously and the data aggregation degree is higher; a larger $\mathrm{switch\_ratio}$ indicates that the zero and non-zero values are more scattered, and the expected performance benefit may not be obtained.

## Example

```python
import torch
import torch_npu

inputs = torch.randn(16, 32).npu()
index = torch.randint(0, 16, [2, 3]).npu()
out = torch_npu.npu_gather_sparse_index(inputs, index)
```

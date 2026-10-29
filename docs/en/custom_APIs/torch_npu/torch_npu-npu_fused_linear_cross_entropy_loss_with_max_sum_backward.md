# torch_npu.npu_fused_linear_cross_entropy_loss_with_max_sum_backward

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->

## Function

- API function: Gradient operator for cross-entropy loss computation in vocabulary-parallel scenarios. It computes the gradients of the leaf nodes `input` and `weight`. The related outputs of npu_fused_linear_online_max_sum and npu_fused_cross_entropy_loss_with_max_sum must be obtained as inputs to this API. It supports the high-performance mode (passing Softmax) and the memory-saving mode (passing `logits_max` and `sum_exp_logits`).

- Formulas:

&emsp;&emsp;High-performance mode (`softmax` is not None):

$$
\text{softmax} \in \mathbb{R}^{BT \times V}
$$

$$
\text{arange\_1d} = [0, 1, \dots, BT-1] \in \mathbb{N}^{BT}
$$

$$
\text{softmax\_update} = \mathbf{1} - \text{target\_mask}.view(-1) \in \mathbb{R}^{BT}
$$

$$
\text{softmax}[\text{arange\_1d}, \text{masked\_target}] \leftarrow \text{softmax}[\text{arange\_1d}, \text{masked\_target}] - \text{softmax\_update}
$$

$$
\text{softmax} \leftarrow \text{softmax} \odot \text{grad}.unsqueeze(-1) \in \mathbb{R}^{BT \times V}
$$

$$
\text{grad\_input} = \text{softmax} \cdot \text{weight}^T \in \mathbb{R}^{BT \times H}
$$

$$
\text{grad\_weight} = \text{softmax}^T \cdot \text{input} \in \mathbb{R}^{V \times H}
$$

&emsp;&emsp;Memory-saving mode (`softmax` is None):

$$
\text{vocab\_parallel\_logits} = \text{input} \cdot \text{weight}^T \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{logits\_sub} = \text{vocab\_parallel\_logits} - \text{logits\_max}.unsqueeze(-1) \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{exp\_logits} = \exp(\text{logits\_sub}) \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{exp\_logits} \gets \frac{\text{exp\_logits}}{\text{sum\_exp\_logits}.unsqueeze(-1)} \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{grad\_logits} = \text{exp\_logits} \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{grad\_2d} = \text{grad\_logits}.view(-1, \text{partition\_vocab\_size}) \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{arange\_1d} = [0, 1, \dots, BT-1] \quad \in \mathbb{N}^{BT}
$$

$$
\text{softmax\_update} = 1 - \text{target\_mask}.view(-1) \quad \in \mathbb{R}^{BT}
$$

$$
\text{grad\_2d}[\text{arange\_1d}, \text{masked\_target\_1d}] \gets \text{grad\_2d}[\text{arange\_1d}, \text{masked\_target\_1d}] - \text{softmax\_update}
$$

$$
\text{grad\_logits} \gets \text{grad\_logits} \odot \text{grad}.unsqueeze(-1) \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{grad\_input} = \text{grad\_logits} \cdot \text{weight} \quad \in \mathbb{R}^{BT \times H}
$$

$$
\text{grad\_weight} = \text{grad\_logits}^T \cdot \text{input} \quad \in \mathbb{R}^{V \times H}
$$

## Prototype

```python
torch_npu.npu_fused_linear_cross_entropy_loss_with_max_sum_backward(grad, input, weight, target_mask, masked_target, label_smoothing=0.0, logits_max=None, sum_exp_logits=None, softmax=None) -> (Tensor, Tensor)
```

## Parameters

- **`grad`** (`Tensor`): Required. Gradient of the current node. A 1D Tensor. The data type can be `torch.float32`. The data layout can be ND. Empty Tensors are supported.
- **`input`** (`Tensor`): Required. Input matrix of the matrix multiplication. A 2D Tensor. The data type can be `torch.float16` or `torch.bfloat16`. The data layout can be ND. The length of dimension 0 must be the same as that of `grad`. Empty Tensors are supported.
- **`weight`** (`Tensor`): Required. Weight matrix of the matrix multiplication. A 2D Tensor. The data type must be the same as that of `input`. The data layout can be ND. The length of dimension 0 must not be less than 128, and the length of dimension 1 must be the same as that of dimension 1 of `input`.
- **`target_mask`** (`Tensor`): Required. Bit mask indicating whether the target word ID is within the range. A 1D Tensor. The data type can be `torch.uint8`. The data layout can be ND. Each bit represents one Boolean value, and the shape length multiplied by 8 must not be less than the length of `grad`.
- **`masked_target`** (`Tensor`): Required. Local index to which the target word ID is mapped on the current device. A 1D Tensor. The data type can be `torch.int32` or `torch.int64`. The data layout can be ND. The shape length must be the same as that of `grad`.
- **`label_smoothing`** (`float`): Optional. Label smoothing coefficient. Currently, only 0.0 is supported. The default value is 0.0.
- **`logits_max`** (`Tensor`): Optional. Global maximum logits value. A 1D Tensor. The data type can be `torch.float32`. It must be provided when `softmax` is None. The default value is None.
- **`sum_exp_logits`** (`Tensor`): Optional. Processed logits. A 1D Tensor. The data type can be `torch.float32`. It must be provided when `softmax` is None. The default value is None.
- **`softmax`** (`Tensor`): Optional. Softmax computation result. A 2D Tensor. The data type can be `torch.float32`. When it is passed, the high-performance mode is used. When it is not passed, the memory-saving mode is used (`logits_max` and `sum_exp_logits` must be provided). The default value is None.

## Return Values

- **`input_grad`** (`Tensor`): Gradient of `input`. The data type is the same as that of `input`, and the shape is `[input.size(0), input.size(1)]`.
- **`weight_grad`** (`Tensor`): Gradient of `weight`. The data type is the same as that of `input`, and the shape is `[weight.size(0), weight.size(1)]`.

## Constraints

- `label_smoothing` currently supports only 0.
- When `softmax` is None, `logits_max` and `sum_exp_logits` must be provided at the same time.
- When `softmax` is not None, `logits_max` and `sum_exp_logits` are invalid.

## Examples

- High-performance mode (passing Softmax):

```python
import torch
import torch_npu

batch = 128
hidden = 64
vocab_size = 256

grad = torch.randn(batch, dtype=torch.float32).npu()
input_tensor = torch.randn(batch, hidden, dtype=torch.float16).npu()
weight_tensor = torch.randn(vocab_size, hidden, dtype=torch.float16).npu()
target_mask = torch.zeros((batch + 7) // 8, dtype=torch.uint8).npu()
masked_target = torch.randint(0, vocab_size, (batch,), dtype=torch.int32).npu()
softmax = torch.randn(batch, vocab_size, dtype=torch.float32).npu()

input_grad, weight_grad = torch_npu.npu_fused_linear_cross_entropy_loss_with_max_sum_backward(
    grad, input_tensor, weight_tensor, target_mask, masked_target,
    label_smoothing=0.0, softmax=softmax
)
```

- Memory-saving mode (passing logits_max and sum_exp_logits):

```python
import torch
import torch_npu

batch = 128
hidden = 64
vocab_size = 256

grad = torch.randn(batch, dtype=torch.float32).npu()
input_tensor = torch.randn(batch, hidden, dtype=torch.float16).npu()
weight_tensor = torch.randn(vocab_size, hidden, dtype=torch.float16).npu()
target_mask = torch.zeros((batch + 7) // 8, dtype=torch.uint8).npu()
masked_target = torch.randint(0, vocab_size, (batch,), dtype=torch.int32).npu()
logits_max = torch.randn(batch, dtype=torch.float32).npu()
sum_exp_logits = torch.abs(torch.randn(batch, dtype=torch.float32)).npu() + 1.0

input_grad, weight_grad = torch_npu.npu_fused_linear_cross_entropy_loss_with_max_sum_backward(
    grad, input_tensor, weight_tensor, target_mask, masked_target,
    label_smoothing=0.0, logits_max=logits_max, sum_exp_logits=sum_exp_logits, softmax=None
)
```

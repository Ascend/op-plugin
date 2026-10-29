# torch_npu.npu_fused_linear_online_max_sum

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

- API function: Fuses matrix multiplication and cross-entropy preprocessing in vocabulary-parallel scenarios. It is functionally equivalent to Megatron's MatMul and fused_vocab_parallel_cross_entropy implementation, and supports splitting and fusing MatMul and CELoss along the vocabulary_size dimension. This API is the pre-communication part and must be used together with npu_fused_cross_entropy_loss_with_max_sum.

- Formulas:

  1. Multiply $input$ by $weight^T$:

     $$
     vocabParallelLogitsOutOptional = input @ weight^T
     $$
  2. Compute the maximum value of each row of $vocabParallelLogitsOutOptional$:

     $$
     logitsMaxLocalOut = max(vocabParallelLogitsOutOptional, dim=-1)
     $$
  3. Compute the difference between $vocabParallelLogitsOutOptional$ and $logitsMaxLocalOut$:

     $$
     subRes[b][n] = vocabParallelLogitsOutOptional[b][n] - logitsMaxLocalOut[b]
     $$
  4. Compute the sum of each row of $subRes$ after exponentiation:

     $$
     sumExpLogitsLocalOut = sum(exp(subRes), dim=-1)
     $$
  5. Compute the mask for $target$ less than $vocabStartIndex$ or greater than $vocabEndIndex$:

     $$
     targetMask = (target < vocabStartIndex) | (target > vocabEndIndex)
     $$
  6. Compute $maskedTargetOut$:

     $$
     maskedTargetOut[b] =
     \begin{cases}
     0 & \text{targetMask[b]=true}\\
     target[b] - vocabStartIndex & \text{targetMask[b]=false}
     \end{cases}
     $$
  7. Compute $predictedLogitsLocalOut$:

     $$
     predictedLogitsLocalOut[b] =
     \begin{cases}
     0 & \text{targetMask[b]=true}\\
     subRes[b][maskedTargetOut[b]] & \text{targetMask[b]=false}
     \end{cases}
     $$
  8. Compute $targetMaskOut$:

     $$
     alignNum = (input.size(0) + 7) / 8 * 8\\
     maskBit[p] = \begin{cases}
     uint8(targetMask[p]) & \text{p < input.size(0)}\\
     1 & \text{input.size(0) <= p < alignNum}
     \end{cases} \\
     targetMaskOut[k] = 0b(maskBit[8*k:8*k+8])
     $$

  Here, $0 \le b \lt input.size(0), 0 \le n \lt weight.size(0), 0 \le p \lt alignNum, 0 \le k \lt alignNum / 8$.

## Prototype

```python
torch_npu.npu_fused_linear_online_max_sum(input, weight, target, vocab_start_index, vocab_end_index, return_logits=False) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)
```

## Parameters

- **`input`** (`Tensor`): Required. Left matrix of the MatMul computation. A 2D Tensor. The data type can be `torch.bfloat16` or `torch.float16`. The data layout can be ND. `input.size(1)` must be less than or equal to 65534. Empty Tensors are supported.
- **`weight`** (`Tensor`): Required. Right matrix of the MatMul computation. A 2D Tensor. The data type must be the same as that of `input`. The data layout can be ND. `weight.size(1)` must be the same as `input.size(1)`. Empty Tensors are supported.
- **`target`** (`Tensor`): Required. Target index. A 1D Tensor. The data type can be `torch.int32` or `torch.int64`. The data layout can be ND. `target.size(0)` must be the same as `input.size(0)`. Empty Tensors are supported.
- **`vocab_start_index`** (`int`): Required. Start index of the vocabulary allocated to the current rank. The value range is `[0, max(target) - 1]`.
- **`vocab_end_index`** (`int`): Required. End index of the vocabulary allocated to the current rank. The value range is `[vocab_start_index, min(vocab_start_index + weight.size(0) - 1, max(target) - 1)]`.
- **`return_logits`** (`bool`): Optional. Whether to return the MatMul result `vocabParallelLogits`. The default value is `False`. When `True`, the high-performance branch is used. When `False`, the memory-saving branch is used.

## Return Values

- **`logits_max`** (`Tensor`): Maximum value of each row after the MatMul computation. The data type can be `torch.float32`. The shape is `[input.size(0)]`.
- **`sum_exp_logits`** (`Tensor`): Result of accumulating each row of `subRes` after exp. The data type can be `torch.float32`. The shape is `[input.size(0)]`.
- **`predicted_logits`** (`Tensor`): Result of filtering `subRes` by `maskedTarget`. The data type can be `torch.float32`. The shape is `[input.size(0)]`.
- **`target_mask`** (`Tensor`): Packed bit representation of the vocabulary mask. The data type can be `torch.uint8`. The shape is `[(input.size(0) + 7) // 8]`.
- **`masked_target`** (`Tensor`): Result of filtering `target` by the mask. The data type is the same as that of `target`. The shape is `[input.size(0)]`.
- **`vocab_parallel_logits`** (`Tensor`): MatMul computation result. When `return_logits` is `True`, the data type is the same as that of `input` and the shape is `[input.size(0), weight.size(0)]`. When `return_logits` is `False`, an empty Tensor is returned.

## Constraints

- The data types of `input` and `weight` must be the same.
- The data types of `target` and `masked_target` must be the same.
- When `return_logits` is `True`, the data type of `vocabParallelLogits` must be the same as that of `input`.
- `vocab_start_index` must not be less than 0.
- `vocab_end_index` must not be less than `vocab_start_index`.

## Examples

```python
import torch
import torch_npu

batch = 128
hidden = 64
vocab_size = 256

input_tensor = torch.randn(batch, hidden, dtype=torch.float16).npu()
weight_tensor = torch.randn(vocab_size, hidden, dtype=torch.float16).npu()
target_tensor = torch.randint(0, vocab_size, (batch,), dtype=torch.int32).npu()

# High-performance mode (return_logits=True)
logits_max, sum_exp_logits, predicted_logits, target_mask, masked_target, vocab_parallel_logits = \
    torch_npu.npu_fused_linear_online_max_sum(
        input_tensor, weight_tensor, target_tensor,
        vocab_start_index=0, vocab_end_index=64, return_logits=True
    )

# Memory-saving mode (return_logits=False)
logits_max, sum_exp_logits, predicted_logits, target_mask, masked_target, _ = \
    torch_npu.npu_fused_linear_online_max_sum(
        input_tensor, weight_tensor, target_tensor,
        vocab_start_index=0, vocab_end_index=64, return_logits=False
    )
```

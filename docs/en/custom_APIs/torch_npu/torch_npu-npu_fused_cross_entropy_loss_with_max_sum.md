# torch_npu.npu_fused_cross_entropy_loss_with_max_sum

> [!NOTICE]  
> This API is new in this version. For details about the dependency requirements, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="A3" id1 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id2 -->

## Function

- API function: This API is part of the cross-entropy computation module in vocabulary-parallel scenarios and addresses the memory and computational efficiency problems caused by ultra-large vocabularies. It computes part of the loss and softmax results and must be used together with `npu_fused_linear_online_max_sum`. In multi-rank scenarios, global communication (all-reduce) must be performed on `logits_max` and `sum_exp_logits` before this API is called.

- Formulas:

    $$
    lossOut = log(sum\_exp\_logits) - predicted\_logits
    $$

    $$
    softMaxOutOptional = exp(vocab\_parallel\_logits -logits\_max.unsqueeze(dim = -1)) / sum\_exp\_logits.unsqueeze(dim = -1)
    $$

## Prototype

```python
torch_npu.npu_fused_cross_entropy_loss_with_max_sum(logits_max, sum_exp_logits, predicted_logits, *, label_smoothing=0.0, input=None, weight=None, vocab_parallel_logits=None) -> (Tensor, Tensor)
```

## Parameters

- **`logits_max`** (`Tensor`): Required. Maximum value in each row of the MatMul result after global communication. A 1D Tensor. The data type can be `torch.float32`. The data layout can be ND.
- **`sum_exp_logits`** (`Tensor`): Required. Accumulated exp result after global communication. A 1D Tensor. The data type can be `torch.float32`. The data layout can be ND. The shape must be the same as that of `logits_max`.
- **`predicted_logits`** (`Tensor`): Required. Predicted logits after global communication. A 1D Tensor. The data type can be `torch.float32`. The data layout can be ND. The shape must be the same as that of `logits_max`.
- **`*`**: Syntax separator used to distinguish positional arguments from keyword arguments. Variables before it are position-dependent and must be provided in order. Variables after it are optional and position-independent; they must be assigned by using key-value pairs. If not assigned, their default values are used.
- **`label_smoothing`** (`float`): Optional. Label smoothing coefficient. Currently, only 0.0 is supported. The default value is 0.0.
- **`input`** (`Tensor`): Optional. Left matrix of the MatMul input. Currently, only None is supported. The default value is None.
- **`weight`** (`Tensor`): Optional. Right matrix of the MatMul input. Currently, only None is supported. The default value is None.
- **`vocab_parallel_logits`** (`Tensor`): Optional. MatMul computation result. When it is passed, the softmax output is computed; when it is not passed, the softmax output is an empty Tensor. The data type can be `torch.float32`, `torch.float16`, or `torch.bfloat16`. It is a 2D Tensor, and its dimension 0 must be the same as that of `logits_max`. The default value is None.

## Return Values

- **`loss`** (`Tensor`): Cross-entropy loss. The data type can be `torch.float32`. The shape is the same as that of `logits_max`.
- **`softmax`** (`Tensor`): Softmax result. When `vocab_parallel_logits` is not None, the data type can be `torch.float32` and the shape is the same as that of `vocab_parallel_logits`; when `vocab_parallel_logits` is None, an empty Tensor is returned.

## Constraints

- The shapes of `logits_max`, `sum_exp_logits`, and `predicted_logits` must be the same.
- `label_smoothing` currently supports only 0.

## Examples

```python
import torch
import torch_npu

# Assume that the output has been obtained by using npu_fused_linear_online_max_sum and global communication has been completed.
batch = 128
vocab_size = 256

logits_max = torch.randn(batch, dtype=torch.float32).npu()
sum_exp_logits = torch.abs(torch.randn(batch, dtype=torch.float32)).npu() + 1.0
predicted_logits = torch.randn(batch, dtype=torch.float32).npu()

# Do not return softmax.
loss, _ = torch_npu.npu_fused_cross_entropy_loss_with_max_sum(
    logits_max, sum_exp_logits, predicted_logits,
    label_smoothing=0.0
)

# Return softmax (pass vocab_parallel_logits).
vocab_parallel_logits = torch.randn(batch, vocab_size, dtype=torch.float16).npu()
loss, softmax = torch_npu.npu_fused_cross_entropy_loss_with_max_sum(
    logits_max, sum_exp_logits, predicted_logits,
    label_smoothing=0.0, vocab_parallel_logits=vocab_parallel_logits
)
```

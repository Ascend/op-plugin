# (beta) torch_npu.npu_softmax_cross_entropy_with_logits

> [!NOTICE]  
> This API is planned for deprecation. The underlying operator kernel is no longer maintained, and performance and accuracy are not guaranteed. This API is not recommended.

## Supported Products

<!-- npu="A3" id1 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id2 -->
<!-- npu="310p" id3 -->
- <term>Atlas inference products</term>: Supported
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas training products</term>: Supported
<!-- end id4 -->

## Function

- Computes the cross-entropy loss for softmax.

- Formula:

$$
     loss = -\sum_{i=1}^{N}y_i * log(softmax(x_i))
$$
$x_i$ represents to the input `features`, $y_i$ represents to the input `labels`, and $N$ represents the length of the input features.

## Prototype

```python
torch_npu.npu_softmax_cross_entropy_with_logits(features, labels) -> Tensor
```

## Parameters

- **`features`** (`Tensor`): Required. Input features, $x_i$ in the formula. This parameter must be a matrix with shape `(1, batch_size * num_classes)`. The data layout can be ND. Non-contiguous tensors are supported. This parameter can be up to 2D. Empty tensors are supported. The data type can be `torch.float`, `torch.float16`, or `torch.bfloat16`.

  <!-- npu="910,310p" id5 -->
  - <term>Atlas training products</term> and <term>Atlas inference products</term>: The `torch.bfloat16` data type is not supported.
  <!-- end id5 -->

- **`labels`** (`Tensor`): Required. Input labels, $y_i$ in the formula. The shape and data type must be identical to those of `features`. The data layout can be ND. Non-contiguous tensors are supported. This parameter can be up to 2D. Empty tensors are supported. The data type can be `torch.float`, `torch.float16`, or `torch.bfloat16`.

  <!-- npu="910,310p" id6 -->
  - <term>Atlas training products</term> and <term>Atlas inference products</term>: The `torch.bfloat16` data type is not supported.
  <!-- end id6 -->

## Return Values

`Tensor`

Output tensor representing the cross-entropy loss computation results of softmax and cross entropy, $loss$ in the formula.

## Example

```python
>>> import torch, torch_npu
>>> batch_size = 4
>>> num_classes = 12
>>> features = torch.rand(1, batch_size * num_classes).npu()
>>> labels = torch.rand(1, batch_size * num_classes).npu()
>>> output = torch_npu.npu_softmax_cross_entropy_with_logits(features, labels)
>>> print(output)
tensor([97.9450], device='npu:0')
```

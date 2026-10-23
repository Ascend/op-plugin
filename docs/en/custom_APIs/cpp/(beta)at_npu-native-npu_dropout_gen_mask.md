# (beta) at_npu::native::npu_dropout_gen_mask

## Supported Products

<!-- npu="910" id1 -->
- <term>Atlas training products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas inference products</term>: Supported
<!-- end id4 -->

## Function

Generates a random mask based on the probability `p` during training, which is used to set elements to zero.

## Definition File

third_party\op-plugin\op_plugin\include\ops.h

## Prototype

```cpp
at::Tensor npu_dropout_gen_mask(const at::Tensor &self, at::IntArrayRef size, double p, int64_t seed, int64_t offset, c10::optional<bool> parallel, c10::optional<bool> sync)
```

## Parameters

- **`self`** (`Tensor`): Input tensor.
- **`size`** (`IntArrayRef`): Shape dimensions of the generated mask.
- **`p`** (`double`): Probability of setting an element to 0.
- **`seed`** (`int64_t`): Seed that determines the generated random number sequence.
- **`offset`** (`int64_t`): Random number offset that controls the execution alignment position of the generated random number sequence.
- **`parallel`** (`bool`): Optional. Specifies whether to enable parallel computation.
- **`sync`** (`bool`): Optional. Specifies whether to enable synchronized execution.

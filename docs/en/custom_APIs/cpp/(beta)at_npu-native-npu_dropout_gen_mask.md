# (beta) at_npu::native::npu_dropout_gen_mask

## Definition File

third_party\op-plugin\op_plugin\include\ops.h

## Prototype

```cpp
at::Tensor npu_dropout_gen_mask(const at::Tensor &self, at::IntArrayRef size, double p, int64_t seed, int64_t offset, c10::optional<bool> parallel, c10::optional<bool> sync)
```

## Function

Generates a random mask based on the probability `p` during training, which is used to set elements to zero.

## Parameters

- **`self`** (`Tensor`): Input tensor.
- **`size`** (`IntArrayRef`): Shape dimensions of the generated mask.
- **`p`** (`double`): Probability of setting an element to 0.
- **`seed`** (`int64_t`): Seed that determines the generated random number sequence.
- **`offset`** (`int64_t`): Random number offset that controls the execution alignment position of the generated random number sequence.
- **`parallel`** (`bool`): Optional. Specifies whether to enable parallel computation.
- **`sync`** (`bool`): Optional. Specifies whether to enable synchronized execution.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>

# torch_npu.npu_advance_step_flashattn

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>                            |    √     |
|<term>Atlas A2 training products</term>                             | √   |

## Function

- Description: Implements the `advance_step_flashattn` functionality from the vLLM library on the NPU. It performs in-place updates of `input_tokens`, `input_positions`, `seq_lens`, and `slot_mapping` during each generation step.
- Non-speculative scenarios: $blockIdx$ represents the hardware execution core index.
     $$
     blockTablesStride = blockTables.stride(0) \\
     inputTokens[blockIdx] = sampledTokenIds[blockIdx]  \\
     inputPositions[blockIdx] = seqLens[blockIdx] \\
     seqLens[blockIdx] = seqLens[blockIdx] + 1 \\
     
     slotMapping[blockIdx] = ({blockTables}[blockIdx] + blockTablesStride * blockIdx) * blockSize + (seqLens[blockIdx]\%blockSize)
     $$
- Speculative scenarios: $i$ represents the request index, and `input_tokens` is interpreted as a 2D view with shape `[num_seqs, 1 + spec_num]`.
     $$
     lastToken = \text{last valid token of each request in }sampledTokenIds \\
     blockTablesStride = blockTables.stride(0) \\
     inputTokens[:numSeqs, 0] = lastToken  \\
     inputTokens[:numSeqs, 1:] = specToken  \\
     inputPositions[i] = inputPositions[i] + 1 + acceptedNum[i] \\
     seqLens[i] = inputPositions[i] + 1 \\
     slotMapping[i] = ({blockTables}[i] + blockTablesStride * i) * blockSize + (inputPositions[i]\%blockSize)
     $$

## Prototype

```python
torch_npu.npu_advance_step_flashattn(input_tokens, sampled_token_ids, input_positions, seq_lens, slot_mapping, block_tables, num_seqs, num_queries, block_size, spec_token, accepted_num) -> ()
```

## Parameters

- **`input_tokens`** (`Tensor`): Required. Input/output tensor used to update token values in the vLLM model, $inputTokens$ in the formulas. The data type can be `int64`. The shape of this parameter is `[num_seqs]` (non-speculative) or `[num_seqs * (1 + spec_num)]` (speculative). Empty tensors are not supported. The value must be a positive integer greater than 0.
- **`sampled_token_ids`** (`Tensor`): Required. Input tensor used to store token IDs, $sampledTokenIds$ in the formula. The data type can be `int64`. The shape of this parameter is `[num_queries, 1]` (non-speculative) or `[num_seqs, 1 + spec_num]` (speculative). Empty tensors are not supported. The value must be a positive integer greater than 0.
- **`input_positions`** (`Tensor`): Required. Input/output tensor used to record token indices, $inputPositions$ in the formulas. The data type can be `int64`. The shape of this parameter is `[num_seqs]` (non-speculative) or `[num_seqs * (1 + spec_num)]` (speculative). Empty tensors are not supported. The value must be a positive integer greater than 0.
- **`seq_lens`** (`Tensor`): Required. Input/output tensor used to record sequence lengths under different `block_idx`, $seqLens$ in the formula. The data type can be `int64`. The shape of this parameter is `[num_seqs]` (non-speculative) or `[num_seqs * (1 + spec_num)]` (speculative). Empty tensors are not supported. The value must be a positive integer greater than 0.
- **`slot_mapping`** (`Tensor`): Required. Input/output tensor used to map token positions in a sequence to physical memory positions, $slotMapping$ in the formulas. The data type can be `int64`. The shape of this parameter is `[num_seqs]` (non-speculative) or `[num_seqs * (1 + spec_num)]` (speculative). Empty tensors are not supported. The value must be a positive integer greater than 0.
- **`block_tables`** (`Tensor`): Required. Input/output tensor used to record block allocation for different `block_idx`, $blockTables$ in the formulas. The data type can be `int64`. The shape must have two dimensions. The first dimension equals `num_seqs`, and the second dimension must be greater than the integer division of the maximum value in `seq_lens` by `block_size`.
- **`num_seqs`** (`int`): Required. Records the number of input sequences. The value must be a positive integer greater than 0.
- **`num_queries`** (`int`): Required. Records the number of input queries. The value must be a positive integer greater than 0.
- **`block_size`** (`int`): Required. Size of each block, $blockSize$ in the formulas. The value must be a positive integer greater than 0.
- **`spec_token`** (`Tensor`): Optional. Input tensor used to record token indices in speculative scenarios. If `spec_token` is omitted (defaults to `None`), non-speculative execution applies. If provided, speculative execution applies and the shape of this parameter is `[num_seqs, spec_num]`. Empty tensors are not supported. The value must be a positive integer greater than 0.
- **`accepted_num`** (`Tensor`): Optional. Input tensor used to record the number of accepted speculative tokens per request in speculative scenarios. The data type can be `int64`.

## Return Values

This API updates `input_tokens`, `input_positions`, `seq_lens`, and `slot_mapping` in place and has no return value.

## Constraints

- `Non-speculative scenarios`: The first dimension of `input_tokens`, `input_positions`, `seq_lens`, `slot_mapping`, and `block_tables` must equal `num_seqs`.
- `Speculative scenarios`: `input_tokens`, `input_positions`, `seq_lens`, and `slot_mapping` must be 1D tensors with a length of `num_seqs * (1 + spec_num)`, where `spec_num` is the number of speculative tokens.
- `sampled_token_ids shape`: The first dimension must equal `num_queries`. In non-speculative scenarios, the second dimension must be 1. In speculative scenarios, the second dimension must equal `1 + spec_num`.
- `block_tables shape`: The second dimension must be greater than the integer division of the maximum value in `seq_lens` by `block_size`.
- In non-speculative scenarios, `num_seqs` must be greater than `num_queries`. In speculative scenarios, `num_queries` must equal `num_seqs`.
- This API is restricted to inference scenarios only and does not support backward propagation.

## Examples

Non-speculative scenarios:

```python
import numpy as np
 
import torch
import torch_npu
 
num_seqs = 16
num_queries = 8
block_size = 8
 
input_tokens = np.random.randint(10, size=(num_seqs,))
sampled_token_ids = np.random.randint(10, size=(num_queries,1))
input_positions = np.random.randint(10, size=(num_seqs,))
seq_lens = np.random.randint(10, size=(num_seqs,))
slot_mapping = np.random.randint(10, size=(num_seqs,))
 
input_tokens = torch.tensor(input_tokens, dtype=torch.int64, device="npu")
sampled_token_ids = torch.tensor(sampled_token_ids, dtype=torch.int64, device="npu")
input_positions = torch.tensor(input_positions, dtype=torch.int64, device="npu")
seq_lens = torch.tensor(seq_lens, dtype=torch.int64, device="npu")
slot_mapping = torch.tensor(slot_mapping, dtype=torch.int64, device="npu")
 
max_seq_len = seq_lens.max().item()
block_tables = np.random.randint(10, size=(num_seqs, max_seq_len // block_size + 1))
block_tables = torch.tensor(block_tables, dtype=torch.int64, device="npu")
 
torch_npu.npu_advance_step_flashattn(input_tokens, sampled_token_ids, input_positions, seq_lens, slot_mapping, block_tables, num_seqs, num_queries, block_size)
```

Speculative scenarios:

```python
import numpy as np

import torch
import torch_npu

num_seqs = 16
num_queries = 16
block_size = 8
spec_num = 2

input_tokens = np.random.randint(10, size=(num_seqs*(1 + spec_num),))
sampled_token_ids = np.random.randint(10, size=(num_seqs, 1 + spec_num))
input_positions = np.random.randint(10, size=(num_seqs*(1 + spec_num),))
seq_lens = np.random.randint(10, size=(num_seqs*(1 + spec_num),))
slot_mapping = np.random.randint(10, size=(num_seqs*(1 + spec_num),))
spec_token = np.random.randint(10, size=(num_seqs, spec_num))
accepted_num = np.random.randint(10, size=(num_seqs,))

input_tokens = torch.tensor(input_tokens, dtype=torch.int64, device="npu")
sampled_token_ids = torch.tensor(sampled_token_ids, dtype=torch.int64, device="npu")
input_positions = torch.tensor(input_positions, dtype=torch.int64, device="npu")
seq_lens = torch.tensor(seq_lens, dtype=torch.int64, device="npu")
slot_mapping = torch.tensor(slot_mapping, dtype=torch.int64, device="npu")
spec_token = torch.tensor(spec_token, dtype=torch.int64, device="npu")
accepted_num = torch.tensor(accepted_num, dtype=torch.int64, device="npu")

max_seq_len = seq_lens.max().item()
block_tables = np.random.randint(10, size=(num_seqs, max_seq_len // block_size + 1))
block_tables = torch.tensor(block_tables, dtype=torch.int64, device="npu")

torch_npu.npu_advance_step_flashattn(input_tokens, sampled_token_ids, input_positions,
                                     seq_lens, slot_mapping, block_tables, num_seqs,
                                     num_queries, block_size, spec_token=spec_token, accepted_num=accepted_num)
```

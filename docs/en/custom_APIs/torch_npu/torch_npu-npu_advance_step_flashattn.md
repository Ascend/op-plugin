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
- Speculative scenarios: $i$ represents the index of the active request.
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

- **`input_tokens`** (`Tensor`): Required. Input/output tensor used to update token values in the vLLM model, $inputTokens$ in the formulas. The data type can be `int64`. The shape of this parameter is `[num_seqs]` (non-speculative) or `[num_seqs, 1 + spec_num]` (speculative). Empty tensors are not supported. The value must be a positive integer greater than 0.
- **`sampled_token_ids`** (`Tensor`): Required. Input tensor used to store token IDs, $sampledTokenIds$ in the formula. The data type can be `int64`. The shape of this parameter is `[num_queries, 1]` (non-speculative) or `[num_seqs, 1 + spec_num]` (speculative). Empty tensors are not supported. The value must be a positive integer greater than 0.
- **`input_positions`** (`Tensor`): Required. Input/output tensor used to record token indices, $inputPositions$ in the formulas. The data type can be `int64`. The shape of this parameter is `[num_queries, 1]` (non-speculative) or `[num_seqs, 1 + spec_num]` (speculative). Empty tensors are not supported. The value must be a positive integer greater than 0.
- **`seq_lens`** (`Tensor`): Required. Input/output tensor used to record sequence lengths under different `block_idx`, $seqLens$ in the formula. The data type can be `int64`. The shape of this parameter is `[num_queries, 1]` (non-speculative) or `[num_seqs, 1 + spec_num]` (speculative). Empty tensors are not supported. The value must be a positive integer greater than 0.
- **`slot_mapping`** (`Tensor`): Required. Input/output tensor used to map token positions in a sequence to physical memory positions, $slotMapping$ in the formulas. The data type can be `int64`. The shape of this parameter is `[num_queries, 1]` (non-speculative) or `[num_seqs, 1 + spec_num]` (speculative). Empty tensors are not supported. The value must be a positive integer greater than 0.
- **`block_tables`** (`Tensor`): Required. Input/output tensor used to record block allocation for different `block_idx`, $blockTables$ in the formulas. The data type can be `int64`. The shape must have two dimensions. The first dimension equals `num_seqs`, and the second dimension must be greater than the integer division of the maximum value in `seq_lens` by `block_size`.
- **`num_seqs`** (`int`): Required. Records the number of input sequences. The value must be a positive integer greater than 0.
- **`num_queries`** (`int`): Required. Records the number of input queries. The value must be a positive integer greater than 0.
- **`block_size`** (`int`): Required. Size of each block, $blockSize$ in the formulas. The value must be a positive integer greater than 0.
- **`spec_token`** (`Tensor`): Optional. Input tensor used to record token indices in speculative scenarios. The data type can be `int64`. If `spec_token` is omitted (defaults to `None`), non-speculative execution applies. If provided, speculative execution applies and the shape of this parameter is `[num_seqs, spec_num]`. Empty tensors are not supported. The value must be a positive integer greater than 0.
- **`accepted_num`** (`Tensor`): Optional. Input tensor used to record the number of accepted speculative tokens per request in speculative scenarios. The data type can be `int64`.

## Return Values

This API updates `input_tokens`, `input_positions`, `seq_lens`, and `slot_mapping` in place and has no return value.

## Constraints

- The first dimension of `input_tokens`, `input_positions`, `seq_lens`, `slot_mapping`, and `block_tables` must equal `num_seqs`.
- In speculative scenarios, the length of the second dimension of `input_tokens` must be `1 + spec_num`, where `spec_num` is the number of speculative tokens.
- The first dimension of `sampled_token_ids` must equal `num_queries`, and the second dimension must be `1`.
- The second dimension must be greater than the integer division of the maximum value in `seq_lens` by `block_size`.
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
 
input_token = np.random.randint(10, size=(num_seqs,))
sampled_token_id = np.random.randint(10, size=(num_queries,1))
input_position = np.random.randint(10, size=(num_seqs,))
seq_len = np.random.randint(10, size=(num_seqs,))
slot_mapping = np.random.randint(10, size=(num_seqs,))
 
input_tokens = torch.tensor(input_token, dtype=torch.int64, device="npu")
sampled_token_ids = torch.tensor(sampled_token_id, dtype=torch.int64, device="npu")
input_positions = torch.tensor(input_position, dtype=torch.int64, device="npu")
seq_lens = torch.tensor(seq_len, dtype=torch.int64, device="npu")
slot_mappings = torch.tensor(slot_mapping, dtype=torch.int64, device="npu")
 
block_table = np.random.randint(10, size=(num_seqs, torch.max(seq_lens.cpu()) // block_size + 1))
block_tables = torch.tensor(block_table, dtype=torch.int64, device="npu")
 
torch_npu.npu_advance_step_flashattn(input_tokens, sampled_token_ids, input_positions, seq_lens, slot_mappings, block_tables, num_seqs, num_queries, block_size)
```

Speculative scenarios:

```python
import numpy as np

import torch
import torch_npu

num_seqs = 16
block_size = 8
spec_num = 2

input_token = np.random.randint(10, size=(num_seqs, 1 + spec_num))
sampled_token_id = np.random.randint(10, size=(num_seqs, 1 + spec_num))
input_position = np.random.randint(10, size=(num_seqs, 1 + spec_num))
seq_len = np.random.randint(10, size=(num_seqs, 1 + spec_num))
slot_mapping = np.random.randint(10, size=(num_seqs, 1 + spec_num))
spec_token = np.random.randint(10, size=(num_seqs, spec_num))
accepted_num = np.random.randint(10, size=(num_seqs,))

input_tokens = torch.tensor(input_token, dtype=torch.int64, device="npu")
sampled_token_ids = torch.tensor(sampled_token_id, dtype=torch.int64, device="npu")
input_positions = torch.tensor(input_position, dtype=torch.int64, device="npu")
seq_lens = torch.tensor(seq_len, dtype=torch.int64, device="npu")
slot_mappings = torch.tensor(slot_mapping, dtype=torch.int64, device="npu")
spec_tokens = torch.tensor(spec_token, dtype=torch.int64, device="npu")
accepted_nums = torch.tensor(accepted_num, dtype=torch.int64, device="npu")

block_table = np.random.randint(10, size=(num_seqs, torch.max(seq_lens.cpu()) // block_size + 1))
block_tables = torch.tensor(block_table, dtype=torch.int64, device="npu")

torch_npu.npu_advance_step_flashattn(input_tokens, sampled_token_ids, input_positions,
                                     seq_lens, slot_mappings, block_tables, num_seqs,
                                     num_seqs, block_size, spec_token=spec_tokens, accepted_num=accepted_nums)
```

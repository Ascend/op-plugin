# torch_npu.npu_advance_step_flashattn

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>                             |    √     |
|<term>Atlas A2 训练系列产品</term>                              | √   |

## 功能说明

- API功能：在NPU上实现vLLM库中advance_step_flashattn的功能，在每个生成步骤中原地更新`input_tokens`，`input_positions`，`seq_lens`和`slot_mapping`。
- 非投机场景计算公式：$blockIdx$是当前代码被执行的核的索引。
$$
blockTablesStride = blockTables.stride(0) \\
inputTokens[blockIdx] = sampledTokenIds[blockIdx]  \\
inputPositions[blockIdx] = seqLens[blockIdx] \\
seqLens[blockIdx] = seqLens[blockIdx] + 1 \\

slotMapping[blockIdx] = ({blockTables}[blockIdx] + blockTablesStride * blockIdx) * blockSize + (seqLens[blockIdx]\%blockSize)
$$
- 投机场景计算公式：$i$是当前代码被执行的请求的索引。
$$
lastToken = \text{last valid token of each request in }sampledTokenIds \\
blockTablesStride = blockTables.stride(0) \\
inputTokens[:numSeqs, 0] = lastToken  \\
inputTokens[:numSeqs, 1:] = specToken  \\
inputPositions[i] = inputPositions[i] + 1 + acceptedNum[i] \\
seqLens[i] = inputPositions[i] + 1 \\
slotMapping[i] = ({blockTables}[i] + blockTablesStride * i) * blockSize + (inputPositions[i]\%blockSize)
$$
## 函数原型

```
torch_npu.npu_advance_step_flashattn(input_tokens, sampled_token_ids, input_positions, seq_lens, slot_mapping, block_tables, num_seqs, num_queries, block_size) -> ()
```

## 参数说明

- **input_tokens** (`Tensor`)：必选参数，输入/输出张量，对应公式中的输出$inputTokens$，用于更新vLLM模型中的token值；数据类型支持`int64`；如果是非投机场景，shape为[num_seqs,]，如果是投机场景，shape为[num_seqs, 1 + spec_num]；第一维长度与`num_seqs`相同，不支持空tensor，取值范围为大于0的正整数。
- **sampled_token_ids** (`Tensor`)：必选参数，输入张量，对应公式中的输入$sampledTokenIds$，用于储存token_id；数据类型支持`int64`；如果是非投机场景，shape为[num_queries, 1]，第二维长度是1；如果是投机场景，shape为[num_seqs, 1 + spec_num]；第一维长度与`num_queries`相同，第二维长度是1，不支持空tensor，取值范围为大于0的正整数。
- **input_positions** (`Tensor`)：必选参数，输入/输出张量，对应公式中的输出$inputPositions$，用于记录token的index；数据类型支持`int64`；如果是非投机场景，shape为[num_queries, 1]，第二维长度是1；如果是投机场景，shape为[num_seqs, 1 + spec_num]；第一维长度与`num_seqs`相同，不支持空tensor，取值范围为大于0的正整数。
- **seq_lens** (`Tensor`)：必选参数，输入/输出张量，对应公式中的输入/输出$seqLens$，用于记录不同block_idx下seq的长度；数据类型支持`int64`；如果是非投机场景，shape为[num_queries, 1]，第二维长度是1；如果是投机场景，shape为[num_seqs, 1 + spec_num]；第一维长度与`num_seqs`相同，不支持空tensor，取值范围为大于0的正整数。
- **slot_mapping** (`Tensor`)：必选参数，输入/输出张量，对应公式中的输出$slotMapping$，用于将token值在序列中的位置映射到物理位置；数据类型支持`int64`；如果是非投机场景，shape为[num_queries, 1]，第二维长度是1；如果是投机场景，shape为[num_seqs, 1 + spec_num]；第一维长度与`num_seqs`相同，不支持空tensor，取值范围为大于0的正整数。
- **block_tables** (`Tensor`)：必选参数，输入/输出张量，对应公式中的输入$blockTables$，用于记录不同block_idx下block的大小；数据类型支持`int64`；shape为二维，第一维长度与`num_seqs`相同，第二维长度需要大于`seq_lens`中最大值除以`block_size`的整数部分，不支持空tensor，取值范围为大于0的正整数。
- **num_seqs** (`int`)：必选参数，记录输入的seq数量；取值范围为大于0的正整数。
- **num_queries** (`int`)：必选参数，记录输入的query数量；取值范围为大于0的正整数。
- **block_size** (`int`)：必选参数，对应公式中的$blockSize$，每个block的大小；取值范围为大于0的正整数。
- **spec_token** (`Tensor`): 可选参数，输入张量，用于记录投机场景下当前的token的idx。数据类型支持`int64`；spec_token为空时，则为非投机场景，默认为`None`；`spec_token`不为空时，则为投机场景，shape为[num_seqs, spec_num]；不支持空tensor，必须为大于0的正整数。
- **accepted_num** (`Tensor`): 可选参数，输入张量，用于记录投机场景下每个request接受的投机的数量。数据类型支持`int64`。

## 返回值说明

此接口将原地更新`input_tokens`，`input_positions`，`seq_lens`和`slot_mapping`的值，无返回值。

## 约束说明

- 输入`input_tokens`，`input_positions`，`seq_lens`，`slot_mapping`和`block_tables`的第一维长度与`num_seqs`相同。
- 投机场景下，输入`input_tokens`的第二维长度为`1 + spec_num`。
- 输入`sampled_token_ids`的第一维长度与`num_queries`相同且第二维长度为1。
- 输入`block_tables`的shape的第二维长度大于`seq_lens`中最大值除以`block_size`的整数部分。
- 非投机场景下，输入`num_seqs`必须大于输入`num_queries`；投机场景下，`num_queries`与`num_seqs`相同。
- 该接口仅限推理场景使用，无反向函数。

## 调用示例

非投机场景：
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

投机场景：
```python
import numpy as np

import torch
import torch_npu

num_seqs = 16
block_size = 8
spec_num = 2

input_token = np.random.randint(10, size=(num_seqs, 1 + spec_num))
sampled_token_id = np.random.randint(10, size=(num_seqs, 1 + spec_num))
input_position = np.random.randint(10, size=(num_seqs,))
seq_len = np.random.randint(10, size=(num_seqs,))
slot_mapping = np.random.randint(10, size=(num_seqs,))
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
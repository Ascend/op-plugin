# torch_npu.npu_advance_step_flashattn

## 功能说明

- API功能：在NPU上实现vLLM库中advance_step_flashattn的功能，在每个生成步骤中原地更新input_tokens，input_positions，seq_lens和slot_mapping。
- 计算公式：$blockIdx$是当前代码被执行的核的$index$。
$$
blockTablesStride = blockTables.stride(0) \\
inputTokens[blockIdx] = sampledTokenIds[blockIdx]  \\
inputPositions[blockIdx] = seqLens[blockIdx] \\
seqLens[blockIdx] = seqLens[blockIdx] + 1 \\

slotMapping[blockIdx] = ({blockTables}[blockIdx] + blockTablesStride * blockIdx) * blockSize + (seqLens[blockIdx]\%blockSize)
$$
## 函数原型

```
torch_npu.npu_advance_step_flashattn(Tensor(a!) input_tokens, Tensor sampled_token_ids, Tensor(b!) input_positions, Tensor(c!) seq_lens, Tensor(d!) slot_mapping, Tensor block_tables, int num_seqs, int num_queries, int block_size) -> ()
```

## 参数说明

- **input_tokens** (`Tensor`)：必选张量，输入/输出张量，公式中的输出$inputTokens$，用于更新vLLM模型中的token值；数据类型支持int64类型；shape为[num_seqs,]，第一维长度与num_seqs相同，不支持空tensor，取值范围为大于0的正整数。
- **sampled_token_ids** (`Tensor`)：必选张量，输入张量，公式中的输入$sampledTokenIds$，用于储存token_id；数据类型支持int64类型；shape为[num_queries, 1]，第一维长度与num_queries相同，第二维长度是1，不支持空tensor，取值范围为大于0的正整数。
- **input_positions** (`Tensor`)：必选张量，输入/输出张量，公式中的输出$inputPositions$，用于记录token的index；数据类型支持int64类型；shape为[num_seqs,]，第一维长度与num_seqs相同，不支持空tensor，取值范围为大于0的正整数。
- **seq_lens** (`Tensor`)：必选张量，输入/输出张量，公式中的输入/输出$seqLens$，用于记录不同block_idx下seq的长度；数据类型支持int64类型；shape为[num_seqs,]，第一维长度与num_seqs相同，不支持空tensor，取值范围为大于0的正整数。
- **slot_mapping** (`Tensor`)：必选张量，输入/输出张量，公式中的输出$slotMapping$，用于将token值在序列中的位置映射到物理位置；数据类型支持int64类型；shape为[num_seqs,]，第一维长度与num_seqs相同，不支持空tensor，取值范围为大于0的正整数。
- **block_tables** (`Tensor`)：必选张量，输入/输出张量，公式中的输入$blockTables$，用于记录不同block_idx下block的大小；数据类型支持int64类型；shape为二维，第一维长度与num_seqs相同，第二维长度需要大于seq_lens中最大值除以block_size的整数部分，不支持空tensor，取值范围为大于0的正整数。
- **num_seqs** (`int`)：必选张量，记录输入的seq数量；取值范围为大于0的正整数。
- **num_queries** (`int`)：必选张量，记录输入的query数量；取值范围为大于0的正整数。
- **block_size** (`int`)：必选张量，对应公式中的$blockSize$，每个block的大小；取值范围为大于0的正整数。

## 返回值说明

此接口将原地更新input_tokens，input_positions，seq_lens和slot_mapping的值，无返回值。

## 约束说明

- 输入input_tokens，input_positions，seq_lens，slot_mapping和block_tables的第一维长度与num_seqs相同。
- 输入sampled_token_ids的第一维长度与num_queries相同且第二维长度为1。
- 输入block_tables的shape的第二维长度大于seq_lens中最大值除以block_size的整数部分。
- 输入num_seqs必须大于输入num_queries。
- 该接口仅限推理场景使用，无反向函数。

## 支持的型号

- <term>Atlas A2 训练系列产品</term>

- <term>Atlas A3 训练系列产品</term>

## 调用示例

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


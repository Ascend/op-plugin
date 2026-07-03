# torch_npu.npu.graph_task_group_begin

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A3 inference products</term>  | √  |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas A2 inference products</term>|    √     |

## Function

Marks the beginning of a task group in `NPUGraph` scenarios.

## Prototype

```python
torch_npu.npu.graph_task_group_begin(stream) -> None
```

## Parameters

**`stream`** (`torch_npu.npu.Stream`): Required. Stream in graph capture mode.

## Return Values

None

## Constraints

In the graph capture phase, use this API together with [torch_npu.npu.graph_task_group_end](torch_npu-npu-graph_task_group_end.md) to generate a task group handle.

## Example

```python
import torch
import torch_npu

torch.npu.set_device(0)

length = [29]
length_new = [100]
scale = 1 / 0.0078125
query = torch.randn(1, 32, 1, 128, dtype=torch.float16).npu()
key = torch.randn(1, 32, 1, 128, dtype=torch.float16).npu()
value = torch.randn(1, 32, 1, 128, dtype=torch.float16).npu()

stream = None
update_stream = torch.npu.Stream()

handle = None
event = torch.npu.ExternalEvent()
workspace = None
output = None
softmax_lse = None

res = torch_npu.npu_fused_infer_attention_score(
    query, key, value, num_heads=32,
    input_layout="BNSD", scale=scale, pre_tokens=65535, next_tokens=65535, softmax_lse_flag=False,
    actual_seq_lengths=length_new)
print(f"res: {res}")


with torch.no_grad():
    g = torch_npu.npu.NPUGraph()
    with torch.npu.graph(g):
        stream = torch_npu.npu.current_stream()
        workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
            query, key, value, num_heads=32, input_layout="BNSD", scale=scale, pre_tokens=65535, next_tokens=65535,
            softmax_lse_flag=False, actual_seq_lengths=length)
        output = torch.empty(1, 32, 1, 128, dtype=torch.float16, device="npu")
        softmax_lse = torch.empty(1, dtype=torch.float16, device="npu")

        event.wait(stream)
        event.reset(stream)
        torch.npu.graph_task_group_begin(stream)
        torch_npu.npu_fused_infer_attention_score.out(
            query, key, value, workspace=workspace, num_heads=32,
            input_layout="BNSD", scale=scale, pre_tokens=65535, next_tokens=65535, softmax_lse_flag=False,
            actual_seq_lengths=length, out=[output, softmax_lse])
        handle = torch.npu.graph_task_group_end(stream)

    with torch.npu.stream(update_stream):
        torch.npu.graph_task_update_begin(update_stream, handle)
        torch_npu.npu_fused_infer_attention_score.out(
            query, key, value, workspace=workspace, num_heads=32,
            input_layout="BNSD", scale=scale, pre_tokens=65535, next_tokens=65535, softmax_lse_flag=False,
            actual_seq_lengths=length_new, out=[output, softmax_lse])
        torch.npu.graph_task_update_end(update_stream)
        event.record(update_stream)

    g.replay()

    print(f"output: {output}")
    print(f"softmax_lse: {softmax_lse}")

    print(f"equal: {torch.equal(output, res[0])}")


```

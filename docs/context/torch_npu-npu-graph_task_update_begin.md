# torch_npu.npu.graph_task_update_begin
## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A3 推理系列产品</term>   | √  |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |


## 功能说明

NPUGraph场景下，用于标记待更新任务的起始。

## 函数原型

```
torch_npu.npu.graph_task_update_begin(stream, handle) -> None
```

## 参数说明

- **stream** (`torch_npu.npu.Stream`)：必选参数，指定图更新状态的Stream。
- **handle**：必选参数，任务组句柄，由[torch_npu.npu.graph_task_group_end](torch_npu-npu-graph_task_group_end.md)接口生成。

## 返回值说明

无

## 约束说明

- 图捕获阶段，提前调用[torch_npu.npu.graph_task_group_begin](torch_npu-npu-graph_task_group_begin.md)与[torch_npu.npu.graph_task_group_end](torch_npu-npu-graph_task_group_end.md)配合使用生成任务组handle。
- 图更新阶段，与[torch_npu.npu.graph_task_update_end](torch_npu-npu-graph_task_update_end.md)配合使用标记待更新任务的起止。
- 图更新阶段的流同图捕获阶段的流必须不同。

## 调用示例
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


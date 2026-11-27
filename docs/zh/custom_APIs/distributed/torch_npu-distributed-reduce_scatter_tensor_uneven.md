# （beta）torch_npu.distributed.reduce_scatter_tensor_uneven

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品/Atlas 800I A2推理服务器</term>：支持
<!-- end id2 -->
<!-- npu="310p" id3 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id3 -->

## 功能说明

参考原生接口`torch.distributed.reduce_scatter_tensor`功能，具体请参考[https://pytorch.org/docs/stable/distributed.html\#torch.distributed.reduce_scatter_tensor](https://pytorch.org/docs/stable/distributed.html#torch.distributed.reduce_scatter_tensor)。`torch_npu.distributed.reduce_scatter_tensor_uneven`接口新增支持零拷贝和非等长切分功能。

## 函数原型

```python
torch_npu.distributed.reduce_scatter_tensor_uneven(output, input, input_split_sizes=None, op=dist.ReduceOp.SUM, group=None, async_op=False) -> Optional[torch.distributed.distributed_c10d.Work]
```

## 参数说明

- **output** (`Tensor`)：必选参数，接收本rank规约后的数据。第0维大小须为`input_split_sizes[rank]`，其余维度与`input`一致。等长切分时，第0维大小为`input.size(0) // world_size`。
- **input** (`Tensor`)：必选参数，输入Tensor，用于提供计算数据，`input`的shape为所有卡上`output`的shape拼接大小。
- **input_split_sizes** (`List[int]`)：可选参数，输入Tensor的第0维切分大小。默认值`None`表示等长切分，此时第0维大小须能被组内进程数整除。显式提供时，数组长度须等于组内进程数，各项之和等于`input.size(0)`。
    - `input_split_sizes`元素之和与`input`的0维不一致时报错：RuntimeError: Split sizes doesn't match total dim 0 size。
    - `input_split_sizes`元素个数与`group`的size不一致时报错：RuntimeError: Number of tensor splits not equal to group size。
- **op** (`torch._C._distributed_c10d.ReduceOp.ReduceOpType`)：可选参数，规约方式。可用`dist.ReduceOp.SUM`（求和，默认值）、`MIN`（最小值）、`MAX`（最大值）和`AVG`（平均值，使用浮点Tensor）。
- **group** (`torch.distributed.distributed_c10d.ProcessGroup`)：可选参数，分布式进程组，默认值None。
- **async_op** (`bool`)：可选参数，是否异步调用，默认值False。

## 返回值说明

`async_op=True`时返回工作句柄，可调用`wait()`等待通信完成。`async_op=False`时等待通信完成后返回`None`。结果写入`output`。

## 约束说明

<!-- npu="950" id4 -->
- 针对<term>Ascend 950DT系列产品</term>，`input`和`output`支持`torch.bfloat16`、`torch.float16`、`torch.float32`、`torch.int8`、`torch.int32`数据类型。
<!-- end id4 -->

- 此接口仅可在单机场景下使用。

- `input_split_sizes`元素之和等于`input`的0维；`input_split_sizes`元素个数等于`group`的size。

## 调用示例

创建以下文件test.py并保存。

两个rank分别为长度为2和3的结果分配输出Tensor。每个rank提供同样形状的输入，使用默认的求和操作。

```python
import os
import torch
import torch_npu
import torch.distributed as dist
dist.init_process_group(backend="hccl")
rank = int(os.getenv('LOCAL_RANK'))
torch.npu.set_device(rank)
input_split_sizes = [2, 3]
input_tensor = torch.ones(sum(input_split_sizes), dtype=torch.int32).npu()
output_tensor = torch.zeros(input_split_sizes[rank], dtype=torch.int32).npu()
torch_npu.distributed.reduce_scatter_tensor_uneven(
    output_tensor,
    input_tensor,
    input_split_sizes=input_split_sizes,
    async_op=False
)
```

通信完成后，rank 0的输出为 `[2, 2]`，rank 1的输出为 `[2, 2, 2]`。

执行如下命令。

```bash
torchrun --nproc-per-node=2 test.py
```

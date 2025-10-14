# （beta）torch_npu.distributed.reduce_scatter_tensor_uneven
## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品</term>            |    √     |
|<term>Atlas 推理系列产品</term>                                       |    √     |

## 功能说明

参考原生接口`torch.distributed.reduce_scatter_tensor`功能，具体请参考[https://pytorch.org/docs/stable/distributed.html\#torch.distributed.reduce_scatter_tensor](https://pytorch.org/docs/stable/distributed.html#torch.distributed.reduce_scatter_tensor)。`torch_npu.distributed.reduce_scatter_tensor_uneven`接口新增支持零拷贝和非等长切分功能。

## 函数原型

```
torch_npu.distributed.reduce_scatter_tensor_uneven(output, input, input_split_sizes =None, op=dist.ReduceOp.SUM, group=None, async_op=False) -> torch.distributed.distributed_c10d.Work
```


## 参数说明

- **output** (`Tensor`)：必选参数，输出Tensor，用于接收计算数据。
- **input** (`Tensor`)：必选参数，输入Tensor，用于提供计算数据，`input`的shape为所有卡上`output`的shape拼接大小。
- **input_split_sizes** (`List[int]`)：可选参数，输入tensor的0维分割数组，默认值None；元素个数需要与当前调用的group的size一致；元素之和需要与input的0维大小一致。
    - `input_split_sizes`元素之和与`input`的0维不一致时报错：RuntimeError: Split sizes doesn't match total dim 0 size。
    - `input_split_sizes`元素个数与`group`的size不一致时报错：RuntimeError: Number of tensor splits not equal to group size。
- **op** (`torch._C._distributed_c10d.ReduceOp.ReduceOpType`)：可选参数，reduce算子，用于控制计算逻辑，默认值dist.ReduceOp.SUM。
- **group** (`torch.distributed.distributed_c10d.ProcessGroup`)：可选参数，分布式进程组，默认值None。
- **async_op** (`bool`)：可选参数，是否异步调用，默认值False。


## 返回值说明

该函数直接返回为进行计算时的工作句柄，实际计算结果传给output。
`output`：类型为Tensor，其shape无特殊约束。


## 约束说明

- 此接口仅可在单机场景下使用。

- `input_split_sizes`元素之和等于`input`的0维；`input_split_sizes`元素个数等于`group`的size。


## 调用示例

创建以下文件test.py并保存。

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

执行如下命令。

```
torchrun --nproc-per-node=2 test.py
```


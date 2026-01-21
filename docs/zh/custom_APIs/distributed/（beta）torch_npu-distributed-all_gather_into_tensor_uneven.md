# （beta）torch_npu.distributed.all_gather_into_tensor_uneven
## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品</term>            |    √     |
|<term>Atlas 推理系列产品</term>                                       |    √     |

## 功能说明

参考原生接口`torch.distributed.all_gather_into_tensor`功能，具体请参考[https://pytorch.org/docs/stable/distributed.html\#torch.distributed.all_gather_into_tensor](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather_into_tensor)。`torch_npu.distributed.all_gather_into_tensor_uneven`接口新增支持零拷贝和非等长切分功能。

## 函数原型

```
torch_npu.distributed.all_gather_into_tensor_uneven(output, input, output_split_sizes =None, group=None, async_op=False) -> torch.distributed.distributed_c10d.Work
```


## 参数说明

- **output** (`Tensor`)：输出Tensor，用于接收计算数据。
- **input** (`Tensor`)：输入Tensor，用于提供计算数据，`input`的shape无特殊约束。
- **output_split_sizes** (`List[int]`)：输出tensor的0维分割数组，默认值None；元素个数需要与当前调用的group的size一致；元素之和需要与output的0维大小一致。
    - `output_split_sizes`元素之和与`output`的0维不一致时报错：RuntimeError: Split sizes doesn't match total dim 0 size。
    - `output_split_sizes`元素个数与`group`的size不一致时报错：RuntimeError: Number of tensor splits not equal to group size。
- **group** (`torch.distributed.distributed_c10d.ProcessGroup`)：分布式进程组，默认值None。
- **async_op** (`bool`)：是否异步调用，默认值False。

## 返回值说明

`output`的shape为所有卡上`input`的shape拼接大小。


## 约束说明

- 此接口仅可在单机场景下使用。

- `output_split_sizes`元素之和等于`output`的0维；`output_split_sizes`元素个数等于`group`的size。


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
 
output_split_sizes = [2, 3]
input_tensors = [torch.tensor([1, 2], dtype=torch.int32).npu(), torch.tensor([4, 5, 6], dtype=torch.int32).npu()]
output_tensor = torch.zeros(sum(output_split_sizes), dtype=torch.int32).npu()
 
torch_npu.distributed.all_gather_into_tensor_uneven(
    output_tensor,
    input_tensors[rank],
    output_split_sizes=output_split_sizes,
    async_op=False
)
```

执行如下命令。

```
torchrun --nproc-per-node=2 test.py
```


# torch\_npu.npu\_swiglu\_quant

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>  | √   |

## 功能说明

-   API功能：在swiglu激活函数后添加quant操作，实现输入`x`的`SwiGluQuant`计算，支持`int8`或`int4`量化输出，支持MoE场景和非MoE场景（`group_index`为空），支持分组量化，支持动态/静态量化。
-   计算公式：
    -   swiglu激活：对`x`做swiglu（通过`activate_left`控制左右激活），以左激活为例:
        $$
        swiglu(x)=swish(x[:,0:N])*x[:,N:2N]
        $$

    -   group分组（支持`cumsum`和`count`两种模式）：

        对输入`x`经过swiglu计算后的结果进行分组计算，`group_index`表示每个group分组的tokens数，每组使用不同的量化`smooth_scales`。

        举例说明：假设x.shape=\[6, 2N\]，在`cumsum`模式下，`group_index`的shape为\[2, 4, 6\]，表示有3个group，对应的`smooth_scales`维度为\[3, H\]。每个group数据使用不同的`smooth_scales`分别做quant量化操作。

        -   group0=x\[0:2, :\]，scale0=smooth_scales\[0, :\]
        -   group1=x\[2:4, :\]，scale1=smooth_scales\[1, :\]
        -   group2=x\[4:6, :\]，scale2=smooth_scales\[2, :\]

        如果是在`count`模式下，`group_index`需要设置成\[2, 2, 2\]，表示每个group都是2个tokens。

    -   quant量化：
        1.  先进行smooth量化，其中out为上一步swiglu激活的输出。
            $$
            out=out*smooth\_scales
            $$

        2.  动态/静态量化操作：对激活后的结果进行量化，以动态量化（dynamic_quant）为例，详细数学公式请参考[aclnnSwGluQuantV2](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/aolapi/context/aclnnSwiGluQuantV2.md)。
            $$
            out,scale=dynamic\_quant(out)
            $$

## 函数原型

```
torch_npu.npu_swiglu_quant(x, *, smooth_scales=None, offsets=None, group_index=None, activate_left=False, quant_mode=0， group_list_type=0, dst_type=None) -> (Tensor, Tensor)
```

## 参数说明

>**说明：**<br>
>Tensor中shape使用的变量说明：
>-   G：表示group_index分组数量，取值\>0。
>-   N：计算输入`x`的最后一维大小的二分之一，取值\>0。

-   **x** (`Tensor`)：必选参数，表示目标张量。数据类型支持`float16`、`bfloat16`、`float32`，支持非连续的`Tensor`，数据格式为$ND$，`x`的维数必须大于1维，尾轴为偶数且长度不超过8192，当`dst_type`为`int4`量化时，`x`的最后一维需要为4的倍数。
- <strong>*</strong>：必选参数，代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
-   **smooth\_scales** (`Tensor`)：可选参数，表示smooth量化系数。数据类型支持`float32`，支持非连续的`Tensor`，数据格式为$ND$。shape支持[G, N]，[G, ]。
-    **offsets** (`Tensor`)：可选参数，表示量化中的偏移项，该参数在动态量化场景下不生效，传入`None`即可。静态量化场景下：数据类型支持`float`，支持非连续的`Tensor`，数据格式为$ND$。per_channel模式下shape支持[G, N]，per_tensor模式下shape支持[G, ]，且数据类型和shape需要与`smooth_scales`保持一致。
-   **group\_index** (`Tensor`)：可选参数，当前支持`cumsum`和`count`两种模式，要求为1维张量，数据类型支持`int32`，数据格式$ND$，shape支持[G, ]，`group_index`内元素要求为非递减，且最大值不得超过输入`x`的除最后一维之外的所有维度大小之积。
-   **activate\_left** (`bool`)：可选参数，swiglu流程中是否进行左激活，默认值为`False`。
-   **quant\_mode** (`int`)：可选参数，表示量化类型，默认值为`0`。`0`表示静态量化，`1`表示动态量化。
-   **group\_list\_type** (`int`)：可选参数，表示`group_index`类型，默认值为`0`。`0`表示`cumsum`模式，`1`表示`count`模式。
-   **dst\_type** (`ScalarType`)：可选参数，表示输出量化类型，支持`int8`和`int4`，传`None`时当做`int8`处理，默认值为`None`。


## 返回值说明

-   **out** (`Tensor`)：表示量化后的输出tensor。数据类型支持`int8`和`int4`，支持非连续的`Tensor`，数据格式为$ND$。
-   **scale** (`Tensor`)：表示量化的scale参数，计算输出scale的shape与计算输入`x`相比，无最后一维，其余维度与计算输入`x`保持一致，数据类型支持`float32`，数据格式为$ND$。

## 调用示例
```python
import os
import shutil
import unittest

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNPUSwigluQuant(TestCase):
    def test_npu_swiglu_quant(self, device="npu"):
        batch_size = 4608
        hidden_size = 2048
        x_shape = (batch_size, hidden_size)
        input_data = np.random.randn(*x_shape).astype(np.float32)

        quant_mode = 1
        group_list_type = 0
        dst_type = torch.int8
        activate_left = False
        offsets = None
        num_groups = 8
        group_sizes = batch_size // num_groups
        group_index = [(i + 1) * group_sizes for i in range(num_groups)]
        smooth_scales = np.random.randn(num_groups, hidden_size // 2).astype(np.float32)

        device = "npu"
        npu_x = torch.from_numpy(input_data).to(device)
        npu_group_index = torch.from_numpy(np.array(group_index)).to(device)
        npu_smooth_scales = torch.from_numpy(smooth_scales).to(device)
        result = torch_npu.npu_swiglu_quant(
            npu_x,
            smooth_scales=npu_smooth_scales,
            offsets=offsets,
            group_index=npu_group_index,
            activate_left=False,
            quant_mode=quant_mode,
            group_list_type=group_list_type,
            dst_type=dst_type
        )

if __name__ == "__main__":
    run_tests()
```
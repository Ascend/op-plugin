# torch_npu.npu_cross_entropy_loss

## 函数原型

```
torch_npu.npu_cross_entropy_loss(Tensor input, Tensor target, Tensor? weight=None, str reduction="mean", int ignore_index=-100, float label_smoothing=0.0, float lse_square_scale_for_zloss=0.0, bool return_zloss=False) -> (Tensor, Tensor, Tensor, Tensor)
```

## 功能说明

将原生CrossEntropyLoss中的log_softmax和nll_loss融合，降低计算时使用的内存。接口允许计算zloss。

## 参数说明

- input: Device侧的Tensor类型，表示输入；数据类型支持torch.float16、torch.float32、torch.bfloat16类型；shape为[N, C]，N为批处理大小，C为标签数，必须大于0。
- target: Device侧的Tensor类型，表示标签；数据类型支持INT64类型；shape为[N]，与input第零维相同，取值范围[0, C)。
- weight: Device侧的Tensor类型，表示每个类别指定的缩放权重，可选；数据类型支持torch.float32类型；shape为[C]，与input第一维相同，取值范围(0, 1]，不指定值时默认全一。
- reduction: str类型，表示loss的归约方式；支持范围["mean", "sum", "none"]，默认为"mean"。
- ignore_index: int类型，指定忽略的标签；数值必须小于C，当小于0时视为无忽略标签；默认值为-100。
- label_smoothing: float类型，表示计算loss时的平滑量；取值范围[0.0, 1.0)；默认值为0.0。
- lse_square_scale_for_zloss: float类型，表示计算zloss所需要的scale；取值范围[0.0, 1.0)；默认值为0.0；当前暂不支持。
- return_zloss: bool类型，控制是否返回zloss；设置为True将返回zloss，设置为False时不返回zloss；默认值为False；当前暂不支持。

## 输出说明

- loss：Device侧的Tensor类型，表示输出损失；数据类型与input相同；reduction为"none"时shape为[N]，与input第零维一致，否则shape为[1]。
- log_prob: Device侧的Tensor类型，输出给反向计算的输出；数据类型与input相同；shape为[N, C]，与input一致。
- zloss: Device侧的Tensor类型，表示辅助损失；数据类型与input相同；shape与loss一致；当return_zloss为True时输出zloss，否则将返回空tensor；当前暂不支持。
- lse_for_zloss: Device侧的Tensor类型，zloss场景输出给反向计算的输出；数据类型与input相同；shape为[N]，与input第零维一致；lse_square_scale_for_zloss不为0.0时将返回该输出，否则将返回空tensor；当前暂不支持。

## 约束说明

- 输入shape中N取值范围(0, 200000]。
- 当input.requires_grad=True时，sum/none模式下不支持修改label_smoothing的默认值；mean模式下不支持修改所有含默认值的入参的值，包括weight，reduction，ignore_index，label_smoothing，lse_square_scale_for_zloss和return_zloss。
- 属性lse_square_scale_for_zloss与return_zloss暂未使能。
- 输出zloss与lse_for_zloss暂未使能。
- 输出中仅loss和zloss支持梯度计算。

## 支持的型号

- <term>Atlas A2 训练系列产品</term>

- <term>Atlas A3 训练系列产品</term>

## 调用示例

```python
import torch
import torch_npu
 
N = 4096
C = 8080
input = torch.randn(N, C).npu()
target = torch.arange(0, N).npu()
 
loss, log_prob, _, _ = torch_npu.npu_cross_entropy_loss(input, target)
```


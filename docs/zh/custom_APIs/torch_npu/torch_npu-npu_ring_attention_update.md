# torch_npu.npu_ring_attention_update

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>                        |    √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>      |    √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>      |    √     |

## 功能说明

- API功能：将两次 FlashAttention 的输出结果按照对应的 softmax max 和 softmax sum 做增量更新，得到新的 attention 输出、softmax max 和 softmax sum。
- 计算公式：

  $$
  softmax\_max = max(prev\_softmax\_max, cur\_softmax\_max)
  $$

  $$
  softmax\_sum = prev\_softmax\_sum \times exp(prev\_softmax\_max - softmax\_max) + cur\_softmax\_sum \times exp(cur\_softmax\_max - softmax\_max)
  $$

  $$
  attn\_out = prev\_attn\_out \times exp(prev\_softmax\_max - softmax\_max) \times prev\_softmax\_sum / softmax\_sum + cur\_attn\_out \times exp(cur\_softmax\_max - softmax\_max) \times cur\_softmax\_sum / softmax\_sum
  $$

> [!NOTE]
>
> - 该接口底层调用 CANN 的 `aclnnRingAttentionUpdateV2`，支持在 `input_layout="TND"` 场景下通过 `input_softmax_layout` 控制 softmax 相关输入是否采用 `TND` 排布。
> - `input_layout="TND"` 时，`actual_seq_qlen` 为必选参数。
> - `input_softmax_layout` 仅支持 `""`、`"SBH"`、`"TND"` 三种取值。

## 函数原型

```python
torch_npu.npu_ring_attention_update(prev_attn_out, prev_softmax_max, prev_softmax_sum, cur_attn_out, cur_softmax_max, cur_softmax_sum, *, actual_seq_qlen=None, input_layout="SBH", input_softmax_layout="") -> (Tensor, Tensor, Tensor)
```

## 参数说明

- **prev_attn_out** (`Tensor`)：必选参数，第一次 FlashAttention 的输出。数据类型支持 `float16`、`float32`、`bfloat16`，数据格式支持 $ND$，支持非连续的 Tensor。`input_layout="SBH"` 时 shape 为 `[S, B, H]`，`input_layout="TND"` 时 shape 为 `[T, N, D]`。
- **prev_softmax_max** (`Tensor`)：必选参数，第一次 FlashAttention 的 softmax max 结果。数据类型支持 `float32`，数据格式支持 $ND$，支持非连续的 Tensor。`input_layout="SBH"` 时 shape 为 `[B, N, S, 8]`，`input_layout="TND"` 且 `input_softmax_layout="TND"` 时 shape 为 `[T, N, 8]`。最后一维 8 个元素应保持相同且为正数。
- **prev_softmax_sum** (`Tensor`)：必选参数，第一次 FlashAttention 的 softmax sum 结果。数据类型支持 `float32`，数据格式支持 $ND$，支持非连续的 Tensor。shape 需要与 `prev_softmax_max` 一致，最后一维 8 个元素应保持相同且为正数。
- **cur_attn_out** (`Tensor`)：必选参数，第二次 FlashAttention 的输出。数据类型、数据格式和 shape 需要与 `prev_attn_out` 保持一致。
- **cur_softmax_max** (`Tensor`)：必选参数，第二次 FlashAttention 的 softmax max 结果。数据类型支持 `float32`，数据格式支持 $ND$，shape 需要与 `prev_softmax_max` 保持一致。
- **cur_softmax_sum** (`Tensor`)：必选参数，第二次 FlashAttention 的 softmax sum 结果。数据类型支持 `float32`，数据格式支持 $ND$，shape 需要与 `prev_softmax_max` 保持一致。
- <strong>*</strong>：位置参数与关键字参数的分隔符。其之前的参数为位置参数，需按顺序传入；其之后的参数为关键字参数，未显式赋值时使用默认值。
- **actual_seq_qlen** (`Tensor`)：可选参数，表示从 0 开始累计的 query 序列长度前缀和。数据类型支持 `int64`，数据格式支持 $ND$。当 `input_layout="TND"` 时必须传入，且张量中的值需要单调递增至总 token 数。
- **input_layout** (`str`)：可选参数，表示 `prev_attn_out` 和 `cur_attn_out` 的数据排布。支持的取值：
  - `"SBH"`：attention 输入输出按 `[S, B, H]` 排布。
  - `"TND"`：attention 输入输出按 `[T, N, D]` 排布。
  默认值为 `"SBH"`。
- **input_softmax_layout** (`str`)：可选参数，表示 softmax 相关输入的排布方式。支持的取值：
  - `""`：使用默认排布。
  - `"SBH"`：softmax 输入按 SBH 语义排布。
  - `"TND"`：softmax 输入按 TND 语义排布。
  默认值为 `""`。仅在 `input_layout="TND"` 时生效。

## 返回值说明

- **attn_out** (`Tensor`)：更新后的 attention 输出。shape 与数据类型与输入 `prev_attn_out` 保持一致。
- **softmax_max** (`Tensor`)：更新后的 softmax max。shape 与 `prev_softmax_max` 保持一致，数据类型为 `float32`。
- **softmax_sum** (`Tensor`)：更新后的 softmax sum。shape 与 `prev_softmax_sum` 保持一致，数据类型为 `float32`。

## 约束说明

- 该接口支持推理、训练场景下使用。
- 该接口支持图模式。
- `prev_attn_out` 与 `cur_attn_out` 的 shape 和数据类型必须一致。
- `prev_softmax_max`、`prev_softmax_sum`、`cur_softmax_max`、`cur_softmax_sum` 的 shape 必须一致，且数据类型均需为 `float32`。
- `input_layout="TND"` 时，`actual_seq_qlen` 为必选参数。
- `input_layout="TND"` 时，`input_softmax_layout` 才生效，且只支持 `""`、`"SBH"`、`"TND"`。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> 在 `input_layout="TND"` 场景下额外要求：

| 约束项 | 限制 |
| ------ | ---- |
| `N`    | `N <= 256` |
| `D`    | `D <= 768` 且 `D` 为 `64` 的倍数 |

## 调用示例

- 单算子模式调用

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> prev_attn_out = torch.randn((4, 2, 32), dtype=torch.float16, device="npu")
    >>> cur_attn_out = torch.randn((4, 2, 32), dtype=torch.float16, device="npu")
    >>> prev_softmax_max = torch.rand((2, 2, 4, 1), dtype=torch.float32, device="npu").repeat(1, 1, 1, 8)
    >>> prev_softmax_sum = torch.rand((2, 2, 4, 1), dtype=torch.float32, device="npu").repeat(1, 1, 1, 8)
    >>> cur_softmax_max = torch.rand((2, 2, 4, 1), dtype=torch.float32, device="npu").repeat(1, 1, 1, 8)
    >>> cur_softmax_sum = torch.rand((2, 2, 4, 1), dtype=torch.float32, device="npu").repeat(1, 1, 1, 8)
    >>> attn_out, softmax_max, softmax_sum = torch_npu.npu_ring_attention_update(
    ...     prev_attn_out, prev_softmax_max, prev_softmax_sum,
    ...     cur_attn_out, cur_softmax_max, cur_softmax_sum)
    >>> attn_out.shape
    torch.Size([4, 2, 32])
    >>> softmax_max.shape
    torch.Size([2, 2, 4, 8])
    >>> softmax_sum.shape
    torch.Size([2, 2, 4, 8])
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    class Model(torch.nn.Module):
        def forward(self, prev_attn_out, prev_softmax_max, prev_softmax_sum,
                    cur_attn_out, cur_softmax_max, cur_softmax_sum):
            return torch_npu.npu_ring_attention_update(
                prev_attn_out, prev_softmax_max, prev_softmax_sum,
                cur_attn_out, cur_softmax_max, cur_softmax_sum)

    model = Model().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
    outputs = model(prev_attn_out, prev_softmax_max, prev_softmax_sum,
                    cur_attn_out, cur_softmax_max, cur_softmax_sum)
    print(outputs[0].shape)
    ```

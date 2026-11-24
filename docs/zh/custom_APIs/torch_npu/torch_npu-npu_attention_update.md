# torch_npu.npu_attention_update

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->

## 功能说明

- API功能：将各SP域PA算子输出的中间结果`lse`、`local_out`两个局部变量更新成全局结果。

- 计算公式：

  $$
  lse_{max} = \text{max}lse_i
  $$

  $$
  lse = \sum_i \text{exp}(lse_i - lse_{max})
  $$

  $$
  lse_m = lse_{max} + \text{log}(lse)
  $$

  $$
  O = \sum_i O_i \cdot \text{exp}(lse_i - lse_m)
  $$

## 函数原型

```python
torch_npu.npu_attention_update(lse, local_out, update_type) -> (Tensor, Tensor)
```

## 参数说明

- **lse**(`Tensor[]`)：必选参数，表示各SP域的局部lse，对应公式中的$lse_i$，tensorList长度为SP，每个Tensor的shape为$(batch \times seqLen \times headNum)$。也支持packed形态输入：tensorList长度为1，Tensor shape为$(SP, batch \times seqLen \times headNum)$。数据类型支持`torch.float32`，数据格式支持$ND$。支持空Tensor。
- **local_out**(`Tensor[]`)：必选参数，表示各SP域的局部attention out，对应公式中的$O_i$，tensorList长度为SP，每个Tensor的shape为$(batch \times seqLen \times headNum, head\_dim)$。也支持packed形态输入：tensorList长度为1，Tensor shape为$(SP, batch \times seqLen \times headNum, head\_dim)$。数据类型支持`torch.float32`、`torch.float16`、`torch.bfloat16`，数据格式支持$ND$。支持空Tensor。
- **update_type**(`int`)：必选参数，指定执行的操作类型。取值为`0`时，仅输出合并后的out；取值为`1`时，同时输出合并后的out和lse_out。

## 返回值说明

- **out**(`Tensor`)：输出的Tensor，对应公式中的$O$。分片形态输入时shape为$(batch \times seqLen \times headNum, head\_dim)$；packed形态输入时shape不变，仍为$(batch \times seqLen \times headNum, head\_dim)$。数据类型与输入`local_out`中的Tensor数据类型一致，数据格式为$ND$。
- **lse_out**(`Tensor`)：可选输出，对应公式中的$lse_m$。分片形态输入时shape为$(batch \times seqLen \times headNum)$；packed形态输入时shape不变，仍为$(batch \times seqLen \times headNum)$。数据类型为`torch.float32`，数据格式为$ND$。仅当`update_type=1`时为Tensor；`update_type=0`时为`None`。

## 约束说明

- 确定性计算：该接口默认为确定性实现，即对于相同的输入，多次执行会产生相同的结果，确保计算结果的可重复性。
<!-- npu="950" id4 -->
- <term>Ascend 950PR&950DT系列产品</term>：序列并行度 `SP` 取值范围为 [1, 16]。
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3系列产品</term>：序列并行度 `SP` 取值范围为 [1, 128]。
<!-- end id5 -->
<!-- npu="910b" id6 -->
- <term>Atlas A2系列产品</term>：序列并行度 `SP` 取值范围为 [1, 128]。
<!-- end id6 -->
- `head_dim` 取值范围为 [8, 512]，且必须是 8 的倍数。
- `lse` 和 `local_out` 列表不能为空，长度必须一致，且必须采用同一种输入形式。
- 分片形式中，各 `lse[i]` 的 shape 为 `[T]`，各 `local_out[i]` 的 shape 为 `[T,D]`；所有分片的 `T`、`D` 和对应数据类型须保持一致。分片形式不支持非连续的Tensor。
- packed 形式要求 `lse` 中的 Tensor 为二维且连续，`local_out` 中的 Tensor 为三维，支持行 padding 等非连续视图。`lse`和`local_out`的 `P`、`T` 必须一致。
- 支持 `T=0` 的空 Tensor。

## 调用示例

- 单算子模式调用：

    ```python
    import torch
    import torch_npu

    dtype = torch.float32
    N = 4
    head_dim = 32

    lse = [
        torch.randn(N, dtype=dtype, device='npu'),
        torch.randn(N, dtype=dtype, device='npu'),
    ]

    local_out = [
        torch.randn(N, head_dim, dtype=dtype, device='npu'),
        torch.randn(N, head_dim, dtype=dtype, device='npu'),
    ]

    # update_type=0：仅输出合并后的out
    out, lse_out = torch_npu.npu_attention_update(lse, local_out, 0)
    print("out:", out)
    print("out.shape:", out.shape)

    # update_type=1：同时输出合并后的out和lse_out
    out, lse_out = torch_npu.npu_attention_update(lse, local_out, 1)
    print("out:", out)
    print("lse_out:", lse_out)
    ```

- packed形态调用（tensorList长度为1，输入为packed单Tensor，输出与分片形态一致）

    ```python
    import torch
    import torch_npu

    dtype = torch.float32
    P = 4          # SP并行度
    N = 128        # batch * seqLen * headNum
    head_dim = 32

    lse = [torch.randn(P, N, dtype=torch.float32, device='npu')]

    local_out = [torch.randn(P, N, head_dim, dtype=dtype, device='npu')]

    out, lse_out = torch_npu.npu_attention_update(lse, local_out, 1)
    print("out.shape:", out.shape)        # 输出shape为[N, head_dim]
    print("lse_out.shape:", lse_out.shape)  # 输出shape为[N]
    ```

- packed 非连续 `local_out` 调用（每行保留 padding）

    ```python
    import torch
    import torch_npu

    P, T, D = 2, 7, 24
    lse_tensor = torch.randn(P, T, dtype=torch.float32, device="npu")
    storage = torch.randn(P, T, D + 4, dtype=torch.float16, device="npu")
    local_out_tensor = storage[..., :D]
    assert not local_out_tensor.is_contiguous()

    # packed：输入仍包装成长度为 1 的列表。
    out, lse_out = torch_npu.npu_attention_update([lse_tensor], [local_out_tensor], 1)
    assert out.shape == (T, D)
    assert lse_out.shape == (T,)

    ```

- 图模式调用：

    ```python
    import torch
    import torch_npu
    import torchair
    from torchair.configs.compiler_config import CompilerConfig

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, lse, local_out, update_type):
            return torch_npu.npu_attention_update(lse, local_out, update_type=update_type)

    model = Net()
    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    model = torch.compile(model, fullgraph=True, backend=npu_backend, dynamic=False)
    out, lse_out = model(lse, local_out, 1)
    ```

# torch\_npu.npu\_fused\_causal\_conv1d

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| <term>Atlas 350 加速卡</term> | √ |

## 功能说明

- **API功能：**

    对序列执行因果一维卷积。沿序列维度，使用缓存数据（长度为卷积核宽减1）对各序列头部进行padding，确保输出依赖当前及历史输入；卷积完成后，将当前序列尾部的数据（长度为卷积核宽减1）原地更新到缓存。

    支持如下场景：

  - npu\_fused\_causal\_conv1d（prefill，run\_mode=0）：
    - x支持shape为\[cu\_seq\_len, dim\]，其中cu\_seq\_len为batch内所有变长序列拼接后的总长度
    - weight的shape为\[K, dim\]，K固定为3
    - 每个序列卷积前，使用长度为K-1的缓存数据对序列头部进行padding，保证因果性

  - npu\_fused\_causal\_conv1d（prefill，run\_mode=0）：
    - x支持shape为\[cu\_seq\_len, dim\]和\[batch, m+1, dim\]，其中cu\_seq\_len为batch内所有变长序列拼接后的总长度，m为投机Token的个数
    - weight的shape为\[K, dim\]，K固定为3
    - 每个序列卷积前，使用长度为K-1的缓存数据对序列头部进行padding，保证因果性

- **计算公式**：

    K是卷积核宽度，L是原始序列长度，dim是特征维度。

    1. 缓存拼接

        ![](figures/zh-cn_formulaimage_0000002567675999.png)

    2. 因果1维卷积

        ![](figures/zh-cn_formulaimage_0000002567676367.png)

    3. 缓存更新

        ![](figures/zh-cn_formulaimage_0000002536756506.png)

    4. 残差连接（可选）

        ![](figures/zh-cn_formulaimage_0000002536639128.png)

## 函数原型

```python
torch_npu.npu_fused_causal_conv1d(x, weight, conv_states, *, query_start_loc=None, cache_indices=None, initial_state_mode=None, bias=None, num_accepted_tokens=None, activation_mode="None", pad_slot_id=-1, run_mode=0, residual_connection=0) -> Tensor
```

## 参数说明

- **x**（`Tensor`）：必选参数，表示输入序列，即公式中的$x$。数据类型支持`float16`、`bfloat16`，数据格式要求为ND，支持非连续的Tensor。不支持空Tensor。
- **weight**（`Tensor`）：必选参数，表示因果1维卷积核，即公式中的$weight$。数据类型、数据格式与`x`保持一致，不支持非连续的Tensor。不支持空Tensor。
- **conv\_states**（`Tensor`）：必选参数，表示缓存状态张量，存储各序列的历史token数据，各序列计算完成后原地更新，即公式中的$conv\_states$。数据类型、数据格式与`x`保持一致，支持非连续的Tensor。不支持空Tensor。
- \*：代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **query\_start\_loc**（`Tensor`）：可选参数，表示序列起始位置索引，记录各序列在拼接张量`x`中的起始位置。数据类型支持`int32`，数据格式要求为ND，不支持非连续的Tensor。当`x`的shape为2维时，不可省略，默认值为None。
- **cache\_indices**（`Tensor`）：可选参数，表示缓存索引，指定每个序列对应的缓存状态在conv\_states中的索引。数据类型支持`int32`，数据格式要求为ND，不支持非连续的Tensor。默认值为None。
- **initial\_state\_mode**（`Tensor`）：可选参数，初始状态标志，表示各序列是否使用缓存数据，仅prefill场景（run\_mode=0）生效。数据类型支持`int32`，数据格式要求为ND，不支持非连续的Tensor。默认值为None。
- **bias**（`Tensor`）：可选参数，表示卷积的偏置。数据类型和数据格式与x保持一致，不支持非连续的Tensor。默认值为None。
- **num\_accepted\_tokens**（`Tensor`）：可选参数，表示投机解码场景下各batch实际接受的token个数，仅decode场景（run\_mode=1）生效。数据类型支持`int32`，数据格式要求为ND，不支持非连续的Tensor。默认值为None。
- **activation\_mode**（`str`）：可选参数，表示激活函数类型。支持"None"、"silu"、"swish"，默认值为"None"。
- **pad\_slot\_id**（`int`）：可选参数，用于跳过不需要参与计算的batch，默认值为-1。
- **run\_mode**（`int`）：可选参数，表示运行模式。0：prefill场景；1：decode场景，默认值为0。
- **residual\_connection**（`int`）：可选参数，用于判断是否输出结果是否要做残差连接。0：不做残差连接；1：输出为卷积结果与输入x之和（残差连接），默认值为0。

## 返回值说明

**y**（`Tensor`）：表示计算结果，公式中的$y$，数据类型和数据格式与`x`保持一致，不支持非连续的Tensor。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和图模式调用。
- 输入shape限制：
  - prefill场景：
    - x支持2维\[cu\_seq\_len, dim\]。
    - weight必须是2维\[K, dim\]，其中K固定为3。
    - conv\_states必须是3维\[..., K-1, dim\]，第0维大小不固定且大于等于batch。
    - cu\_seq\_len范围\[batch, 65536\]，dim范围\[128, 16384\]且是128的倍数，batch范围\[1, 256\]。

  - decode场景（固定batch）：
    - x支持3维\[batch, seq\_len, dim\]。
    - weight必须是2维\[K, dim\]，其中K固定为3。
    - conv\_states必须是3维\[..., K-1+seq\_len-1, dim\]，第0维大小不固定且大于等于batch。
    - seq\_len范围\[1, 6\]，dim范围\[128, 16384\]且是128的倍数，batch范围\[1, 256\]。

  - decode场景（变长序列）：
    - x支持2维\[cu\_seq\_len, dim\]。
    - weight必须是2维\[K, dim\]，其中K固定为3。
    - conv\_states必须是3维\[..., state\_len, dim\]，第0维大小不固定且大于等于batch，state\_len必须大于所有batch中最大的token个数加K-1。
    - cu\_seq\_len范围\[batch, batch\*6\]，每个batch的token个数范围为\[1, 6\]。dim范围\[128, 16384\]且是128的倍数，batch范围\[1, 256\]。

## 调用示例

- 单算子模式调用
  - prefill场景：

    ```python
    import torch
    import torch_npu
    K = 3
    dim = 128
    batch = 4
    dtype = torch.bfloat16
    weight = torch.randn(K, dim, dtype=dtype).npu()
    seq_lens = [5, 3, 7, 4]
    cu_seq_len = sum(seq_lens)
    x = torch.randn(cu_seq_len, dim, dtype=dtype).npu()
    query_start_loc = torch.tensor([0, 5, 8, 15, 19], dtype=torch.int32).npu()
    num_slots = 8
    conv_states = torch.randn(num_slots, K - 1, dim, dtype=dtype).npu()
    cache_indices = torch.tensor([0, 3, 1, 5], dtype=torch.int32).npu()
    initial_state_mode = torch.tensor([1, 1, 0, 0], dtype=torch.int32).npu()
    out = torch_npu.npu_fused_causal_conv1d(
        x,
        weight,
        conv_states,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        initial_state_mode=initial_state_mode,
        bias=None,
        activation_mode="None",
        pad_slot_id=-1,
        run_mode=0,
        residual_connection=0,
    )
    print(f"output shape: {out.shape}")
    ```

  - decode场景（固定batch）：

    ```python
    import torch
    import torch_npu
    K = 3
    dim = 128
    batch = 4
    seq_len = 3
    dtype = torch.bfloat16
    weight = torch.randn(K, dim, dtype=dtype).npu()
    x = torch.randn(batch, seq_len, dim, dtype=dtype).npu()
    num_slots = 8
    conv_states = torch.randn(num_slots, K-1+seq_len-1, dim, dtype=dtype).npu()
    cache_indices = torch.tensor([0, 2, 4, 6], dtype=torch.int32).npu()
    num_accepted_tokens = torch.tensor([2, 1, 3, 2], dtype=torch.int32).npu()
    out = torch_npu.npu_fused_causal_conv1d(
        x,
        weight,
        conv_states,
        cache_indices=cache_indices,
        num_accepted_tokens=num_accepted_tokens,
        activation_mode="None",
        pad_slot_id=-1,
        run_mode=1,
        residual_connection=0,
    )
    print(f"output shape: {out.shape}")
    ```

  - decode场景（变长序列）：

    ```python
    import torch
    import torch_npu
    K = 3
    dim = 128
    batch = 4
    dtype = torch.bfloat16
    weight = torch.randn(K, dim, dtype=dtype).npu()
    seq_lens = [1, 1, 1, 1]
    cu_seq_len = sum(seq_lens)
    x = torch.randn(cu_seq_len, dim, dtype=dtype).npu()
    query_start_loc = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32).npu()
    num_slots = 8
    conv_states = torch.randn(num_slots, K - 1, dim, dtype=dtype).npu()
    cache_indices = torch.tensor([0, 2, 4, 6], dtype=torch.int32).npu()
    out = torch_npu.npu_fused_causal_conv1d(
        x,
        weight,
        conv_states,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        num_accepted_tokens=None,
        activation_mode="None",
        pad_slot_id=-1,
        run_mode=1,
        residual_connection=0,
    )
    print(f"output shape: {out.shape}")
    ```

- 图模式调用
  - prefill场景：

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    K = 3
    dim = 128
    batch = 4
    dtype = torch.bfloat16
    weight = torch.randn(K, dim, dtype=dtype).npu()
    seq_lens = [5, 3, 7, 4]
    cu_seq_len = sum(seq_lens)
    x = torch.randn(cu_seq_len, dim, dtype=dtype).npu()
    query_start_loc = torch.tensor([0, 5, 8, 15, 19], dtype=torch.int32).npu()
    num_slots = 8
    conv_states = torch.randn(num_slots, K - 1, dim, dtype=dtype).npu()
    cache_indices = torch.tensor([0, 3, 1, 5], dtype=torch.int32).npu()
    initial_state_mode = torch.tensor([1, 1, 0, 0], dtype=torch.int32).npu()
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    def causal_conv1d_prefill(x, weight, conv_states, query_start_loc,
                               cache_indices, initial_state_mode):
        return torch_npu.npu_fused_causal_conv1d(
            x,
            weight,
            conv_states,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            initial_state_mode=initial_state_mode,
            bias=None,
            activation_mode="None",
            pad_slot_id=-1,
            run_mode=0,
            residual_connection=0,
        )
    compiled_func = torch.compile(causal_conv1d_prefill, backend=npu_backend)
    out = compiled_func(x, weight, conv_states, query_start_loc, cache_indices, initial_state_mode)
    print(f"output shape: {out.shape}")
    ```

  - decode场景（固定batch）：

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    K = 3
    dim = 128
    batch = 4
    seq_len = 3
    dtype = torch.bfloat16
    weight = torch.randn(K, dim, dtype=dtype).npu()
    x = torch.randn(batch, seq_len, dim, dtype=dtype).npu()
    num_slots = 8
    conv_states = torch.randn(num_slots, K-1+seq_len-1, dim, dtype=dtype).npu()
    cache_indices = torch.tensor([0, 2, 4, 6], dtype=torch.int32).npu()
    num_accepted_tokens = torch.tensor([2, 1, 3, 2], dtype=torch.int32).npu()
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    def causal_conv1d_decode_3d(x, weight, conv_states, cache_indices, num_accepted_tokens):
        return torch_npu.npu_fused_causal_conv1d(
            x,
            weight,
            conv_states,
            cache_indices=cache_indices,
            num_accepted_tokens=num_accepted_tokens,
            activation_mode="None",
            pad_slot_id=-1,
            run_mode=1,
            residual_connection=0,
        )
    compiled_func = torch.compile(causal_conv1d_decode_3d, backend=npu_backend)
    out = compiled_func(x, weight, conv_states, cache_indices, num_accepted_tokens)
    print(f"output shape: {out.shape}")
    ```

  - decode场景（变长序列）：

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    K = 3
    dim = 128
    batch = 4
    dtype = torch.bfloat16
    weight = torch.randn(K, dim, dtype=dtype).npu()
    seq_lens = [1, 1, 1, 1]
    cu_seq_len = sum(seq_lens)
    x = torch.randn(cu_seq_len, dim, dtype=dtype).npu()
    query_start_loc = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32).npu()
    num_slots = 8
    conv_states = torch.randn(num_slots, K - 1, dim, dtype=dtype).npu()
    cache_indices = torch.tensor([0, 2, 4, 6], dtype=torch.int32).npu()
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    def causal_conv1d_decode_2d(x, weight, conv_states, query_start_loc, cache_indices):
        return torch_npu.npu_fused_causal_conv1d(
            x,
            weight,
            conv_states,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            num_accepted_tokens=None,
            activation_mode="None",
            pad_slot_id=-1,
            run_mode=1,
            residual_connection=0,
        )
    compiled_func = torch.compile(causal_conv1d_decode_2d, backend=npu_backend)
    out = compiled_func(x, weight, conv_states, query_start_loc, cache_indices)
    print(f"output shape: {out.shape}")
    ```
    
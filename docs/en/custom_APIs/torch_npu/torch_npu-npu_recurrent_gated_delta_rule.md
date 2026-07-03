# torch_npu.npu_recurrent_gated_delta_rule

> [!NOTICE]  
> This API is a new feature introduced in this version. For details about the specific dependency requirements, see [API Changes](https://gitcode.com/Ascend/pytorch/blob/v2.7.1/docs/zh/release_notes/release_notes.md#%E6%8E%A5%E5%8F%A3%E5%8F%98%E6%9B%B4%E8%AF%B4%E6%98%8E). 

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>  |     √    |

## Function

- Description: Implements the computation logic of the variable-step Recurrent Gated Delta Rule (RGDR), which is one of the key operators in Transformer linear attention mechanisms. By introducing a gating mechanism and a recursive update strategy, RGDR can effectively capture long-range dependencies while maintaining linear time complexity, significantly reducing model sensitivity to sequence length.

- Formula:

  At each time step $t$, the network computes the current output $o_t$ and the new hidden state $S_t$ based on the current inputs $q_t$, $k_t$, and $v_t$, as well as the previous hidden state $S_{t-1}$.
  During this process, the gating mechanism determines how much new information is stored in the hidden state and how much existing information is forgotten.
  $$
  S_t := S_{t-1}(\alpha_t Diag(\alpha_{kt}) (I - \beta_t k_t k_t^T)) + \beta_t v_t k_t^T = \alpha_t Diag(\alpha_{kt}) S_{t-1} + \beta_t (v_t - \alpha_t Diag(\alpha_{kt}) S_{t-1}k_t)k_t^T
  $$
  $$
  o_t := \frac{S_t q_t}{\sqrt{D_k}}
  $$

  where $S_{t-1}, S_t \in R^{D_v \times D_k}$, $q_t, k_t \in R^{D_k}$, $v_t \in R^{D_v}$, $\alpha_t \in R$, $\alpha_{kt} \in R^{D_k}$, $\beta_t \in R$, $o_t \in R^{D_v}$.

## Prototype

```python
torch_npu.npu_recurrent_gated_delta_rule(query, key, value, state, *, beta=None, scale=None, actual_seq_lengths=None, ssm_state_indices=None, num_accepted_tokens=None, g=None, gk=None) -> Tensor
```

## Parameters

- **`query`** (`Tensor`): Required. $q$ in the formula. The data type can be `bfloat16`, the data format can be `ND`, and the shape is ($T$, $N_k$, $D_k$).

- **`key`** (`Tensor`): Required. $k$ in the formula. The data type can be `bfloat16`, the data format can be `ND`, and the shape is ($T$, $N_k$, $D_k$).

- **`value`** (`Tensor`): Required. $v$ in the formula. The data type can be `bfloat16`, the data format can be `ND`, and the shape is ($T$, $N_v$, $D_v$).

- **`state`** (`Tensor`): Required. Input and output parameter corresponding to the state matrix $S$ in the formula. The data type can be `bfloat16`. The data layout can be ND. The shape of this parameter is `(BlockNum, N_v, D_v, D_k)`. 

- **`beta`** (`Tensor`): Required. $β$ in the formula. The data type can be `bfloat16`, the data format can be `ND`, and the shape is ($T$, $N_v$).

- **`scale`** (`float`): Required. Scaling factor of `query`, $1/\sqrt{D_k}$ in the formula. The data type can be `float32`.

- **`actual_seq_lengths`** (`Tensor`): Required. Input sequence length of each batch. The data type can be `int32`. The data layout can be ND. The shape of this parameter is `(B,)`. Each value $L_i$ must remain within `[1, 8]`, where $L_i$ represents the value of the i-th element in `actual_seq_lengths`.

- **`ssm_state_indices`** (`Tensor`): Required. Mapping indices from the input sequence to the state matrix, where `state[ssm_state_indices[i]]` represents the state matrix corresponding to the i-th token. Each value of `ssm_state_indices[i]` must be greater than or equal to 0 and less than $BlockNum$. This parameter must be 1D with shape `(T,)`. The data type can be `int32`. The data layout can be ND.

- **`num_accepted_tokens`** (`Tensor`): Optional. Number of tokens accepted by the i-th batch during speculative decoding, where each value must be greater than or equal to 1 and less than or equal to $L_i$. The default value is `None`, indicating that each batch accepts one token. The data type can be `int32`. The data layout can be ND. The shape is `(B,)`.

- **`g`** (`Tensor`): Optional. Decay coefficient, $α=e^g$ in the formula. The default value is `None`, indicating all zeros. The data type can be `float32`. The data layout can be ND. The shape is ($T$, $N_v$).

- **`gk`** (`Tensor`): Optional. Decay coefficient, $\alpha = e^{gk}$ in the formula. The default value is `None`, indicating all zeros. The data type can be `float32`. The data layout can be ND. The shape is ($T$, $N_v$, $D_k$).

## Return Value

`Tensor`

Attention computation result, $o$ in the formula. The output data type can be `bfloat16`. The data layout is ND. The shape is ($T$, $N_v$, $D_v$).

## Constraints

- Variables used in parameter tensor shapes:
    - $T=\sum_i^B L_i$ indicates the accumulated sequence length.
    - $B$ indicates the batch size.
    - $L_i$ indicates the length of the i-th sequence, provided through `actual_seq_lengths`, where the value range is $1 \le L_i \le 8$.
    - $N_k$ indicates the head count of `key`. The value range is $1 \le N_k \le 256$.
    - $N_v$ indicates the number of value heads. The value range is $1 \le N_v \le 256$.
    - $D_k$ indicates the dimension of the key vector. The value range is $1 \le D_k \le 512$.
    - $D_v$ indicates the dimension of the value vector. The value range is $1 \le D_v \le 512$.
    - $BlockNum$ indicates the number of blocks in the state matrix. The value must be greater than or equal to $T$.
- This API can be used only in inference scenarios.
- This API supports only single-operator and static graph modes.

## Example

- Single-operator call

    ```python
    import torch
    import torch_npu

    # Construct the input.
    bs, mtp, nk, nv, dk, dv = (2, 3, 4, 8, 128, 128)
    actual_seq_lengths = (torch.ones(bs) * mtp).npu().to(torch.int32)
    T = int(torch.sum(actual_seq_lengths))

    state = torch.rand((T, nv, dv, dk), dtype=torch.bfloat16).npu()
    query = torch.rand((T, nk, dk), dtype=torch.bfloat16).npu()
    key = torch.rand((T, nk, dk), dtype=torch.bfloat16).npu()
    value = torch.rand((T, nv, dv), dtype=torch.bfloat16).npu()
    g = torch.rand((T, nv), dtype=torch.float32).npu() * (-1.0)
    beta = torch.rand((T, nv), dtype=torch.bfloat16).npu()

    ssm_state_indices = (torch.arange(T).npu()).to(torch.int32)
    query = torch.nn.functional.normalize(query, p=2, dim=-1)
    key = torch.nn.functional.normalize(key, p=2, dim=-1)
    scale = dk ** -0.5
    num_accepted_tokens = (torch.randint(1, mtp + 1, (bs,)).npu()).to(torch.int32)

    # Call the operator.
    o = torch_npu.npu_recurrent_gated_delta_rule(
        query, key, value, state, beta=beta, scale=scale,
        actual_seq_lengths=actual_seq_lengths, ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens, g=g, gk=None)
    print(o)
    ```

- Static graph mode call

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig
    import logging
    from torchair.core.utils import logger

    logger.setLevel(logging.DEBUG)
    import os
    import numpy as np

    # ENABLE_ACLNN specifies whether to use ACLNN. Valid values: true (uses ACLNN execution) or false (uses online compilation).
    os.environ["ENABLE_ACLNN"] = "false"
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
 
        def forward(self, query, key, value, state, beta, scale, actual_seq_lengths, ssm_state_indices, num_accepted_tokens, g):
            return torch_npu.npu_recurrent_gated_delta_rule(
                query, key, value, state, beta=beta, scale=scale,
                actual_seq_lengths=actual_seq_lengths, ssm_state_indices=ssm_state_indices,
                num_accepted_tokens=num_accepted_tokens, g=g, gk=None)


    # Construct the input.
    bs, mtp, nk, nv, dk, dv = (2, 3, 4, 8, 128, 128)
    actual_seq_lengths = (torch.ones(bs) * mtp).npu().to(torch.int32)
    T = int(torch.sum(actual_seq_lengths))

    state = torch.rand((T, nv, dv, dk), dtype=torch.bfloat16).npu()
    query = torch.rand((T, nk, dk), dtype=torch.bfloat16).npu()
    key = torch.rand((T, nk, dk), dtype=torch.bfloat16).npu()
    value = torch.rand((T, nv, dv), dtype=torch.bfloat16).npu()
    g = torch.rand((T, nv), dtype=torch.float32).npu() * (-1.0)
    beta = torch.rand((T, nv), dtype=torch.bfloat16).npu()

    ssm_state_indices = (torch.arange(T).npu()).to(torch.int32)
    query = torch.nn.functional.normalize(query, p=2, dim=-1)
    key = torch.nn.functional.normalize(key, p=2, dim=-1)
    scale = dk ** -0.5
    num_accepted_tokens = (torch.randint(1, mtp + 1, (bs,)).npu()).to(torch.int32)

    #Call
    model = MyModel()
    model = model.npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    o = model(query, key, value, state, beta, scale, actual_seq_lengths, ssm_state_indices, num_accepted_tokens, g)
    print(o)
    ```

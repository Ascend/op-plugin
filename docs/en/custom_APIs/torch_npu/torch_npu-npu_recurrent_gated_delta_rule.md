# torch_npu.npu_recurrent_gated_delta_rule

> [!NOTICE]  
> This API has changed in this version. For details about the changes, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->

## Function

- Function: implements the computation logic of the variable-step Recurrent Gated Delta Rule (RGDR). It is one of the key operators of the Transformer linear attention mechanism. By introducing a gating mechanism and a recurrent update strategy, RGDR can effectively capture long-range dependencies while maintaining linear time complexity, significantly reducing the model sensitivity to sequence length.

- Formulas:

  At each time step $t$, the network computes the current output $o_t$ and the new hidden state $S_t$ based on the current inputs $q_t$, $k_t$, and $v_t$ and the previous hidden state $S_{t-1}$.
  In this process, the gating unit determines how much new information is stored in the hidden state and how much old information is forgotten.
  $$
  S_t := S_{t-1}(\alpha_t Diag(\alpha_{kt}) (I - \beta_t k_t k_t^T)) + \beta_t v_t k_t^T = \alpha_t Diag(\alpha_{kt}) S_{t-1} + \beta_t (v_t - \alpha_t Diag(\alpha_{kt}) S_{t-1}k_t)k_t^T
  $$
  $$
  o_t := \frac{S_t q_t}{\sqrt{D_k}}
  $$

  In the formulas, $S_{t-1},S_t \in R^{D_v \times D_k}$, $q_t, k_t \in R^{D_k}$, $v_t \in R^{D_v}$, $\alpha_t \in R$, $\alpha_{kt} \in R^{D_k}$, $\beta_t \in R$, and $o_t \in R^{D_v}$.

## Prototype

```python
torch_npu.npu_recurrent_gated_delta_rule(query, key, value, state, *, beta=None, scale=None, actual_seq_lengths=None, ssm_state_indices=None, num_accepted_tokens=None, g=None, gk=None) -> Tensor
```

## Parameters

- **`query`** (`Tensor`): Required. $q$ in the formula. The data type can be `torch.bfloat16`, the data format can be `ND`, and the shape is ($T$, $N_k$, $D_k$). Empty tensors are not supported.

- **`key`** (`Tensor`): Required. $k$ in the formula. The data type can be `torch.bfloat16`, the data format can be `ND`, and the shape is ($T$, $N_k$, $D_k$). Empty tensors are not supported.

- **`value`** (`Tensor`): Required. $v$ in the formula. The data type can be `torch.bfloat16`, the data format can be `ND`, and the shape is ($T$, $N_v$, $D_v$). Empty tensors are not supported.

- **`state`** (`Tensor`): Required. Input and output parameter corresponding to the state matrix $S$ in the formula. The data type can be `torch.bfloat16` or `torch.float32`. The data layout can be ND. The shape of this parameter is `(BlockNum, N_v, D_v, D_k)`. Empty tensors are not supported.

- \*: Variables before it are position-dependent and must be passed in order; variables after it are optional keyword arguments and can be passed in any order using key-value pairs. If not specified, their default values are used.

- **`beta`** (`Tensor`): Optional. $β$ in the formula. The data type can be `torch.bfloat16`, the data format can be `ND`, and the shape is ($T$, $N_v$). Empty tensors are not supported.

- **`scale`** (`float`): Optional. Scaling factor of `query`, $1/\sqrt{D_k}$ in the formula. The data type can be `float32`.

- **`actual_seq_lengths`** (`Tensor`): Optional. Input sequence length of each batch. The data type can be `torch.int32`. The data layout can be ND. The shape of this parameter is `(B,)`. Each value $L_i$ must remain within `[1, 8]`, where $L_i$ represents the value of the i-th element in `actual_seq_lengths`. Empty tensors are not supported.

- **`ssm_state_indices`** (`Tensor`): Optional. Mapping indices from the input sequence to the state matrix, where `state[ssm_state_indices[i]]` represents the state matrix corresponding to the i-th token. Each value of `ssm_state_indices[i]` must be greater than or equal to 0 and less than $BlockNum$. This parameter must be 1D with shape `(T,)`. The data type can be `torch.int32`. The data layout can be ND. Empty tensors are not supported.

- **`num_accepted_tokens`** (`Tensor`): Optional. Number of tokens accepted by the i-th batch during speculative decoding, where each value must be greater than or equal to 1 and less than or equal to $L_i$. The default value is `None`, indicating that each batch accepts one token. The data type can be `torch.int32`. The data layout can be ND. The shape is `(B,)`.

- **`g`** (`Tensor`): Optional. Decay coefficient, $α=e^g$ in the formula. The default value is `None`, indicating all zeros. The data type can be `torch.float32`. The data layout can be ND. The shape is ($T$, $N_v$). Empty tensors are not supported.

- **`gk`** (`Tensor`): Optional. Decay coefficient, $\alpha = e^{gk}$ in the formula. The default value is `None`, indicating all zeros. The data type can be `torch.float32`. The data layout can be ND. The shape is ($T$, $N_v$, $D_k$). Empty tensors are not supported.

## Return Values

**`out`** (`Tensor`): Computation result, $o$ in the formulas, that is, the attention computation result. The output data type is `torch.bfloat16`, the data layout is ND, and the shape is ($T$, $N_v$, $D_v$).

## Constraints

- Variables used in the shapes of the parameters are as follows:
    - $T=\sum_i^B L_i$ indicates the cumulative sequence length.
    - $B$ indicates the batch size.
    - $L_i$ indicates the length of the i-th sequence, which is passed through actual_seq_lengths. Its value range is $1 \le L_i \le 8$.
    - $N_k$ indicates the number of key heads. Its value range is $1 \le N_k \le 256$.
    - $N_v$ indicates the number of value heads. Its value range is $1 \le N_v \le 256$, and $N_v \bmod N_k = 0$.
    - $D_k$ indicates the dimension of the key vector. Its value range is $1 \le D_k \le 512$.
    - $D_v$ indicates the dimension of the value vector. Its value range is $1 \le D_v \le 512$.
    - $BlockNum$ is the number of state matrix blocks. Its value range is $T \le BlockNum$.
- This API can be used only in inference scenarios. In the current TND scenario, beta, scale, actual_seq_lengths, and ssm_state_indices are mandatory.
- This API supports only single-operator mode and static graph mode calls.
- For state, only non-contiguous tensors on the 0th and 1st axes are supported.
- The following constraints cannot be verified by the operator because it cannot obtain the specific values in the tensor. Therefore, the user must ensure them:
    - $ssm\_state\_indices[i] \lt BlockNum$
    - $1 \le actual\_seq\_lengths[i] \le 8$, and the cumulative sum of actual_seq_lengths[i] is equal to $T$
    - $1 \le num\_accepted\_tokens[i] \le actual\_seq\_lengths[i]$
    - $-1 \le query[i][j][k] \le 1$
    - $-1 \le key[i][j][k] \le 1$
    - $g[i][j] \lt 0$
    - $gk[i][j][k] \lt 0$
    - $0 \lt beta[i][j] \lt 1$

## Example

- Single-operator call

    ```python
    import torch
    import torch_npu

    # Construct the inputs
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

    # Call the operator
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

    # Whether "ENABLE_ACLNN" uses aclnn. true: the callback uses aclnn. false: online compilation.
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


    # Construct the inputs
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

    # Call
    model = MyModel()
    model = model.npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    o = model(query, key, value, state, beta, scale, actual_seq_lengths, ssm_state_indices, num_accepted_tokens, g)
    print(o)
    ```

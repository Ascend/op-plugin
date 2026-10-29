# torch_npu.npu_chunk_gated_delta_rule

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

- Function: implements the computation logic of the chunk version of the Gated Delta Rule (GDR). It is one of the key operators of the Transformer linear attention mechanism.

- Formulas:

  The Gated Delta Rule (GDR) is an operator applied to recurrent neural networks and is also used in a linear attention mechanism. At each time step $t$, GDR computes the current attention output $o_t$ and the new hidden state $S_t$ based on the current inputs $q_t$, $k_t$, and $v_t$, the previous hidden state $S_{t-1}$, the decay coefficient $\alpha_t$, and the update strength $\beta_t$. The formula is as follows:
  $$
  S_t := S_{t-1}(\alpha_t(I - \beta_t k_t k_t^T)) + \beta_t v_t k_t^T = \alpha_t S_{t-1} + \beta_t (v_t - \alpha_t S_{t-1}k_t)k_t^T
  $$
  $$
  o_t := S_t (q_t \cdot scale)
  $$

  In the formulas, $S_{t-1},S_t \in R^{D_v \times D_k}$, $q_t, k_t \in R^{D_k}$, $v_t \in R^{D_v}$, $\alpha_t \in R$, $\beta_t \in R$, and $o \in R^{D_v}$.
  
  Chunked Gated Delta Rule (CGDR) is the chunk version of GDR ([reference paper](https://arxiv.org/abs/2412.06464)). It splits the input sequence into chunks to achieve a degree of parallelism. In long-context scenarios, its computation efficiency is higher than that of the Recurrent Gated Delta Rule, so it is suitable for the prefill phase. Given an input sequence of length L, the CGDR operator can compute the output of each step $o_t, t \in \{1, .., L\}$ and the final state matrix $S_L$.

## Prototype

```python
torch_npu.npu_chunk_gated_delta_rule(query, key, value, *, beta=None, initial_state=None, actual_seq_lengths=None, scale=None, g=None) -> tuple(Tensor, Tensor)
```

## Parameters

> [!NOTE]  
> $T=\sum_i^B L_i$ indicates the cumulative sequence length, and $B$ indicates the batch size. $L_i$ indicates the length of the i-th sequence.<br>
> $N_k$ indicates the number of key heads, and $N_v$ indicates the number of value heads.<br>
> $D_k$ indicates the hidden size of key, and $D_v$ indicates the hidden size of value.

- **`query`** (`Tensor`): Required. $q$ in the formula. The data type can be `torch.bfloat16`. The data layout can be ND. The shape is ($T$, $N_k$, $D_k$). Empty tensors are not supported.

- **`key`** (`Tensor`): Required. $k$ in the formula. The data type can be `torch.bfloat16`. The data layout can be ND. The shape is ($T$, $N_k$, $D_k$). Empty tensors are not supported.

- **`value`** (`Tensor`): Required. $v$ in the formula. The data type can be `torch.bfloat16`. The data layout can be ND. The shape is ($T$, $N_v$, $D_v$). Empty tensors are not supported.

- **`beta`** (`Tensor`): Optional. $β$ in the formula. The data type can be `torch.bfloat16`. The data layout can be ND. The shape is ($T$, $N_v$). Empty tensors are not supported.

- **`initial_state`** (`Tensor`): Optional. State matrix $S_0$ in the formula. The data type can be `torch.bfloat16` or `torch.float32`. The data layout can be ND. The shape is ($B$, $N_v$, $D_v$, $D_k$). Empty tensors are not supported.

- **`actual_seq_lengths`** (`Tensor`): Optional. Input sequence length of each batch. The data type can be `torch.int32`. The data layout can be ND. The shape is `(B,)`.

- **`scale`** (`float`): Optional. Scaling factor of the query, $scale$ in the formula. The data type can be `float32`. The default value `None` indicates 1.0. In practice, it is typically set to $1/\sqrt{D_k}$.

- **`g`** (`Tensor`): Optional. Decay coefficient, $α=e^g$ in the formula. The default value is `None`, indicating all zeros. The data type can be `torch.float32`. The data layout can be ND. The shape is ($T$, $N_v$).

## Return Values

- **`out`** (`Tensor`): attention computation result, $o_t$ in the formula. The data type can be `torch.bfloat16`. The data layout is ND. The shape is ($T$, $N_v$, $D_v$).

- **`final_state`** (`Tensor`): Final state matrix $S_L$ in the formula. The data type can be `torch.bfloat16` or `torch.float32`. The data layout is ND. The shape is ($B$, $N_v$, $D_v$, $D_k$).

## Constraints

- This API can be used only in inference scenarios. In the current TND scenario, beta, initial_state, and actual_seq_lengths are mandatory.
<!-- npu="950" id4 -->
- The `torch.float32` data type of initial_state and final_state is supported only on Ascend 950PR&950DT products.
<!-- end id4 -->
- Dimension constraints:
  - $0 \lt Nv \le 64, 0 \lt Nk \le 64$, and $Nv \bmod Nk = 0$
  - $0 \lt Dv \le 128$, $0 \lt Dk \le 128$
  - $B \gt 0$, $T \gt 0$
- Due to algorithm characteristics, the user must ensure the following numerical constraints; otherwise, the computation result may overflow:
  - $-1 \le query[i][j][k] \le 1$, and query must be normalized
  - $-1 \le key[i][j][k] \le 1$, and key must be normalized
  - $-1 \le g[i][j] \le 0$
  - $0 < beta[i][j] < 1$

## Example

- Single-operator call

    ```python
    import torch
    import torch_npu

    # Construct the inputs
    B, seqlen, nk, nv, dk, dv = (2, 100, 4, 8, 128, 128)
    actual_seq_lengths = (torch.ones(B) * seqlen).npu().to(torch.int32)
    T = int(torch.sum(actual_seq_lengths))

    state = torch.rand((B, nv, dv, dk), dtype=torch.bfloat16).npu()
    query = torch.rand((T, nk, dk), dtype=torch.bfloat16).npu()
    key = torch.rand((T, nk, dk), dtype=torch.bfloat16).npu()
    value = torch.rand((T, nv, dv), dtype=torch.bfloat16).npu()
    g = torch.rand((T, nv), dtype=torch.float32).npu() * (-1.0)
    beta = torch.rand((T, nv), dtype=torch.bfloat16).npu()

    query = torch.nn.functional.normalize(query, p=2, dim=-1)
    key = torch.nn.functional.normalize(key, p=2, dim=-1)
    scale = dk ** -0.5

    # Call the operator
    o, final_state = torch_npu.npu_chunk_gated_delta_rule(
        query, key, value, beta=beta, initial_state=state, actual_seq_lengths=actual_seq_lengths, scale=scale, g=g)
    print(o.shape, final_state.shape)
    ```

- Graph mode call

    ```python
    import torch
    import torch_npu
    import torchair
    import logging
    import os
    import warnings
    import torch.nn.functional as F

    from torchair.core.utils import logger

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    logger.setLevel(logging.DEBUG)

    os.environ["ENABLE_ACLNN"] = "false"

    # Configure the graph mode config
    config = torchair.CompilerConfig()

    # Configure the graph execution mode. The aclgraph mode is reduce-overhead.
    config.mode = "reduce-overhead"
    npu_backend = torchair.get_npu_backend(compiler_config=config)

    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self,
        query,
        key,
        value,
        initial_state,
        beta,
        actual_seq_lengths,
        scale,
        gamma):
            chunk_gated_delta_rule = torch_npu.npu_chunk_gated_delta_rule(
                query, key, value,
                beta = beta,
                initial_state = initial_state,
                actual_seq_lengths = actual_seq_lengths,
                scale = scale,
                g = gamma)

            return chunk_gated_delta_rule

    if __name__ == "__main__":
        """
        bs: number of statements in a batch;
        seqlen: number of tokens in a statement, that is, the length;
        nk, nv: number of attention heads corresponding to QK and V;
        dk, dv: dimension of the word and position space embedding corresponding to QK and V.
        """
        bs, seqlen = 2, 100
        nk, nv = 4, 4
        dk, dv = 128, 128
        actual_seq_lengths = (torch.ones(bs) * seqlen).npu().to(torch.int32)
        T = int(torch.sum(actual_seq_lengths))

        print(f"Input Info:\n{bs=}, {seqlen=}, {nk=}, {nv=}, {dk=}, {dv=}, {actual_seq_lengths=}, {T=}")

        # Initialize the input tensor
        query = torch.rand((T, nk, dk), dtype=torch.bfloat16, device='npu')
        key = torch.rand((T, nk, dk), dtype=torch.bfloat16, device='npu')
        value = torch.rand((T, nv, dv), dtype=torch.bfloat16, device='npu')
        initial_state = torch.rand((bs, nv, dk, dv), dtype=torch.bfloat16, device='npu')
        beta = torch.rand((T, nv), dtype=torch.bfloat16, device='npu')
        gamma = torch.rand((T, nv), dtype=torch.float32).npu() * (-1.0)
        cu_seqlens = F.pad(actual_seq_lengths, (1, 0)).cumsum(dim=0).npu().to(torch.int32)
        cu_seqlens = cu_seqlens[1:]

        # query = torch.nn.functional.normalize(query, p=2, dim=-1)
        # key = torch.nn.functional.normalize(key, p=2, dim=-1)
        scale = dk ** -0.5

        print(f"\nInput Shape:\nQ:{query.shape}, K:{key.shape}, V:{value.shape},"
            f"\nstate:{initial_state.shape}, beta:{beta.shape}, gamma:{gamma.shape},"
            f"\ninitial_state:{initial_state.shape}, beta:{beta.shape}, gamma:{gamma.shape},"
            f"\ncu_seqlens:{cu_seqlens.shape}, scale:{scale}")

        print("\nTest Torch Adapter Graph...")

        print("\nCreate model...")
        model = MyModel()
        model = model.npu()

        print("\nModel compile...")
        model = torch.compile(model, backend=npu_backend, dynamic=False)

        print("\nInference...\n")
        o_chunk, state_chunk = model(
            query=query,
            key=key,
            value=value,
            initial_state=initial_state,
            beta=beta,
            actual_seq_lengths=cu_seqlens,
            scale=scale,
            gamma=gamma
        )
    ```

# torch_npu.npu_chunk_gated_delta_rule

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     √    |

## Function

- Description: Computes the chunked version of Gated Delta Rule (GDR), which is one of the key operators of the Transformer linear attention mechanism.

- Formula:

  GDR is an operator designed for recurrent neural networks (RNNs) and serves as a component in linear attention mechanisms. At each time step $t$, GDR computes the current attention output $o_t$ and the new hidden state $S_t$ based on the current inputs $q_t$, $k_t$, $v_t$, the previous hidden state $S_{t-1}$, the decay coefficient $\alpha_t$, and the update strength $\beta_t$. The formula is as follows:
  $$
  S_t := S_{t-1}(\alpha_t(I - \beta_t k_t k_t^T)) + \beta_t v_t k_t^T = \alpha_t S_{t-1} + \beta_t (v_t - \alpha_t S_{t-1}k_t)k_t^T
  $$
  $$
  o_t := S_t (q_t \cdot scale)
  $$

  where, $S_{t-1},S_t \in R^{D_v \times D_k}$, $q_t, k_t \in R^{D_k}$, $v_t \in R^{D_v}$, $\alpha_t \in R$, $\beta_t \in R$, $o \in R^{D_v}$.
  
  Chunked Gated Delta Rule (CGDR) is a chunked version of GDR ([reference paper](https://arxiv.org/abs/2412.06464)). It splits the input sequence into chunks to achieve parallel execution. This operator provides higher computation efficiency than Recurrent Gated Delta Rule in long-context scenarios. It is applicable to the prefill phase. Given an input sequence of length $L$, the CGDR operator can compute the output $o_t$ for each step $t \in \{1, ..., L\}$ and the final state matrix $S_L$.

## Prototype

```python
torch_npu.npu_chunk_gated_delta_rule(query, key, value, *, beta=None, initial_state=None, actual_seq_lengths=None, scale=None, g=None) -> tuple(Tensor, Tensor)
```

## Parameters

$T=\sum_i^B L_i$ indicates the accumulated sequence length, and $B$ indicates the batch size. $L_i$ indicates the length of the ith sequence.
$N_k$ indicates the number of key heads, and $N_v$ indicates the number of value heads.
$D_k$ indicates the hidden size of the key, and $D_v$ indicates the hidden size of the value.

- **`query`** (`Tensor`): Required. $q$ in the formula. The data type can be `bfloat16`. The data layout can be ND. The shape is ($T$, $N_k$, $D_k$).

- **`key`** (`Tensor`): Required. $k$ in the formula. The data type can be `bfloat16`. The data layout can be ND. The shape is ($T$, $N_k$, $D_k$).

- **`value`** (`Tensor`): Required. $v$ in the formula. The data type can be `bfloat16`. The data layout can be ND. The shape is ($T$, $N_v$, $D_v$).

- **`beta`** (`Tensor`): Required. $β$ in the formula. The data type can be `bfloat16`. The data layout can be ND. The shape is ($T$, $N_v$).

- **`initial_state`** (`Tensor`): Required. Initial state matrix $S_0$ in the formula. The data type can be `bfloat16`. The data layout can be ND. The shape is ($B$, $N_v$, $D_v$, $D_k$).

- **`actual_seq_lengths`** (`Tensor`): Required. Input sequence length of each batch. The data type can be `int32`. The data layout can be ND. The shape is `(B,)`.

- **`scale`** (`float`): Required. Scaling factor of the query, $scale$ in the formula. The data type can be `float32`. The default value `None` indicates 1.0. In practice, it is typically set to $1/\sqrt{D_k}$.

- **`g`** (`Tensor`): Required. Decay coefficient, $α=e^g$ in the formula. The default value is `None`, indicating all zeros. The data type can be `float32`. The data layout can be ND. The shape is ($T$, $N_v$).

## Return Value

- **`out`** (`Tensor`): attention computation result, $o_t$ in the formula. The data type can be `bfloat16`. The data layout is ND. The shape is ($T$, $N_v$, $D_v$).

- **`final_state`** (`Tensor`): final state matrix, $S_L$ in the formula. The data type can be `bfloat16`. The data layout is ND. The shape is ($B$, $N_v$, $D_v$, $D_k$).

## Constraints

- This API can be used only in inference scenarios.
- The input shape must meet the following requirements:
  - $Nv \le 64$, $Nk \le 64$, and $Nv$ must be a multiple of $Nk$.
  - $Dv = Dk = 128$.

## Examples

- Single-operator call

    ```python
    import torch
    import torch_npu

    # Construct the input.
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

    # Call the operator.
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

    # Configure the graph mode.
    config = torchair.CompilerConfig()

    # Set the graph execution mode: aclgraph mode is reduce-overhead, and GE mode is max-autotune.
    config.mode = "max-autotune"
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
        bs: number of statements in a batch.
        seqlen: number of tokens in a statement, indicating the sequence length.
        nk, nv: number of attention heads corresponding to QK and V.
        dk, dv: word and position space embedding dimensions corresponding to QK and V.
        """
        bs, seqlen = 2, 100
        nk, nv = 4, 4
        dk, dv = 128, 128
        actual_seq_lengths = (torch.ones(bs) * seqlen).npu().to(torch.int32)
        T = int(torch.sum(actual_seq_lengths))

        print(f"Input Info:\n{bs=}, {seqlen=}, {nk=}, {nv=}, {dk=}, {dv=}, {actual_seq_lengths=}, {T=}")

        # Initialize input tensors.
        query = torch.rand((T, nk, dk), dtype=torch.bfloat16, device='npu')
        key = torch.rand((T, nk, dk), dtype=torch.bfloat16, device='npu')
        value = torch.rand((T, nv, dv), dtype=torch.bfloat16, device='npu')
        initial_state = torch.rand((bs, nv, dk, dv), dtype=torch.bfloat16, device='npu')
        beta = torch.rand((T, nv), dtype=torch.bfloat16, device='npu')
        gamma = torch.rand((T, nv), dtype=torch.float32, device='npu')
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

# torch_npu.npu_dense_lightning_indexer_grad_kl_loss

> [!NOTICE]  
> This API is a new feature introduced in this version. For details about the specific dependency requirements, see [API Changes](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.0.0/docs/en/release_notes/release_notes.md#api-changes).

## Supported Products

| Product                          | Supported|
|------------------------------| :------: |
| <term>Atlas A3 training products</term>| √  |
| <term>Atlas A2 training products</term>| √  |

## Function

- Description: Implements the backward gradient computation during the warmup phase training of the Lightning Indexer component, while integrating the loss computation.
     - Compared with the `npu_sparse_lightning_indexer_grad_kl_loss` API, this API applies to dense scenarios, and its inputs `key` and `key_index` do not require sparsification.
     - This API is used together with `npu_dense_lightning_indexer_softmax_lse`. `softmax_max_index` and `softmax_sum_index` computed by `npu_dense_lightning_indexer_softmax_lse` serve as inputs to this API. Using these two APIs in combination optimizes device memory utilization.

- Formulas:

    1. Formula for top-k value computation:
        $$
        I_{t,:}=W_{t,:}@ReLU(\tilde{q}_{t,:}@\tilde{K}_{:t,:}^\top)
        $$
    
        - $W_{t,:}$ is the weights corresponding to the $t$-th token in the $W$ matrix.
        - $\tilde{q}_{t,:}$ is the result after collapsing the $G$ query heads corresponding to the $t$-th token in the $\tilde{Q}$ matrix.
        - $\tilde{K}_{:t,:}$ is the rows up to index $t$ in the $\tilde{K}$ matrix.
  
    2. Formula for forward Softmax computation:

        $$
        p_{t,:} = \text{Softmax}(q_{t,:} @ K_{:t,:}^\top/\sqrt{d})
        $$
        
        - $p_{t,:}$ is the Softmax result corresponding to the $t$-th token.
        - $q_{t,:}$ is the result of merging the $G$ query heads along the axis corresponding to the $t$-th token in the $Q$ matrix.
        - $K_{:t,:}$ is the rows up to index $t$ in the $K$ matrix.

    3. Lightning Indexer is trained separately, and the corresponding loss function is as follows:
        $$
        Loss{=}\sum_tD_{KL}(p_{t,:}||Softmax(I_{t,:}))
        $$

        $p_{t,:}$ is the target distribution, which is obtained by summing the main attention scores across all heads and then performing L1 normalization along the context dimension. $D_{KL}$ is the KL divergence, expressed as:

        $$
        D_{KL}(a||b){=}\sum_ia_i\mathrm{log}{\left(\frac{a_i}{b_i}\right)}
        $$

    4. The gradient of the loss is derived as:
        $$
        dI\mathop{{}}\nolimits_{{t,:}}=Softmax \left( I\mathop{{}}\nolimits_{{t,:}} \left) -p\mathop{{}}\nolimits_{{t,:}}\right. \right.
        $$

        Using the chain rule, the gradients for the `weights`, `query`, and `key` matrices are computed as follows:
        $$
        dW\mathop{{}}\nolimits_{{t,:}}=dI\mathop{{}}\nolimits_{{t,:}}\text{@} \left( ReLU \left( S\mathop{{}}\nolimits_{{t,:}} \left) \left) \mathop{{}}\nolimits^{\top}\right. \right. \right. \right.
        $$
    
        $$
        d\mathop{{\tilde{q}}}\nolimits_{{t,:}}=dS\mathop{{}}\nolimits_{{t,:}}@\tilde{K}\mathop{{}}\nolimits_{{:t,:}}
        $$
    
        $$
        d\tilde{K}\mathop{{}}\nolimits_{{:t,:}}=\left(dS\mathop{{}}\nolimits_{{t,:}} \left) \mathop{{}}\nolimits^{\top}@\tilde{q}\mathop{{}}\nolimits_{{:t, :}}\right. \right.
        $$

        $S$ is the result of matrix multiplication between the $\tilde{Q}$ and $\tilde{K}$ matrices, and $S_{t,:}$ is the $t$-th row in the $S$ matrix. $dW_{t,:}$ is the $t$-th row in the $dW$ matrix, $d\mathop{{\tilde{q}}}\nolimits_{{t,:}}$ is the $t$-th row in the $d\tilde{Q}$ matrix, and $d\tilde{K}\mathop{{}}\nolimits_{{:t,:}}$ is the accumulated value of the rows up to index $t$ in the $d\tilde{K}$ matrix.

## Prototype

```python
npu_dense_lightning_indexer_grad_kl_loss(query, key, query_index, key_index, weights, softmax_max, softmax_sum, softmax_max_index, softmax_sum_index, scale_value, *, query_rope=None, key_rope=None, actual_seq_qlen=None, actual_seq_klen=None, layout='BSND', sparse_mode=3, pre_tokens=2^63-1, next_tokens=2^63-1) -> (Tensor, Tensor, Tensor, Tensor)
```

## Parameters

**`query`** (`Tensor`): Required. Query in the attention, $Q$ in the formula. The data layout can be ND. The data type can be `bfloat16` or `float16`. The shape can be `(B, S1, N1, D)` or `(T1, N1, D)`.

**`key`** (`Tensor`): Required. Key in the attention, $K$ in the formula. The data layout can be ND. The data type can be `bfloat16` or `float16`. The shape can be `(B, S2, N2, D)` or `(T2, N2, D)`.

**`query_index`** (`Tensor`): Required. Forward input query of the Lightning Indexer, $\tilde{Q}$ in the formula. The data layout can be ND. The data type can be `bfloat16` or `float16`. The shape can be `(B, S1, N1index, D)` or `(T1, N1index, D)`.

**`key_index`** (`Tensor`): Required. Forward input key of the Lightning Indexer, $\tilde{K}$ in the formula. The data layout can be ND. The data type can be `bfloat16` or `float16`. The shape can be `(B, S2, N2index, D)` or `(T2, N2index, D)`.

**`weights`** (`Tensor`): Required. Weight coefficients of the Lightning Indexer, $W$ in the formula. The data layout can be ND. The data type can be `bfloat16`, `float16`, or `float32`. The shape can be `(B, S1, N1index)` or `(T1, N1index)`.

**`softmax_max`** (`Tensor`): Required. Maximum value in the attention Softmax results. The data layout can be ND. The data type can be `float32`. The shape can be `(B, N2, S1, G)` or `(N2, T1, G)`. $G$ is equal to $N1/N2$.

**`softmax_sum`** (`Tensor`): Required. Sum of the attention Softmax results. The data layout can be ND. The data type can be `float32`. The shape can be `(B, N2, S1, G)` or `(N2, T1, G)`. $G$ is equal to $N1/N2$.

**`softmax_max_index`** (`Tensor`): Required. Maximum value in the index attention Softmax results. The data layout can be ND. The data type can be `float32`. The shape can be `(B, N2index, S1)` or `(N2index, T1)`.

**`softmax_sum_index`** (`Tensor`): Required. Sum of the index attention Softmax results. The data layout can be ND. The data type can be `float32`. The shape can be `(B, N2index, S1)` or `(N2index, T1)`.

**`scale_value`** (`float`): Required. Scaling coefficient. The data type can be `float32`.

**`query_rope`** (`Tensor`): Optional. RoPE information of the query in the MLA structure. The data layout can be ND. The data type can be `bfloat16` or `float16`. The shape can be `(B, S1, N1, Dr)` or `(T1, N1, Dr)`.

**`key_rope`** (`Tensor`): Optional. RoPE information of the key in the MLA structure. The data layout can be ND. The data type can be `bfloat16` or `float16`. The shape can be `(B, S2, N2, Dr)` or `(T2, N2, Dr)`.

**`actual_seq_qlen`** (`list[int]`): Optional. Accumulated sum of sequence lengths for each $S$ in the query tensor. This parameter is required in the `TND` scenario. The data type can be `int64`. The data layout can be ND. The default value is `None`.

**`actual_seq_klen`** (`list[int]`): Optional. Accumulated sum of sequence lengths for each $S$ in the key tensor. This parameter is required in the TND scenario. The data type can be `int64`. The data layout can be ND. The default value is `None`.

**`layout`** (`str`): Optional. Data layout format of the input `query`. Valid values are `BSND` or `TND`. The default value is `BSND`.

**`sparse_mode`** (`int`): Optional. Sparse mode. The data type can be `int32`. The default value is `3`. Currently, only mode `3` is supported.

**`pre_tokens`** (`int`): Optional. Number of preceding tokens to associate in attention computation for sparse computation. The data type can be `int64`. The default value is `2^{63}-1`.

**`next_tokens`** (`int`): Optional. Number of subsequent tokens to associate in attention computation for sparse computation. The data type can be `int64`. The default value is `2^{63}-1`.

## Return Values

- **`d_query_index`** (`Tensor`): Gradient of `query_index`, $d\tilde{Q}$ in the formula. The data type can be `bfloat16` or `float16`.
- **`d_key_index`** (`Tensor`): Gradient of `key_index`, $d\tilde{K}$ in the formula. The data type can be `bfloat16` or `float16`.
- **`d_weights`** (`Tensor`): Gradient of `weights`, $dW$ in the formula. The data type can be `bfloat16`, `float16`, or `float32`.
- **`loss`** (`Tensor`): Loss tensor representing the difference between the forward network output and the golden reference values, $Loss$ in the formula. The data type can be `float32`.

## Constraints

- The data types of `query`, `key`, `query_index`, and `key_index` must be identical.
- When the data type of `weights` is not `float32`, the data types of `query`, `key`, `query_index`, `key_index`, and `weights` must be identical.
- Shape variable constraints:

| Item   | Value      | Description        |
|-----------|------------|-----------------|
| B         | 1 to 256     | -               |
| S1, S2   | 1 to 128K    | `S1` and `S2` can have different lengths.|
| N1        | 32, 64, 128| -               |
| N1index   | 8, 16, 32, 64| -               |
| N2        | 32, 64, 128| N2=N1           |
| N2index   | 1          | -               |
| D         | 128        | -               |
| Dr        | 64         | -               |

## Examples

- Single-operator call

    ```python
    import torch
    import torch_npu

    def gen_inputs(isTnd):
        B = 1
        N1 = 64
        N2 = N1
        N1_index = 64
        N2_index = 1
        S1 = 128
        S2 = 256
        D = 128
        Dr = 64
        output_dtype = torch.float16
        q = torch.randn(B, S1, N1, D, dtype=output_dtype, device=torch.device('npu'))
        k = torch.randn(B, S2, N2, D, dtype=output_dtype, device=torch.device('npu'))
    
        q_index = torch.randn(B, S1, N1_index, D, dtype=output_dtype, device=torch.device('npu'))
        k_index = torch.randn(B, S2, N2_index, D, dtype=output_dtype, device=torch.device('npu'))
        if Dr != 0:
            q_rope = torch.randn(B, S1, N1, Dr, dtype=output_dtype, device=torch.device('npu'))
            k_rope = torch.randn(B, S2, N2, Dr, dtype=output_dtype, device=torch.device('npu'))
        else:
            q_rope = None
            k_rope = None
        weights = torch.randn(B, S1, N1_index, dtype=output_dtype, device=torch.device('npu'))
        if isTnd:
            q_tnd = q.squeeze(dim=0)
            k_tnd = k.squeeze(dim=0)
            q_index_tnd = q_index.squeeze(dim=0)
            k_index_tnd = k_index.squeeze(dim=0)
            if q_rope is not None:
                q_rope_tnd = q_rope.squeeze(dim=0)
                k_rope_tnd = k_rope.squeeze(dim=0)
            else :
                q_rope_tnd = None
                k_rope_tnd = None
            weights_tnd = weights.squeeze(dim=0)
            
            softmax_max = (torch.randn(N2, S1, 1, dtype=torch.float32, device=torch.device('npu')).abs() + 0.4) * D
            softmax_sum = torch.ones(N2, S1, 1, dtype=torch.float32, device=torch.device('npu'))
            actual_seq_qlen = [S1]
            actual_seq_klen = [S2]
            return q_tnd, k_tnd, q_index_tnd, k_index_tnd, q_rope_tnd, k_rope_tnd, weights_tnd, softmax_max, softmax_sum, actual_seq_qlen, actual_seq_klen
        else :
            softmax_max = (torch.randn(B, N2, S1, 1, dtype=torch.float32, device=torch.device('npu')).abs() + 0.4) * D
            softmax_sum = torch.ones(B, N2, S1, 1, dtype=torch.float32, device=torch.device('npu'))
            actual_seq_qlen = None
            actual_seq_klen = None
            return q, k, q_index, k_index, q_rope, k_rope, weights, softmax_max, softmax_sum, actual_seq_qlen, actual_seq_klen

    input_layout = 'TND'
    isTnd = True
    sparse_mode = 3
    scale = 1.0
    q, k, q_index, k_index, q_rope, k_rope, weights, softmax_max, softmax_sum, actual_seq_qlen, actual_seq_klen = gen_inputs(isTnd)
    softmax_max_index, softmax_sum_index = torch_npu.npu_dense_lightning_indexer_softmax_lse(q_index, k_index, weights,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_klen=actual_seq_klen,
            layout=input_layout,
            sparse_mode=sparse_mode)
    
    torch_npu.npu_dense_lightning_indexer_grad_kl_loss(
            q, k, q_index, k_index, weights, softmax_max, softmax_sum, softmax_max_index, softmax_sum_index, scale,
            query_rope=q_rope, key_rope=k_rope, actual_seq_qlen=actual_seq_qlen,
            actual_seq_klen=actual_seq_klen, layout=input_layout, sparse_mode=sparse_mode)
    ```

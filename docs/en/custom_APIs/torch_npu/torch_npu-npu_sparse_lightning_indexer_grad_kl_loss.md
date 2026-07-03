# torch_npu.npu_sparse_lightning_indexer_grad_kl_loss

## Supported Products

| Product                          | Supported|
|------------------------------| :------: |
| <term>Atlas A3 training products</term>| √  |
| <term>Atlas A2 training products</term>| √  |

## Function

- Description: Implements the backward computation of `npu_lightning_indexer` and integrates the computation of loss. `npu_lightning_indexer` selects the top-k items with the highest intrinsic connection between `query` or `key` for attention. The items are stored in `sparse_indices`. This reduces the attention computation workload in long-sequence scenarios and improves training performance.

- Formulas:

    1. Formula for top-k value computation:
        $$
        I_{t,:}=W_{t,:}@ReLU(\tilde{q}_{t,:}@\tilde{K}_{:t,:}^\top)
        $$
  
        - $W_{t,:}$ is the $weights$ corresponding to the $t$-th token.
        - $\tilde{q}_{t,:}$ is the result of merging the $G$ heads along the axis corresponding to the $t$-th token in the $Q_{index}$ matrix.
        - $\tilde{K}_{:t,:}$ is row $t$ of the $K_{index}$ matrix.
  
    2. Formula for forward Softmax computation:

        $$
        p_{t,:} = \text{Softmax}(q_{t,:} @ K_{:t,:}^\top/\sqrt{d})
        $$
    
        - $p_{t,:}$ is the Softmax result corresponding to the $t$-th token.
        - $q_{t,:}$ is the result of merging the $G$ query heads along the axis corresponding to the $t$-th token in the $Q$ matrix.
        - $K$ is the $t$-th row of the $K$ matrix.

    3. `npu_lightning_indexer` is trained separately, and the corresponding loss function is as follows:
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

        $S$ is the matrix multiplication result of $Q_{index}$ and $K_{index}$.

## Prototype

```python
npu_sparse_lightning_indexer_grad_kl_loss(query, key, query_index, key_index, weights, sparse_indices, softmax_max, softmax_sum, scale_value, *, query_rope=None, key_rope=None, actual_seq_qlen=None, actual_seq_klen=None, layout='BSND', sparse_mode=3, pre_tokens=2^63-1, next_tokens=2^63-1) -> (Tensor, Tensor, Tensor, Tensor)
```

## Parameters

**`query`** (`Tensor`): Required. Query in the attention, $q_{t}$ in the formula. The data layout can be ND. The data type can be `bfloat16` or `float16`. The shape can be `(B, S1, N1, D)` or `(T1, N1, D)`.

**`key`** (`Tensor`): Required. Key in the attention, $K_{t}$ in the formula. The data layout can be ND. The data type can be `bfloat16` or `float16`. The shape can be `(B, S2, N2, D)` or `(T2, N2, D)`.

**`query_index`** (`Tensor`): Required. Forward input `query` of `lightning_indexer`, $\tilde{q}_{t}$ in the formulas. The data layout can be ND. The data type can be `bfloat16` or `float16`. The shape can be `(B, S1, N1index, D)` or `(T1, N1index, D)`.

**`key_index`** (`Tensor`): Required. Forward input `key` of `lightning_indexer`, $\tilde{K}_{t}$ in the formulas. The data layout can be ND. The data type can be `bfloat16` or `float16`. The shape can be `(B, S2, N2index, D)` or `(T2, N2index, D)`.

**`weights`** (`Tensor`): Required. Weight coefficients of `lightning_indexer`, $W_{t}$ in the formula. The data layout can be ND. The data type can be `bfloat16`, `float16`, or `float32`. The shape can be `(B, S1, N1index)` or `(T1, N1index)`.

**`sparse_indices`** (`Tensor`): Required. Token indices of `key` or `key_index` after sorting. The data layout can be ND. The data type can be `int32`. The shape can be `[B, S1, N2index, topK]` or `[T1, N2index, topK]`.

**`softmax_max`** (`Tensor`): Required. Maximum value in the attention Softmax results. The data layout can be ND. The data type can be `bfloat16` or `float16`. The shape can be `(B, N2, S1, G)` or `(N2, T1, G)`.

**`softmax_sum`** (`Tensor`): Required. Sum of the attention Softmax results. The data layout can be ND. The data type can be `float32`. The shape can be `(B, N2, S1, G)` or `(N2, T1, G)`.

**`scale_value`** (`float`): Required. Scaling coefficient. The data type can be `float`.

**`query_rope`** (`Tensor`): Optional. RoPE information of the query in the MLA structure. The data layout can be ND. The data type can be `bfloat16` or `float16`. The shape can be `(B, S1, N1, Dr)` or `(T1, N1, Dr)`.

**`key_rope`** (`Tensor`): Optional. RoPE information of the key in the MLA structure. The data layout can be ND. The data type can be `bfloat16` or `float16`. The shape can be `(B, S2, N2, Dr)` or `(T2, N2, Dr)`.

**`actual_seq_qlen`** (`list[int]`): Optional. Accumulated sum of sequence lengths for each $S$ in the query tensor. This parameter is required in the `TND` scenario. The data type can be `int64`. The data layout can be ND. The default value is `None`.

**`actual_seq_klen`** (`list[int]`): Optional. This parameter must be provided in `TND` scenarios. Accumulated sum of sequence lengths for each S in the `key` tensor. The data type can be `int64`. The data layout can be ND. The default value is `None`.

**`layout`** (`string`): Optional. Data layout format of the input `query`. Valid values are `BSND` or `TND`. The default value is `BSND`.

**`sparse_mode`** (`int`): Optional. Sparse mode. The data type can be `int32`. The default value is `3`.

**`pre_tokens`** (`int`): Optional. Number of preceding tokens to associate in attention computation for sparse computation. The data type can be `int64`. The default value is `2^{63}-1`.

**`next_tokens`** (`int`): Optional. Number of subsequent tokens to associate in attention computation for sparse computation. The data type can be `int64`. The default value is `2^{63}-1`.

## Return Values

- **`d_query_index`** (`Tensor`): Gradient of `query_index`, $d\tilde{q}_{t}$ in the formula. The data type can be `bfloat16` or `float16`.
- **`d_key_index`** (`Tensor`): Gradient of `key_index`, $d\tilde{K}_{t}$ in the formula. The data type can be `bfloat16` or `float16`.
- **`d_weights`** (`Tensor`): Gradient of `weights`, $dW_{t}$ in the formula. The data type can be `bfloat16`, `float16`, or `float32`.
- **`loss`** (`Tensor`): Loss tensor representing the difference between the forward network output and the golden reference values, $dI_{t}$ in the formula. The data type can be `float`.

## Constraints

- The data types of `query`, `key`, `query_index`, and `key_index` must be identical.
- When the data type of `weights` is not `float32`, the data types of `query`, `key`, `query_index`, `key_index`, and `weights` must be identical.
- The following table describes the specification constraints.

| Item    | Specifications                                   | Description                 |
|---------|---------------------------------------|-----------------------|
| B       | 1 to 256                                | -                     |
| S1, S2  | S1 can be 1–8K; S2 can be 1–128K                  | `S1` and `S2` can have different lengths.           |
| N1      | 32, 64, 128                            | -                     |
| N1index | 8, 16, 32, 64                           | -                     |
| N2      | 1                                       | -                     |
| N2index | 1                                     | -                     |
| D       | 512                                   | The values of `D` for `query` and `query_index` are different.|
| Dr      | 64                                    | -                     |
| K       | 1024, 2048, 3081, 4096, 5120, 6144, 7168, 8192| -                     |
| layout  | BSND/TND                              | -                     |

## Examples

- Single-operator call

    ```python
    import torch
    import torch_npu
    import math

    def gen_inputs(seqlens_list_array, seqlens_list_kv_array, isTnd):
        B = 1
        NQuery = 64
        NQueryIndex = 64
        N2 = 1
        S1 = 128
        S2 = 128
        topK = 2048
        D = 512
        DIndex = 128
        DR = 64
        output_dtype = torch.float16
        q = torch.randn(B, S1, NQuery, D, dtype=output_dtype, device=torch.device('npu'))
        k = torch.randn(B, S2, N2, D, dtype=output_dtype, device=torch.device('npu'))
    
        q_index = torch.randn(B, S1, NQueryIndex, DIndex, dtype=output_dtype, device=torch.device('npu'))
        k_index = torch.randn(B, S2, N2, DIndex, dtype=output_dtype, device=torch.device('npu'))
        if DR != 0:
            q_rope = torch.randn(B, S1, NQuery, DR, dtype=output_dtype, device=torch.device('npu'))
            k_rope = torch.randn(B, S2, N2, DR, dtype=output_dtype, device=torch.device('npu'))
        else:
            q_rope = None
            k_rope = None
        weights = torch.randn(B, S1, NQueryIndex, dtype=output_dtype, device=torch.device('npu'))
        a = -0.05  # Minimum value
        b = 0.05 # Maximum value
        kk = 3.0  # Control the distribution range (3σ covers most values)
        scale = (b - a) / (2 * kk)
        shift = (a + b) / 2
        weights = weights * scale + shift
        if isTnd:
            sparse_indices = torch.zeros(S1, N2, topK).to(torch.int32).npu()
            tIdx = 0
            for bIdx in range(B):
                for s1Idx in range(seqlens_list_array[bIdx]):
                    s2RealSize = (int)((seqlens_list_kv_array[bIdx] - seqlens_list_array[bIdx]) + s1Idx + 1)
                    if s2RealSize <= 0:
                        s2RealSize = seqlens_list_kv_array[bIdx]
                    
                    if s2RealSize > topK:
                        s2RealLen = topK
                    else:
                        s2RealLen = s2RealSize
                    #Process invalid S2 row scenarios and set corresponding sparse indices to -1
                    sparse_indices[tIdx, :, 0 : s2RealLen] = (torch.randint(0, s2RealSize, (s2RealLen,)).to(torch.int32)).npu()
                    sparse_indices[tIdx, :, s2RealLen : topK] = -1
                    tIdx = tIdx + 1
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
            
            softmax_max = torch.randn(N2, S1, NQuery, dtype=torch.float, device=torch.device('npu'))
            softmax_sum = torch.randn(N2, S1, NQuery, dtype=torch.float, device=torch.device('npu'))
            return q_tnd, k_tnd, q_index_tnd, k_index_tnd, q_rope_tnd, k_rope_tnd, weights_tnd, sparse_indices, softmax_max, softmax_sum
        else :
            sparse_indices = torch.zeros(B, S1, N2, topK).to(torch.int32).npu()
            for s1Idx in range(S1):
                s2RealSize = (int)(S2 - S1 + s1Idx + 1)
                if s2RealSize <= 0:
                    s2RealSize = S2
    
                if s2RealSize > topK:
                    s2RealLen = topK
                else:
                    s2RealLen = s2RealSize
                sparse_indices[:, s1Idx, 0,  0 : s2RealLen] = (torch.randint(0, s2RealSize, (s2RealLen,)).to(torch.int32)).npu()
                sparse_indices[:, s1Idx, 0,  s2RealLen : topK] = -1
    
            softmax_max = torch.randn(B, N2, S1, NQuery, dtype=torch.float, device=torch.device('npu'))
            softmax_sum = torch.randn(B, N2, S1, NQuery, dtype=torch.float, device=torch.device('npu'))
            return q, k, q_index, k_index, q_rope, k_rope, weights, sparse_indices, softmax_max, softmax_sum


   actual_seq_qlen = [128]
   actual_seq_kvlen = [128]
   input_layout = 'TND'
   isTnd = True
   sparse_mode = 3
   scale = 1.0 / math.sqrt(512)
   q, k, q_index, k_index, q_rope, k_rope, weights, sparse_indices, softmax_max, softmax_sum = gen_inputs(actual_seq_qlen, actual_seq_kvlen, isTnd)
   
   torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
          q, k, q_index, k_index, weights, sparse_indices, softmax_max, softmax_sum, scale,
          query_rope=q_rope, key_rope=k_rope, actual_seq_qlen=actual_seq_qlen,
          actual_seq_klen=actual_seq_kvlen, layout=input_layout, sparse_mode=sparse_mode,
          pre_tokens=65536, next_tokens=65536)
    ```

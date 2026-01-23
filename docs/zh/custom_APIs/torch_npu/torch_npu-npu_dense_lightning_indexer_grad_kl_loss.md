# torch_npu.npu_dense_lightning_indexer_grad_kl_loss

## 产品支持情况
| 产品                           | 是否支持 |
|------------------------------| :------: |
| <term>Atlas A3 训练系列产品</term> | √  |
| <term>Atlas A2 训练系列产品</term> | √  |

## 功能说明

- API功能：该接口实现了npu_lightning_indexer warmup阶段训练的反向功能(dense计算)，并融合了Loss的计算。npu_lightning_indexer用于筛选Attention的`query`与`key`间最高内在联系的Top-k项，以减少长序列场景下的Attention计算量，提升训练性能。该API为稠密场景下对应的接口，相比较于`npu_sparse_lightning_indexer_grad_kl_loss`接口的输入，`key`、`key_index`不用做稀疏化处理。该函数与`npu_dense_lightning_indexer_softmax_lse`函数搭配使用，使用后者计算出来的`softmax_max_index`与`softmax_sum_index`降低算子显存占用。

- 计算公式：

  1. Top-k value的计算公式：
  $$
  I_{t,:}=W_{t,:}@ReLU(\tilde{q}_{t,:}@\tilde{K}_{:t,:}^\top)
  $$
  
    - $W_{t,:}$是第$t$个token对应的$weights$；
    - $\tilde{q}_{t,:}$是$\tilde{q}$矩阵第$t$个token对应的$G$个query头合轴后的结果；
    - $\tilde{K}_{:t,:}$为$t$行$\tilde{K}$矩阵。
  
  2. 正向的Softmax对应公式：

  $$
  p_{t,:} = \text{Softmax}(q_{t,:} @ K_{:t,:}^\top/\sqrt{d})
  $$
    
    - $p_{t,:}$是第$t$个token对应的Softmax结果；
    - $q_{t,:}$是$q$矩阵第$t$个token对应的$G$个query头合轴后的结果；
    - $K_{:t,:}$为$t$行$K$矩阵。

  3. npu_lightning_indexer会单独训练，对应的loss function为：
  $$
  Loss{=}\sum_tD_{KL}(p_{t,:}||Softmax(I_{t,:}))
  $$

  其中，$p_{t,:}$是target distribution，通过对main attention score 进行所有的head的求和，然后把求和结果沿着上下文方向进行L1正则化得到。$D_{KL}$为KL散度，其表达式为：

  $$
  D_{KL}(a||b){=}\sum_ia_i\mathrm{log}{\left(\frac{a_i}{b_i}\right)}
  $$

  4. 通过求导可得Loss的梯度表达式：
  $$
  dI\mathop{{}}\nolimits_{{t,:}}=Softmax \left( I\mathop{{}}\nolimits_{{t,:}} \left) -p\mathop{{}}\nolimits_{{t,:}}\right. \right.
  $$

  利用链式法则可以进行weights，query和key矩阵的梯度计算：
  $$
  dW\mathop{{}}\nolimits_{{t,:}}=dI\mathop{{}}\nolimits_{{t,:}}\text{@} \left( ReLU \left( S\mathop{{}}\nolimits_{{t,:}} \left) \left) \mathop{{}}\nolimits^{\top}\right. \right. \right. \right.
  $$
  
  $$
  d\mathop{{\tilde{q}}}\nolimits_{{t,:}}=dS\mathop{{}}\nolimits_{{t,:}}@\tilde{K}\mathop{{}}\nolimits_{{:t,:}}
  $$
  
  $$
  d\tilde{K}\mathop{{}}\nolimits_{{:t,:}}=\left(dS\mathop{{}}\nolimits_{{t,:}} \left) \mathop{{}}\nolimits^{\top}@\tilde{q}\mathop{{}}\nolimits_{{:t, :}}\right. \right.
  $$

  其中，$S$为$\tilde{q}$和$\tilde{K}$矩阵乘的结果。

## 函数原型

```
npu_dense_lightning_indexer_grad_kl_loss(query, key, query_index, key_index, weights, softmax_max, softmax_sum, softmax_max_index, softmax_sum_index, scale_value=1, *, query_rope=None, key_rope=None, actual_seq_qlen=None, actual_seq_klen=None, layout='BSND', sparse_mode=3, pre_tokens=2^63-1, next_tokens=2^63-1) -> (Tensor, Tensor, Tensor, Tensor)
```

## 参数说明
**query**(`Tensor`)：必选参数，表示Attention中的query，对应公式中的$q_{t}$。数据格式支持$ND$，数据类型支持`bfloat16`、`float16`。shape支持$(B, S1, N1, D)$、$(T1, N1, D)$。

**key**(`Tensor`)：必选参数，表示Attention中的key，对应公式中的$K_{t}$。数据格式支持$ND$，数据类型支持`bfloat16`、`float16`。shape支持$(B, S2, N2, D)$、$(T2, N2, D)$。

**query_index**(`Tensor`)：必选参数，表示lightning_indexer正向的输入`query`，对应公式中的$\tilde{q}_{t}$。数据格式支持$ND$，数据类型支持`bfloat16`、`float16`。shape支持$(B, S1, N1index, D)$、$(T1, N1index, D)$。

**key_index**(`Tensor`)：必选参数，表示lightning_indexer正向的输入`key`，对应公式中的$\tilde{K}_{t}$。数据格式支持$ND$，数据类型支持`bfloat16`、`float16`。shape支持$(B, S2, N2index, D)$、$(T2, N2index, D)$。

**weights**(`Tensor`)：必选参数，表示lightning_indexer的权重系数，对应公式中的$W_{t}$。数据格式支持$ND$，数据类型支持`bfloat16`、`float16`。shape支持$(B, S1, N1index)$、$(T1, N1index)$。

**softmax_max**(`Tensor`)：必选参数，表示Attention softmax结果中的最大值。数据格式支持$ND$，数据类型支持`float`。shape支持$(B, N2, S1, G)$、$(N2, T1, G)$。$G$等于$N1/N2$。

**softmax_sum**(`Tensor`)：必选参数，表示Attention softmax结果的求和。数据格式支持$ND$，数据类型支持`float`。shape支持$(B, N2, S1, G)$、$(N2, T1, G)$。$G$等于$N1/N2$。

**softmax_max_index**(`Tensor`)：必选参数，表示Index attention softmax结果中的最大值。数据格式支持$ND$，数据类型支持`float`。shape支持$(B, N2index, S1)$、$(N2index, T1)$。

**softmax_sum_index**(`Tensor`)：必选参数，表示Index attention softmax结果的求和。数据格式支持$ND$，数据类型支持`float`。shape支持$(B, N2index, S1)$、$(N2index, T1)$。

**scale_value**(`float`)：必选参数，表示缩放系数，数据类型支持`float`。默认值为`1`。

**query_rope**(`Tensor`)：可选参数，表示MLA结构中的query的rope信息。数据格式支持$ND$，数据类型支持`bfloat16`、`float16`。shape支持$(B, S1, N1, Dr)$、$(T1, N1, Dr)$。

**key_rope**(`Tensor`)：可选参数，表示MLA结构中的key的rope信息。数据格式支持$ND$，数据类型支持`bfloat16`、`float16`。shape支持$(B, S2, N2, Dr)$、$(T2, N2, Dr)$。

**actual_seq_qlen**(`list[int]`)：可选参数，TND场景时需传入此参数。表示query每个S的累加和长度，数据类型支持`int64`，数据格式支持$ND$，默认值为`None`。

**actual_seq_klen**(`list[int]`)：可选参数，TND场景时需传入此参数。表示key每个S的累加和长度，数据类型支持`int64`，数据格式支持$ND$，默认值为`None`。

**layout**(`str`)：可选参数，用于标识输入`query`的数据排布格式。当前支持$BSND$、$TND$，默认值为$BSND$。

**sparse_mode**(`int`)：可选参数，表示sparse的模式，数据类型支持`int32`，默认值为`3`。当前仅支持模式`3`。

**pre_tokens**(`int`)：可选参数，用于稀疏计算，表示Attention需要和前几个token计算关联。数据类型支持`int64`，默认值2^63-1。

**next_tokens**(`int`)：可选参数，用于稀疏计算，表示Attention需要和后几个token计算关联。数据类型支持`int64`，默认值2^63-1。




## 返回值说明
-   **d\_query\_index**(`Tensor`)：对应公式中的$d\tilde{q}_{t}$，表示`query_index`的梯度，数据类型支持`bfloat16`、`float16`。
-   **d\_key\_index**(`Tensor`)：对应公式中的$d\tilde{K}_{t}$，表示`key_index`的梯度，数据类型支持`bfloat16`、`float16`。
-   **d\_weights**(`Tensor`)：对应公式中的$dW_{t}$，表示`weights`的梯度，数据类型支持`bfloat16`、`float16`。
-   **loss**(`Tensor`)：对应公式中的$dI_{t}$，表示网络正向输出和golden值的差异，数据类型支持`float`。

## 约束说明
shape数值约束：
| 规格项    | 规格       | 规格说明         |
|-----------|------------|-----------------|
| B         | 1~256      | -               |
| S1、S2    | 1~128K     | S1、S2支持不等长 |
| N1        | 32、64、128| -               |
| N1index   | 16、32、64 | -               |
| N2        | 32、64、128| N2=N1           |
| N2index   | 1          | -               |
| D         | 128        | -               |
| Dr        | 64         | -               |


## 调用示例
- 单算子模式调用

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
    
    torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
            q, k, q_index, k_index, weights, softmax_max, softmax_sum, softmax_max_index, softmax_sum_index, scale,
            query_rope=q_rope, key_rope=k_rope, actual_seq_qlen=actual_seq_qlen,
            actual_seq_klen=actual_seq_klen, layout=input_layout, sparse_mode=sparse_mode)
    ```

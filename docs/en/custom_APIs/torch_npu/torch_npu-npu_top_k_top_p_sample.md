# torch\_npu.npu\_top\_k\_top\_p\_sample

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |

## Function

- Description:
  Performs top-K, top-P, and min-P sampling computations based on the input logit tensor `logits`, sampling parameters (`top_k`, `top_p`, `min_ps`), and random sampling weight distribution `q`. It outputs the index of the maximum logit value for each batch (`logits_select_idx`) and the logit distribution after top-K and top-P sampling (`logits_top_kp_select`).

  The operator provides four sampling algorithms that can be enabled independently while preserving the same processing pipeline (from raw input to final output): top-K sampling, top-P sampling, min-P saliency sampling, and no sampling/exponential sampling/multinomial random sampling. The following table describes the currently supported 12 computation scenarios.

  | Computation Scenario| Top-K Sampling| Top-P Sampling| Min-P Sampling| Subsequent Processing|Remarks|
  | :-------:| :------:|:-------:|:-------:|:-------:|:-------:|
  |Argmax sampling|×|×|×|None|Selects the maximum logit value and its corresponding index for each batch in `logits`. The result is assigned to `logits_select_idx[batch, 1]`.|
  |Top-K sampling|√|×|×|None|None|
  |Top-P sampling|×|√|×|None|None|
  |qSample sampling|×|×|×|qSample|Performs exponential sampling on each batch in `logits` using `q[i]`, and sets the maximum value and its index as `logits_select_idx[batch, 1]`.|
  |Top-K-Top-P sampling|√|√|×|None|None|
  |Top-K-qSample sampling|√|×|×|qSample|None|
  |Top-K-multiNomial sampling|√|×|×|multiNomial|None|
  |top-K-min-P-multiNomial sampling|√|×|√|multiNomial|None|
  |top-P-qSample sampling|×|√|×|qSample|None|
  |top-K-top-P-qSample sampling|√|√|×|qSample|Standard configuration used by the vLLM framework.|
  |top-K-top-P-multiNomial sampling|√|√|×|multiNomial|`min_ps` is an invalid value, but multinomial sampling is still performed.|
  |top-K-top-P-min-P-multiNomial sampling|√|√|√|multiNomial|Standard configuration used by the SGLang framework.|

- Formulas:
  The input `logits` is a logit tensor of shape `[batch, voc_size]`, where each batch corresponds to one input sequence and `voc_size` represents the uniform vocabulary size of each batch.<br>
  Each row in `logits` (`logits[batch][:]`), undergoes a different computation scenario determined by the corresponding `top_k[batch]`, `top_p[batch]`, `q[batch, :]`, and `min_ps[batch]`.<br>
  In the following descriptions, `b` and `v` represent the indices along the `batch` and `voc_size` dimensions, respectively.

  Top-K sampling
  1. A segmented top-K merge sort is performed based on the segment length `v`. The top-K results of block `{s-1}` are used to pre-filter the input of block `{s}`, progressively updating the top-K results for each batch and reducing redundant data and computation.
  2. `top_k[batch]` specifies the `k` value used for sampling the current batch. The valid range is $1 \le top\_k[batch] \le \min(voc\_size[batch], 1024)$. If `top_k[batch]` falls outside this range, the top-K sampling stage is skipped for that batch. Sorting is also skipped, and the input `logits[batch]` is passed directly to the next stage.<br>

  * The detailed computation process is as follows:

  * Determine whether to perform top-K sampling based on the relationship between the input `top_k[b]` and `ks_max`.

  | Parameter| ≤ | Valid Range| Invalid Range|
  | :-------:| :------:|:-------:|:-------:|
  |`top_k[b]`|Skip top-K sampling.|If `1≤topK≤min(voc_size,ks_max)`, perform top-K sampling.|If `top_k>min(voc_size,ks_max)`, skip top-K sampling.|

  * Divide the current batch into multiple segments and compute `top_k_value[b]` in a rolling manner:

    $$
    top\_k\_value[b] = {Max(top\_k[b])}_{s=1}^{\left \lceil \frac{S}{v} \right \rceil }\left \{ top\_k\_value[b]\left \{s-1 \right \}  \cup \left \{ logits[b][v] \ge top\_k\_min[b][s-1] \right \} \right \}\\
    Card(top\_k\_value[b])=top\_k[b]
    $$
    where
    $$
    top\_k\_min[b][s] = Min(top\_k\_value[b]\left \{  s \right \})
    $$
    `v` is the fixed segment length for the rolling top-K computation:

    $$
    v=8*ks\_max
    $$
    The value range of `ks_max` is [1, 1024] with a default value of `1024`, and it must be rounded up to a multiple of 8.

  * Generate the mask required for top-K filtering:

    $$
    top\_k\_mask = sorted\_value>top\_k\_value
    $$

  * Set the values smaller than the threshold to the default invalid value `defLogit` based on the mask:

    $$
    sorted\_value[b][v]=
    \begin{cases}
    -inf & \text{top\_k\_mask[b][v]=true} \\
    sorted\_value[b][v] & \text{top\_k\_mask[b][v]=false} &
    \end{cases}
    $$

  * `defLogit` depends on the input attribute property `Attr.optional.Bool.input_is_logits`, which controls the normalization of the input `logits` and the output `logits_top_kp_select`:
    $$
    \text{defLogit} =
      \begin{cases}
      -inf, & \text{inputIsLogits} = \text{True} \\
      0, & \text{inputIsLogits} = \text{False}
      \end{cases}
    $$

  Top-P sampling

  * Normalize the sorted results if the input constraint attribute `Attr.optional.Bool.input_is_logits` is set to `True`, which is `False` by default.

    $$
    \text{logit\_sortProb} =
    \begin{cases}
    \text{softmax}(\text{logits\_sort}), & \text{inputIsLogits} = \text{True} \\
    \text{logits\_sort}, & \text{inputIsLogits} = \text{False}
    \end{cases}
    $$

  * The processing strategy of this module varies depending on the value of the input `top_p[b]`:

  | Parameter| ≤ | Valid Range| Invalid Range|
  | :-------:| :------:|:-------:|:-------:|
  |`top_p[b]`|Retain one token with the maximum logit value.|If `0< top_p <1`, perform top-P sampling.|If `top_p ≥ 1`, skip top-P sampling.|

  * If regular top-P sampling is performed and the preceding top-K stage has already produced sorted results, the cumulative probabilities are computed based on the top-K output, and sampling is truncated according to `top_p`.

    $$
    topPMask[b] =
    \begin{cases}
    0, & \sum_{\text{topKMask}[b]}^{} \text{logits\_sortProb}[b][*] > p[b] \\
    1, & \sum_{\text{topKMask}[b]}^{} \text{logits\_sortProb}[b][*] \leq p[b]
    \end{cases}
    $$

  * If regular top-P sampling is performed but the preceding top-K stage is skipped, the top-P mask is computed.

    $$
    topPMask[b] =
    \begin{cases}
    topKMask[b][0:GuessK], & \sum_{\text{GuessK}}^{} probValue[b][*] \ge p[b] \\
    probSum[b][v] \le 1 - p[b], & \text{others}
    \end{cases}
    $$

  * Set the positions to be filtered to the default invalid value `defLogit` to obtain `logits_sort`, represented as $sortedValue[b][v]$.

    $$
    sortedValue[b][v] =
    \begin{cases}
    defLogit & \quad \text{topPMask}[b][v] = \text{false} \\
    logit\_sortProb[b][v] & \quad \text{topPMask}[b][v] = \text{true}
    \end{cases}
    $$

  * Select the top-K elements from each row of $sortedValue[b][v]$ after filtering, locate their original indices in the input, and combine them into `logits_idx`.

    $$
    logitsIdx[b][v] = Index(sortedValue[b][v] \in Logits)
    $$

  * Traverse the input `logits` according to the order of `logits_idx` to extract elements, and fill the remaining positions with `defLogit` to obtain `logitsSortMasked`.

    $$
    \text{logitsSortMasked}[b, \text{:Len}(\text{logitsIdx}[b][:])] = \text{Logits}[b, \text{logitsIdx}[b][:]]
    $$
    $$
    \text{logitsSortMasked}[b, \text{Len}(\text{logitsIdx}[b][:])\text{:}] = \text{defLogit}
    $$

  * (SGLang framework support update) Use the truncated `sortedValue` directly as `logitsSortMasked`.

    $$
    logitsSortMasked[b,:] = sortedValue[b]
    $$

  min-P sampling

  * If `min_ps[b] ∈ (0, 1)`, perform min-P sampling.

    $$
    \text{logitsMax}[b] = \text{Max}(\text{logitsSortMasked}[b])
    $$
    $$
    \text{minPThd} = \text{logitsMax}[b] * \text{minPs}[b]
    $$
    $$
    \text{minPMask}[b] =
    \begin{cases}
    0, & \text{logitsSortMasked}[b] < \text{minPThd} \\
    1, & \text{logitsSortMasked}[b] \geq \text{minPThd}
    \end{cases}
    $$
    $$
    \text{logitsSortMasked}[b,:] =
    \begin{cases}
    \text{defLogit}, & \text{minPMask}[b] = 0 \\
    \text{logitsSortMasked}[b,:], & \text{minPMask}[b] = 1
    \end{cases}
    $$

  * In other cases:

    $$
    \text{logitsSortMasked}[b, :] =
    \begin{cases}
        \text{logitsSortMasked}[b, :], & \text{if } minPs[b] \leq 0 \\
        \max(\text{logitsSortMasked}[b, :]), & \text{if } minPs[b] \geq 1
    \end{cases}
    $$
    When `min_ps[b] ≥ 1`, only 1 token with the maximum logit value is retained for each batch, and all other positions are filled with `defLogit`.

  Optional output

  * If the input attribute `Attr.Bool.is_need_logits=True`, `logitsIndexMasked` (generated after the joint top-K, top-P, and min-P sampling) is used to generate the `logits_top_kp_select` output.

    $$
    \text{logitsIndex}[b][v] = \text{Index}(\text{logitsSortMasked}[b][v] \in \text{Logits})
    $$
    $$
    \text{logitsIndexMasked}[b,:] = \text{logitsIndex}[b,:] * \text{topKMask}[b] * \text{topPMask}[b] * \text{minPMask}[b]
    $$
    If the top-K, top-P, or min-P sampling stage is skipped, its corresponding mask is set to all ones.

  * `logitsIndexMasked` is then used to perform a select operation on the input `logits`, extracting the high-frequency tokens to generate the `logits_top_kp_select` output.

    $$
    \text{logitsTopKpSelect}[b][v] =
    \begin{cases}
    \text{logits}[b][v], & \text{if } logitsIndexMasked[b,v] = \text{True} \\
    \text{defLogit}, & \text{if } logitsIndexMasked[b,v] = \text{False}
    \end{cases}
    $$

  Subsequent processing

  * The input to this stage is `logitsSortMasked`, which is the combined result of the preceding top-K, top-P, and min-P sampling stages.

  * `logitsSortMasked` must satisfy $\text{logitsSortMasked} \in (0,1)$. Configure the input constraint attribute `Attr.optional.Bool.input_is_logits` according to the actual input `logits`:

    $$
    \text{inputIsLogits} =
    \begin{cases}
    True, & \text{Logits} \notin [0,1] \\
    False, & \text{Logits} \in [0,1]
    \end{cases}
    $$
    Ensure that
    $$
    \text{probs}[b] = \text{logitsSortMasked}[b, :]
    $$
    Three modes are supported in this stage: `None`, `qSample`, and `multiNomial`, which are controlled by the input constraint attribute `attr.optional.Str.post_sample`.

  * None

  * For each batch, apply `Argmax` directly to obtain the maximum element and its index, and the result is output through `gatherOut`.
    $$
    \text{logitsSelectIdx}[b] = \text{LogitsIdx}[b]\left[\text{ArgMax}(\text{probs}[b][:])\right]
    $$

  * qSample

  * First, perform exponential-distribution sampling on `probs`: 

    $$
    qCnt = \text{Sum}(\text{MinPMask} == 1)
    $$
    $$
    \text{probsOpt}[b] = \frac{\text{probs}[b]}{q[b, :qCnt] + \text{eps}}
    $$

  * Then, generate the output through `Argmax` and `GatherOut`:

    $$
    \text{logitsSelectIdx}[b] = \text{LogitsIdx}[b][\text{ArgMax}(\text{probsOpt}[b][:])]
    $$

  * multiNomial

  * Perform multiNomial random sampling without replacement based on the probability values in `logitsSortMasked`. One sample is selected for each batch, and the sampled result is used as the output of the current batch:

    $$
    \text{sampleIdx}[b] = \text{multiNomial}(\text{logitsSortMasked}[b,:], \text{numSamples}=1, \text{seed}[b], \text{offset}[b])
    $$
    $$
    \text{logitsSelectIdx}[b] = \text{LogitsIdx}[b][\text{sampleIdx}[b]]
    $$

  * For the sampling seed, when `attr.optional.Str.post_sample="multiNomial"`, `q` must be of type `INT64`. The seed and offset for multiNomial sampling are obtained from the first and second columns of `q`, respectively:

    $$
    \text{seed}[b] =
    \begin{cases}
    q[b, 0], & b < qRows \\
    q[-1, 0], & b \ge qRows
    \end{cases}
    $$

    $$
    \text{offset}[b] =
    \begin{cases}
    q[b, 1], & b < qRows \\
    q[-1, 1], & b \ge qRows
    \end{cases}
    $$

  * This sampling process is based on `aclnn.Multinomial`. For details, see <https://gitcode.com/cann/ops-math-dev/blob/master/random/dsa_random_uniform/docs/aclnnMultinomial.md>.

  * When called through Ascend Extension for PyTorch, built-in values are used as the default sampling seed and offset. For details, see <https://gitcode.com/Ascend/op-plugin/blob/master/op_plugin/ops/opapi/MultinomialKernelNpuOpApi.cpp>.

## Prototype

```python
torch_npu.npu_top_k_top_p_sample(logits, top_k, top_p, q=None, eps=1e-8, is_need_logits=False, top_k_guess=32, min_ps=None, ks_max=1024, input_is_logits=True, post_sample='qSample', generator=None) -> (Tensor, Tensor)
```

## Parameters

- **`logits`** (`Tensor`): Required. Input logit tensor to be sampled. Currently, 2D tensors are supported, and the vocabulary dimension is fixed as the last dimension. The data type can be `float16`, `bfloat16`, or `float32`. The data layout can be ND. Non-contiguous tensors are supported.
- **`top_k`** (`Tensor`): Required. `k` value used for sampling each batch. The valid range is `1≤top_k[batch]≤min(voc_size[batch], 1024)`. If the value falls outside this range, the top-K stage is skipped for the corresponding batch. Currently, 1D tensors are supported. The data type can be `int32`, The data layout can be ND. Non-contiguous tensors are supported.
- **`top_p`** (`Tensor`): Required. `p` value used for sampling each batch. The valid range is `0 < top\_p[batch] < 1`. Currently, 1D tensors are supported. The data type and data layout match those of `logits`. Non-contiguous tensors are supported.
    - In all cases, the top-P output for each batch retains at least 1 token.
    - When `top_p[batch] ≤ 0`, only 1 token with the maximum logit value is retained for the current batch.
    - When `top_p[batch]` falls within the valid range (0, 1), standard top-P sampling is performed for the current batch.
    - When `top_p[batch] ≥ 1`, the top-P stage is skipped for the corresponding batch. The entire batch information is extracted, and an all-ones mask is generated as the output.
- **`q`** (`Tensor`): Optional. Random sampling weight distribution matrix for the top-K-top-P sampling output. The data type can be `float32`. The data layout can be ND. Non-contiguous tensors are supported. The default value is `None`. When `None` is provided, subsequent sampling is skipped, and `logits_select_idx` is computed directly from `probs`.
    - The constraints on this parameter depend on the `post_sample` mode:
    - When `post_sample = "qSample"`, the shape must be `[batch, voc_size]`, the data type must be `float32`, and the exponential distribution sampling matrix must have dimensions identical to those of `logits`.
    - When `post_sample = "multiNomial"`, `q` represents the multiNomial random sampling parameter matrix used to provide control parameters for `aclnnMultinomial` sampling. The data type must be `int64`. The valid shape is `[q_row, 2]`, where $q\_row \ge 1$.
        - The 1st column corresponds to the `aclnnMultinomial.seed` parameter, representing the random seed for the current batch.
        - The 2nd column corresponds to the `aclnnMultinomial.offset` parameter, representing the offset of the random number generator, which affects the starting position of the generated random number sequence. After the offset is set, the generated random number sequence starts from the specified position.
        - If `q_row < batch`, the sampling parameters of the last row are used by default as the multiNomial sampling parameters for all subsequent batches.
- **`eps`** (`float`): Optional. Prevents division by zero during softmax and weighted sampling. The default is `1e-8`.
- **`is_need_logits`** (`bool`): Optional. Controls the output conditions for `logits_top_kp_select`. The default is `False`.
- **`top_k_guess`** (`int`): Optional. Enabled only when `top_k` is an invalid value for the current batch. It is applicable for accelerated top-P sampling that skips the top-K stage (`top_k_guess` top-P accelerated sampling). The value range is `top_k_guess > 0` with a default value of `32`, which is used for direct index filtering based on `top_k_guess` during accelerated top-P sampling. If a non-positive value is passed, the `top_k_guess` stage is skipped, and the standard cumsum-based top-P implementation is used directly to perform full-sorting top-P sampling for the current batch, maintaining baseline performance.
- **`ks_max`** (`int`): Optional. Constrains the upper limit of valid `top_k[batch]` values allowed in top-K sampling, which affects the conditions for skipping top-K sampling. Any non-zero positive integer can be provided. The valid values are integers in the range [1, 1024]. Providing a value greater than `1024` automatically sets it to `1024`.
- **`input_is_logits`** (`bool`): Optional. Controls whether the input `logits` undergo normalization before top-P and subsequent steps, and determines the default invalid logit type for the optional output `logits_top_kp_select`. `logits` represents unnormalized raw values, whereas normalized values are defined as `probs`. The effects of this parameter value are as follows:
    - When this parameter is set to `True`, the values in the input `logits` are not guaranteed to be within the `[0, 1]` interval. Since the `logits` are unnormalized, softmax processing is applied to the input before top-P sampling and subsequent steps. The default invalid logit value in `logits_top_kp_select` is `defLogit = -inf`.
    - When this parameter is set to `False`, all elements in the input `logits` are guaranteed to be within the `[0, 1]` interval. Since the input `logits` are already normalized, subsequent steps such as top-P sampling directly use the results of the preceding stage to avoid gradient smoothing. The default invalid logit value in `logits_top_kp_select` is `defLogit = 0`.
- **`post_sample`** (`str`): Optional. Controls the subsequent processing strategy after top-K-top-P sampling. First, check whether `q` is `None`. If so, the content provided in the `post_sample` parameter is ignored, and the subsequent processing mode is forcibly set to `None`. Valid values are:
    - `qSample` (default): prioritizes the use of qSample sampling.
    - `multiNomial`: uses multiNomial sampling (multiNomial random sampling). In this case, the input `q` matrix is parsed as random seeds to execute multiNomial-gather operations.
    - `None`: explicitly emphasizes that no subsequent processing is used. In this case, `q!= None` is ignored.
- **`generator`** (`Generator`): Optional. The random number generator used for multiNomial sampling. A seed must be specified before this parameter can be provided.

## Return Values

- **`logits_select_idx`** (`Tensor`): Indicates the position index in the input `logits` of the element with the maximum logit value, `max(probs_opt[batch, :])`, for each batch after the top-K-top-P sampling computation process. The data type can be `int64`. The data layout can be ND.
- **`logits_top_kp_select`** (`Tensor`): Indicates the filtering and retention result of the high-frequency tokens in the original input `logits`, obtained by applying the mask generated through top-K-top-P-min-P sampling. Output computation and data transfer are enabled only when `is_need_logits` is set to `True`. Otherwise, an empty tensor of the corresponding size is returned directly. The data type can be `float32`. The data layout can be ND.

## Constraints

- This API can be used in inference scenarios.
- Currently, this API does not support graph mode.
- The shapes and dimensions of `logits`, `q`, and `logits_top_kp_select` must be identical.
- All dimensions except the last dimension of `logits`, `top_k`, `top_p`, and `logits_select_idx` must match exactly in size and order. Currently, `logits` must be a 2D tensor, and `top_k`, `top_p`, and `logits_select_idx` must be non-empty 1D tensors. Empty tensors are not allowed for `logits`, `top_k`, or `top_p`. To skip a corresponding module, configure the input according to the specified rules.
- To skip only the top-K module, provide a tensor with shape `[batch, 1]`, with all elements set to invalid values.
- If $1024 < top\_k[batch] < voc\_size[batch]$, all valid elements in the current batch are selected and the top-K stage is skipped.
- To skip only the top-P module, provide a tensor with shape `[batch, 1]`, with all elements set to values $\ge 1$ or $\le 0$.
- To skip only the sample module, use its default value or set `q` to `None`. To enable the sample module, you must provide a tensor with the corresponding shape.

## Example

```python
import numpy as np
import torch
import torch_npu
logits = torch.from_numpy(np.random.uniform(-2, 2, size=[2, 4])).type(torch.float16).npu()
top_ks = torch.from_numpy(np.random.uniform(1, 2, size=[2, ])).type(torch.int32).npu()
top_ps = torch.from_numpy(np.random.uniform(0.4, 0.5, size=[2, ])).type(torch.float16).npu()
q = None
min_ps = torch.from_numpy(np.random.uniform(0.1, 0.5, size=[2, ])).type(torch.float16).npu()
post_sample = 'multiNomial'
if post_sample == "multiNomial":
    generator_npu = torch.Generator(device="npu")
    generator_npu.manual_seed(1)
else:
    generator_npu = None
npu_out_index, logits_top_kp_select = torch_npu.npu_top_k_top_p_sample(logits, top_ks, top_ps, q=q, eps=1e-8, is_need_logits=True, top_k_guess=32, min_ps=min_ps, ks_max=1024, input_is_logits=True, post_sample=post_sample, generator=generator_npu)

print(npu_out_index)
print(logits_top_kp_select)
```

#

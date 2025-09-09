# torch_npu-npu_top_k_top_p_sample.md

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |

## 功能说明

- 基本功能：
  根据输入词频logits、topK/topP采样参数、随机采样权重分布q，进行topK-topP-sample采样计算，输出每个batch的最大词频logits_select_idx，以及topK-topP采样后的词频分布logitsTopKPSelect。

- 支持场景：
  算子包含三个可单独使能，但上下游处理关系保持不变的采样算法（从原始输入到最终输出）：TopK采样、TopP采样、指数采样（本文档中Sample所指）。它们可以构成八种计算场景。如下表所示：
  | 计算场景 | TopK采样 | TopP采样 | 指数分布采样 |备注|
  | :-------:| :------:|:-------:|:-------:|:-------:|
  |Softmax-Argmax采样|×|×|×|对输入logits按每个batch，取SoftMax后取最大结果|
  |topK采样|√|×|×||
  |topP采样|×|√|×||
  |Sample采样|×|×|√||
  |topK-topP采样|√|√|×||
  |topK-Sample采样|√|×|√||
  |topP-Sample采样|×|√|√||
  |topK-topP-Sample采样|√|√|√||

### 计算公式：
输入logits为大小为[batch, voc_size]的词频表，其中每个batch对应一条输入序列，而voc_size则是约定每个batch的统一长度。<br>
logtis中的每一行logtis[batch][:]根据相应的top_k[batch]、top_p[batch]、q[batch, :]，执行不同的计算场景。<br>
下述公式中使用b和v来分别表示batch和voc_size方向上的索引。

#### TopK采样
1. 按分段长度v采用分段topk归并排序，用{s-1}块的topK对当前{s}块的输入进行预筛选，渐进更新单batch的topK，减少冗余数据和计算。
2. top_k[batch]对应当前batch采样的k值，有效范围为1≤top_k[batch]≤min(voc_size[batch], 1024)，如果top_k[k]超出有效范围，则视为跳过当前batch的topK采样阶段，也同样会则跳过当前batch的排序，将输入logits[batch]直接传入下一模块。

* 对当前batch分割为若干子段，滚动计算top_k_value[b]：

$$
top\_k\_value[b] = {Max(top\_k[b])}_{s=1}^{\left \lceil \frac{S}{v} \right \rceil }\left \{ top\_k\_value[b]\left \{s-1 \right \}  \cup \left \{ logits[b][v] \ge top\_k\_min[b][s-1] \right \} \right \}\\
Card(top\_k\_value[b])=top\_k[b]
$$

其中：

$$
top\_k\_min[b][s] = Min(top\_k\_value[b]\left \{  s \right \})
$$

v表示预设的滚动topK时固定的分段长度：

$$
v=8*1024
$$

* 生成需要过滤的mask

$$
sorted\_value[b] = sort(top\_k\_value[b], descendant)
$$

$$
top\_k\_mask[b] = sorted\_value[b]<Min(top\_k\_value[b])
$$

* 将小于阈值的部分通过mask置为-inf

$$
sorted\_value[b][v]=
\begin{cases}
-inf & \text{top\_k\_mask[b][v]=true} \\
sorted\_value[b][v] & \text{top\_k\_mask[b][v]=false} &
\end{cases}
$$

* 通过softmax将经过topK过滤后的logits按最后一轴转换为概率分布

$$
probs\_value[b]=sorted\_value[b].softmax (dim=-1)
$$

* 按最后一轴计算累积概率（从最小的概率开始累加）

$$
probs\_sum[b]=probs\_value[b].cumcum (dim=-1)
$$

#### TopP采样

* 如果前序topK采样已有排序输出结果，则根据topK采样输出计算累积词频，并根据topP截断采样：
  $$
  top\_p\_mask[b] = probs\_sum[b][*] < top\_p[b]
  $$

* 如果topK采样被跳过，则先对输入logits[b]进行softmax处理：
$$
logits\_value[b] = logits[b].softmax(dim=-1)
$$
* 尝试使用top_k_guess，对logits进行滚动排序，获取计算topP的mask：
$$
top\_p\_value[b] = {Max(top\_k\_guess)}_{s=1}^{\left \lceil \frac{S}{v} \right \rceil }\left \{ top\_p\_value[b]\left \{s-1 \right \}  \cup \left \{ logits\_value[b][v] \ge top\_k\_min[b][s-1] \right \} \right \}
$$

* 如果在访问到logits_value[b]的第1e4个元素之前，下条件得到满足，则视为top_k_guess成功，
$$
\sum^{top\_k\_guess}(top\_p\_value[b]) \ge top\_p[b]\\
top\_p\_mask[b][Index(top\_p\_value[b])] = false
$$
* 如果top_k_guess失败，则对当前序logits_value[b]进行全排序和cumsum，按top_p[b]截断采样：
$$
sorted\_logits[b] = sort(logits\_value[b], descendant) \\
probs\_sum[b]=sorted\_logits[b].cumcum (dim=-1) \\
top\_p\_mask[b] = (probs\_sum[b] - sorted\_logits[b])>top\_p[b] 
$$

* 将需要过滤的位置设置为-inf，得到sorted_value[b][v]：

  $$
  sorted\_value[b][v] = \begin{cases} -inf& \text{top\_p\_mask[b][v]=true}\\sorted\_value[b][v]& \text{top\_p\_mask[b][v]=false}\end{cases}
  $$

  取过滤后sorted_value[b][v]每行中前topK个元素，查找这些元素在输入中的原始索引，整合为`logits_idx`:

  $$
  logits\_idx[b][v] = Index(sorted\_value[b][v] \in logits)
  $$
#### 指数采样（Sample）

* 如果`isNeedLogits=true`，则根据`logits_idx`，选取采样后结果输出到`logitsTopKPSelect`：

$$
logits\_top\_kp\_select[b][logits\_idx[b][v]]=sorted_value[b][v]
$$

* 对`logits_sort`进行指数分布采样：

  $$
  probs = softmax(logits\_sort)
  $$

  $$
  probs\_opt = \frac{probs}{q + eps}
  $$
* 从`probs_opt`中取出每个batch的最大元素，从`logits_idx`中gather相应元素的输入索引，作为输出`logits_select_idx`：
  
  $$
  logits\_select\_idx[b] = logits\_idx[b][argmax(probs\_opt[b][:])]
  $$

其中0≤b<sorted_value.size(0)，0≤v<sorted_value.size(1)。

## 函数原型:
torch_npu.npu_top_k_top_p_sample(logits, top_k, top_p, q=None, eps=1e-8, is_need_logits=False, top_k_guess=32)

## 参数说明:
-   logits（Tensor）：必选参数，表示待采样的输入词频，目前支持2维，词频索引固定为最后一维。数据类型支持float16和bfloat16，数据格式支持ND，支持非连续Tensor。
-   top_k（Tensor）：必选参数，表示每个batch采样的k值，有效范围为1≤top_k[batch]≤min(voc_size[batch], 1024)，目前支持1维。数据类型支持int32,数据格式支持ND，支持非连续Tensor。
-   top_p（Tensor）：必选参数，表示每个batch采样的p值，有效范围为0<top_p[batch]<1，目前支持1维。数据类型和数据格式与logits保持一致，支持非连续Tensor。
-   q（Tensor）：可选参数，topK-topP采样输出的指数采样矩阵，数据类型支持float32，数据格式支持ND，支持非连续Tensor。
-   eps（float）：可选参数，在softmax和权重采样中防止除零，默认值1e-8。
-   is_need_logits（bool）：可选参数，控制logits_top_kp_select的输出条件，默认为False。
-   top_k_guess（int）：可选参数，表示每个batch在尝试topP部分遍历采样时的候选logits大小，必须为正整数。

## 返回值说明：
-   logits_select_idx（Tensor）：表示经过topK-topP-sample计算流程后，每个batch中词频最大元素max(probs_opt[batch, :])在输入logits中的位置索引。数据类型支持int64，数据格式支持ND。
-   logits_top_kp_select（Tensor）：表示经过topK-topP计算流程后，输入logtis中剩余未被过滤的logits。数据类型支持float32，数据格式支持ND。

## 约束说明:
-   该接口支持推理场景下使用。
-   该接口目前不支持图模式。
-   logits、q、logits_top_kp_select的尺寸和维度必须完全一致。
-   logits、top_k、top_p、logits_select_idx除最后一维以外的所有维度必须顺序和大小完全一致。目前logits只能是2维，top_k、top_p、logits_select_idx必须是1维非空Tensor。logits、top_k、top_p不允许空Tensor作为输入，如需跳过相应模块，需按相应规则设置输入。
-   如果需要单独跳过topK模块，请传入[batch, 1]大小的Tensor，并使每个元素均为无效值。
-   如果1024<top_k[batch]<voc_size[batch]，则视为选择当前batch的全部有效元素并跳过topK环节。
-   如果需要单独跳过topP模块，请传入[batch, 1]大小的Tensor，并使每个元素均≥1。
-   如果需要单独跳过sample模块，传入q=None或使用其默认值即可；如需使用sample模块，则必须传入尺寸为[batch, voc_size]的Tensor。

## 调用示例
```python
>>> logits = torch.from_numpy(np.random.uniform(-2, 2, size=[2, 4])).type(torch.float16).npu()
>>> top_ks = torch.from_numpy(np.random.uniform(1, 2, size=[2, ])).type(torch.int32).npu()
>>> top_ps = torch.from_numpy(np.random.uniform(0.4, 0.5, size=[2, ])).type(torch.float16).npu()
>>> q = torch.from_numpy(np.random.uniform(0.1, 0.5, size=[2, 4])).type(torch.float32).npu()

>>> npu_out_index, logits_top_kp_select = torch_npu.npu_top_k_top_p_sample(logits, top_ks, top_ps, q, 1E-8, True)
>>> print(npu_out_index)
>>> print(logits_top_kp_select)
```
#
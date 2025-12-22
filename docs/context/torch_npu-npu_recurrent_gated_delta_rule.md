# torch_npu.npu_recurrent_gated_delta_rule

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |

## 功能说明

- API功能：该接口实现了变步长Recurrent Gated Delta Rule（RGDR）的计算逻辑，是Transformer线性注意力机制的关键算子之一。通过引入门控机制与递归更新策略，RGDR能够在保持线性时间复杂度的同时，有效捕捉长距离依赖关系，显著降低模型对序列长度的敏感性。

- 计算公式：

  在每个时间步 $t$，网络根据当前的输入 $q_t$、$k_t$、$v_t$ 和上一个隐藏状态 $S_{t-1}$，计算当前的输出 $o_t$ 和新的隐藏状态 $S_t$。
  在这个过程中，门控单元会决定有多少新信息存入隐藏状态，以及有多少旧信息需要被遗忘。
  $$
  S_t := S_{t-1}(\alpha_t(I - \beta_t k_t k_t^T)) + \beta_t v_t k_t^T = \alpha_t S_{t-1} + \beta_t (v_t - \alpha_t S_{t-1}k_t)k_t^T
  $$
  $$
  o_t := \frac{S_t q_t}{\sqrt{D_k}}
  $$

  其中，$S_{t-1},S_t \in R^{D_v \times D_k}$，$q_t, k_t \in R^{D_k}$，$v_t \in R^{D_v}$，$\alpha_t \in R$，$\beta_t \in R$，$o_t \in R^{D_v}$。

## 函数原型

```
torch_npu.npu_recurrent_gated_delta_rule(query, key, value, state, *, beta=None, scale=None, actual_seq_lengths=None, ssm_state_indices=None, num_accepted_tokens=None, g=None, gk=None) -> Tensor
```

## 参数说明

$T=\sum_i^B L_i$ 表示累积序列长度，$B$ 表示batch size。$L_i$ 表示第i个序列的长度，其取值范围为$1 \le L_i \le 8$。$N_k$ 表示key的头数，其取值范围为$1 \le N_k \le 256$。$N_v$ 表示value的头数，其取值范围为$1 \le N_v \le 256$。$D_k$ 表示key向量的维度，其取值范围为$1 \le D_k \le 512$。$D_v$ 表示value向量的维度，其取值范围为$1 \le D_v \le 512$。$BlockNum$为状态矩阵分块数量，其值不小于$T$。
- **query** (`Tensor`)：必选输入，对应公式中的$q$，数据类型支持`bfloat16`，数据格式支持ND，shape为（$T$, $N_k$, $D_k$）。

- **key** (`Tensor`)：必选输入，对应公式中的$k$，数据类型支持`bfloat16`，数据格式支持ND，shape为（$T$, $N_k$, $D_k$）。

- **value** (`Tensor`)：必选输入，对应公式中的$v$，数据类型支持`bfloat16`，数据格式支持ND，shape为（$T$, $N_v$, $D_v$）。

- **state** (`Tensor`)：必选输入&输出，对应公式中的状态矩阵$S$，数据类型支持`bfloat16`，数据格式支持ND，shape为（$BlockNum$, $N_v$, $D_v$, $D_k$）。

- **beta** (`Tensor`)：必选输入。对应公式中的$β$，数据类型支持`bfloat16`，数据格式支持ND，shape为（$T$, $N_v$）。

- **scale** (`float`)：必选输入。表示query的缩放因子，对应公式中的 $1/\sqrt{D_k}$。数据类型支持`float32`。

- **actual_seq_lengths** (`Tensor`)：必选输入。表示各batch的输入序列长度。数据类型支持`int32`，数据格式支持ND，shape为（$B$,）。

- **ssm_state_indices** (`Tensor`)：必选输入。表示输入序列到状态矩阵的映射索引。`state[ssm_state_indices[i]]`表示第i个token的状态矩阵。数据类型支持`int32`，数据格式支持ND，shape为（$T$,）。

- **num_accepted_tokens** (`Tensor`)：可选输入，投机推理每个batch接受的token数量。默认为None，表示每个batch接受的token数为1。数据类型支持`int32`，数据格式支持ND，shape为（$B$,）。

- **g** (`Tensor`)：可选输入，衰减系数，对应公式中的$α=e^g$。默认为None，表示全0。数据类型支持`float32`，数据格式支持ND，shape为（$T$, $N_v$）。

- **gk** (`Tensor`)：可选输入，衰减系数，当前版本暂不支持，传None即可。

## 返回值说明

`Tensor`

公式中的$o$，注意力计算结果。输出的数据类型为`bfloat16`，数据格式为ND，shape为($T$, $N_v$, $D_v$)。

## 约束说明

- 该接口仅支持推理场景下使用。
- 该接口仅支持单算子和静态图模式。

## 调用示例

- 单算子调用
    ```python
    import torch
    import torch_npu

    # 构造输入
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

    # 调用算子
    o = torch_npu.npu_recurrent_gated_delta_rule(
        query, key, value, state, beta=beta, scale=scale,
        actual_seq_lengths=actual_seq_lengths, ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens, g=g, gk=None)
    print(o)
    ```

- 静态图模式调用
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

    # "ENABLE_ACLNN"是否使能走aclnn, true: 回调走aclnn, false: 在线编译
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


    # 构造输入
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

    #调用
    model = MyModel()
    model = model.npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    o = model(query, key, value, state, beta, scale, actual_seq_lengths, ssm_state_indices, num_accepted_tokens, g)
    print(o)
    ```

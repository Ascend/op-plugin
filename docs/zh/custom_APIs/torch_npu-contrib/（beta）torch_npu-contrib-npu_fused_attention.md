# （beta）torch_npu.contrib.npu_fused_attention

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

BERT自注意力的融合实现。

## 函数原型

```python
torch_npu.contrib.npu_fused_attention(hidden_states, attention_mask, query_kernel, key_kernel, value_kernel, query_bias, key_bias, value_bias, scale=1, keep_prob=0)
```

## 参数说明

- **hidden_states** (`Tensor`)：必选参数，最后一层的hidden_states。数据格式支持FRACTAL_NZ。shape为2维，其中第0维需为32的整数倍，第1维支持768或1024。
- **attention_mask** (`Tensor`)：必选参数，掩码张量，用于屏蔽不需要参与注意力计算的token位置（如padding位置）。数据格式支持FRACTAL_NZ。shape为4维，形状为$(B, 1, S, S)$，其中B为batch size，S为序列长度。
- **query_kernel** (`Tensor`)：必选参数，query的权重矩阵。数据格式支持FRACTAL_NZ。shape为2维，第0维支持768或1024。
- **key_kernel** (`Tensor`)：必选参数，key的权重矩阵。数据格式支持FRACTAL_NZ。shape为2维，第0维支持768或1024。
- **value_kernel** (`Tensor`)：必选参数，value的权重矩阵。数据格式支持FRACTAL_NZ。shape为2维，第0维支持768或1024。
- **query_bias** (`Tensor`)：必选参数，query的偏差值。数据格式支持ND。
- **key_bias** (`Tensor`)：必选参数，key的偏差值。数据格式支持ND。
- **value_bias** (`Tensor`)：必选参数，value的偏差值。数据格式支持ND。
- **scale** (`double`)：可选参数，计算score的缩放系数，默认值为1。
- **keep_prob** (`float`)：可选参数，计算中保留数据的概率，值等于1 - drop rate，默认值为0。

## 返回值说明

`Tensor`

self attention的结果。

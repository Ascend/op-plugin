# （beta）torch\_npu.npu\_bert\_apply\_adam
## 产品支持情况


| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √    |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

针对bert模型，获取adam优化器的计算结果。
## 函数原型

```
torch_npu.npu_bert_apply_adam(lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm, weight_decay, step_size=None, adam_mode=0, *, var,m,v)
```


## 参数说明

-   **var** (`Tensor`)：`float16`或`float32`类型张量。
-   **m** (`Tensor`)：数据类型和shape与`var`相同。
-   **v** (`Tensor`)：数据类型和shape与`var`相同。
-   **lr** (`Scalar`)：数据类型与`var`相同。Shape为\(1, \)。
-   **beta1** (`Scalar`)：数据类型与`var`相同。Shape为\(1, \)。
-   **beta2** (`Scalar`)：数据类型与`var`相同。Shape为\(1, \)。
-   **epsilon** (`Scalar`)：数据类型与`var`相同。Shape为\(1, \)。
-   **grad** (`Tensor`)：数据类型和shape与`var`相同。
-   **max\_grad\_norm** (`Scalar`)：数据类型与`var`相同。Shape为\(1, \)。
-   **global\_grad\_norm** (`Scalar`)：数据类型与`var`相同。Shape为\(1, \)。
-   **weight\_decay**(`Scalar`)：数据类型与`var`相同。Shape为\(1, \)。
-   **step\_size** (`Tensor`)：可选参数，数据类型与`var`相同。Shape为\(1, \)。默认值为None。
-   **adam\_mode** (`int`)：选择adam模式。0表示“adam”，1表示“mbert\_adam”。默认值为0。


## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> var_in = torch.rand(321538).uniform_(-32., 21.).npu()
>>> m_in = torch.zeros(321538).npu()
>>> v_in = torch.zeros(321538).npu()
>>> grad = torch.rand(321538).uniform_(-0.05, 0.03).npu()
>>> max_grad_norm = -1.
>>> beta1 = 0.9
>>> beta2 = 0.99
>>> weight_decay = 0.
>>> lr = 0.
>>> epsilon = 1e-06
>>> global_grad_norm = 0.
>>> var_out, m_out, v_out = torch_npu.npu_bert_apply_adam(lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm, weight_decay, out=(var_in, m_in, v_in))
>>> var_out
tensor([ 14.7733, -30.1218,  -1.3647,  ..., -16.6840,   7.1518,   8.4872],
      device='npu:0')
```


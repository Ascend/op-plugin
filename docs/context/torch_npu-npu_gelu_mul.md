# torch_npu.npu_gelu_mul

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>            |    √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>               | √   |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

- API功能：当输入Tensor的尾轴为32B对齐场景时，使用该API可对输入Tensor进行GELU与MUL结合的复合计算操作以提高算子性能，若尾轴为非32B对齐场景时，建议走小算子拼接逻辑，即按照下述公式分步拼接计算。
- 计算公式：

    给定输入张量`input`（最后一维长度为$2d$，$d$为正整数），函数$\text{GELUMUL}$计算流程如下：

    1. 拆分输入张量：
    
       沿最后一维将`input`拆分为两个形状相同的张量$x1$和$x2$。
         $$x_1 = \text{input}[..., :d], \quad x_2 = \text{input}[..., d:]$$
                 
         其中$x_1$、$x_2$形状与$\text{input}$除最后一维外一致，最后一维长度均为$d$。

    2. 应用GELU激活函数：

       对$x_1$应用GELU激活函数（模式由`approximate`控制），即$x_1 = \text{GELU}(x_1)$。
        - 若`approximate`为"tanh"（近似模式，计算效率高）：
        $$\text{GELU}(x) = 0.5 \cdot x \cdot \left[ 1 + \tanh\left( \sqrt{\frac{2}{\pi}} \cdot \left( x + 0.044715 \cdot x^3 \right) \right) \right]$$
        
        - 若`approximate`为"none"（高精度模式）：
        $$\text{GELU}(x) = 0.5 \cdot x \cdot \left[ 1 + \text{erf}\left( \frac{x}{\sqrt{2}} \right) \right]$$

    3. 逐元素乘积输出：

       将激活后的$x_1$与$x_2$逐元素相乘，得到最终输出张量。
        $$\text{out} = x_1 \cdot x_2$$
                 
        其中$\text{out}$形状与原始输入`input`完全一致。


## 函数原型

```
torch_npu.npu_gelu_mul(input, *, approximate="none") -> Tensor
```

## 参数说明

- **input** (`Tensor`)：必选参数，输入张量，数据类型支持`bfloat16`、`float16`、`float`。支持非连续的Tensor，数据格式支持$ND$，shape维度2至8维，且shape需满足最后一维值为偶数且小于等于1024。其他维度的乘积小于等于20000。
- **approximate** (`String`)：可选参数，指定GELU激活函数的计算模式。默认值为 "none"。支持以下选项：
  - "none"：使用误差函数（erf）模式，计算精度高，适用于对精度要求严格的场景。
  - "tanh"：使用双曲正切（tanh）近似模式，计算效率高，适用于大规模训练或推理加速场景。

## 返回值说明
`Tensor`

输出张量，对应公式中的$out$，数据类型支持bfloat16、float16、float。shape维度2至8维。支持非连续的Tensor，数据格式支持$ND$，输出的数据类型与输入`input`保持一致，输出shape和输入shape其他维度一致，最后一维的值为输入shape最后一维值的二分之一。

## 调用示例

```python
>>> import torch, torch_npu
>>> shape = [100, 400]
>>> input = torch.rand(shape, dtype=torch.float16).npu()
>>> output = torch_npu.npu_gelu_mul(input, approximate=mode)

```
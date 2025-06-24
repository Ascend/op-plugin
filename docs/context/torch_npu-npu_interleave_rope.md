# torch\_npu.npu\_interleave\_rope

## 功能说明

-   算子功能：针对单输入x进行旋转位置编码。
-   计算公式：

    ![](./figures/zh-cn_formulaimage_0000002238091144.png)

     其中：RotateHalf\(q\)表示将q的D维后半部分元素移至前半部分并乘以-1，后半部分用前半部分的值。

    ![](./figures/zh-cn_formulaimage_0000002237943254.png)

## 函数原型

```
torch_npu.npu_interleave_rope(Tensor x, Tensor cos, Tensor sin) -> Tensor
```

## 参数说明

-   x：Tensor类型，表示目标张量。要求为4D的Tensor，shape为\(B, N, S, D\)，数据类型支持bfloat16、float16，数据格式为ND，不支持非连续的Tensor。
-   cos：Tensor类型，表示RoPE旋转位置编码的余弦分量。要求为4D的Tensor，shape为\(B, N, S, D\)，S可以为1或与x的S相同，数据类型、数据格式与x一致，不支持非连续的Tensor。
-   sin：Tensor类型，表示RoPE旋转位置编码的正弦分量。shape、数据类型、数据格式需要与cos保持一致，不支持非连续的Tensor。

## 输出说明

y：Tensor类型，表示旋转编码后的结果。shape、数据类型、数据格式与输入x保持一致，不支持非连续的Tensor。

## 约束说明

-   该接口支持推理场景下使用。
-   该接口支持图模式（PyTorch 2.1版本）。
-   输入x、cos、sin的D维度均必须等于64。
-   cos、sin的N维度必须等于1。

## 支持的型号

-   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>

-   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>

## 调用示例

-   单算子模式调用

    ```python
    import torch
    import torch_npu
    
    # 生成随机数据
    x = torch.randn(32, 32, 1, 64, dtype = torch.float16)
    cos = torch.randn(32, 1, 1, 64, dtype = torch.float16)
    sin = torch.randn(32, 1, 1, 64, dtype = torch.float16)
    x_npu = x.npu()
    cos_npu = cos.npu()
    sin_npu = sin.npu()
    
    # 调用InterleaveRope算子
    q_embed_npu = torch_npu.npu_interleave_rope(x_npu, cos_npu, sin_npu)
    ```

-   图模式调用

    ```python
    # 入图方式
    import torch
    import torch_npu
    import torchair
    from torchair.configs.compiler_config import CompilerConfig
    
    # 生成随机数据
    x = torch.randn(32, 32, 1, 64, dtype = torch.float16)
    cos = torch.randn(32, 1, 1, 64, dtype = torch.float16)
    sin = torch.randn(32, 1, 1, 64, dtype = torch.float16)
    x_npu = x.npu()
    cos_npu = cos.npu()
    sin_npu = sin.npu()
    
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x_npu, cos_npu, sin_npu):
            return torch_npu.npu_interleave_rope(x_npu, cos_npu, sin_npu)
    
    # 实例化模型model
    model = Model().npu()
    # 从TorchAir获取NPU提供的默认backend
    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    # 使用TorchAir的backend去调用compile接口编译模型
    model = torch.compile(model, backend=npu_backend)
    
    # 调用InterleaveRope算子
    q_embed_npu = model(x_npu, cos_npu, sin_npu)
    ```


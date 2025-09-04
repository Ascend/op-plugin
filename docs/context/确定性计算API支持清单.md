# 确定性计算API支持清单

## 简介

在使用PyTorch框架进行训练时，若需要输出结果排除随机性，则需要设置确定性计算开关。在开启确定性计算时，当使用相同的输入在相同的硬件和软件上执行相同的操作，输出的结果每次都是相同的。

>**说明：**<br>
>- 确定性计算固定方法都必须与待固定的网络、算子等在同一个主进程，部分模型脚本中main()与训练网络并不在一个进程中。
>- 当前同一线程中只能设置一次确定性状态，多次设置以第一次有效设置为准，后续设置不会生效。
>    有效设置：在设置确定性状态后，真正执行了一次算子的任务下发，如果仅设置，没有算子下发，只能是确定性变量开启，并未下发给算子，因为不执行算子，不知道哪个算子需要执行确定性。
>    解决方案：
>    1.  暂不推荐一个线程多次设置确定性。
>    2.  该问题在二进制开启和关闭情况下均存在，在后续版本中会解决该问题。

## 使用方法

确定性计算的用法和效果具体可参考相应官方文档[torch.use_deterministic_algorithms](https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms)，本小节仅介绍开启确定性计算的方法。

>**注意：**<br>
>开启确定性开关可能会导致性能下降。

1.  开启确定性计算开关：

    ```
    torch.use_deterministic_algorithms(True)
    ```

2.  验证设置是否成功。

    1.  执行如下命令查询接口设置：

        ```
        torch.are_deterministic_algorithms_enabled()
        ```

    2.  回显示例如下：

        ```
        print(torch.are_deterministic_algorithms_enabled())
        ```

    执行训练时，打印此接口的返回值为True表示当前已开启确定性计算开关，返回False则表示未开启。

## API支持清单

目前昇腾支持确定性计算的自定义API为[（beta）torch_npu.npu_group_norm_swish](（beta）torch_npu-npu_group_norm_swish.md)。



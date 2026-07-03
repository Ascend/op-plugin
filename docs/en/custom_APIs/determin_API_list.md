# Supported Deterministic Computing APIs

## Overview

When using the PyTorch framework for training, you must configure the deterministic computation switch if you need to eliminate randomness from the output results. When deterministic computation is enabled, executing the same operation with the same input on the same hardware and software always produces the same output.

> [!NOTE]  
>
> - The configuration method for deterministic computation must run in the same main process as the target network or operator to be fixed. In some model scripts, `main()` and the training network do not run in the same process.
> - Currently, the deterministic state can be configured only once within the same thread. If it is configured multiple times, only the first effective configuration takes effect, and subsequent configurations do not take effect.<br>
>    An effective configuration means that after the deterministic state is configured, at least one operator task is actually dispatched and executed. If only the deterministic state is configured and no operator is dispatched, only the deterministic variable is enabled, but it is not dispatched to any operator. This is because without executing an operator, the framework cannot determine which operator requires deterministic execution.<br>
>    Solution:
>      1. You are advised not to configure the deterministic state multiple times within a single thread.
>      2. This issue exists regardless of whether binary mode is enabled or disabled, and it will be resolved in a future release.

## Usage

For details about the usage and effects of deterministic computation, see the official documentation for [torch.use_deterministic_algorithms](https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms). This section describes only how to enable deterministic computation.

> [!CAUTION]  
> Enabling the deterministic computation switch may degrade performance.

1. Enable deterministic computation:

    ```python
    torch.use_deterministic_algorithms(True)
    ```

2. Verify whether the configuration is successful.

    1. Run the following command to query the API configuration:

        ```python
        torch.are_deterministic_algorithms_enabled()
        ```

    2. The following output is displayed:

        ```python
        print(torch.are_deterministic_algorithms_enabled())
        ```

    During training, a return value of `True` from this API indicates that deterministic computation is enabled, whereas `False` indicates that it is disabled.

## Supported APIs

Currently, Ascend supports deterministic computation for the custom API [(beta) torch_npu.npu_group_norm_swish](torch_npu/torch_npu-npu_group_norm_swish.md).

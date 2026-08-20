# Supported Deterministic Computing APIs

## Overview

When training with the PyTorch framework, some operators may produce non-deterministic results during computation. If deterministic output is required, enable deterministic computation. When deterministic computation is enabled, performing the same operation with the same input on the same hardware and software produces the same output every time.

> [!NOTE]  
>
> - The configuration method for deterministic computation must run in the same main process as the target network or operator to be fixed. In some model scripts, `main()` and the training network do not run in the same process.
> - Currently, the deterministic state can be configured only once within the same thread. If it is configured multiple times, only the first effective configuration takes effect, and subsequent configurations do not take effect.<br>
>   Effective configuration: After the deterministic state is set, at least one operator task must actually be dispatched and executed. If the deterministic state is set without dispatching any operator, only the deterministic variable is enabled and the setting is not applied to any operator. This is because the framework cannot determine which operator requires deterministic computation until an operator is executed.<br>
>    Solution:
>      1. Repeatedly configuring deterministic computation within a single thread is not recommended.
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

## API Lists

When using <term>Atlas A2 training products/Atlas A2 inference products</term> or <term>Atlas A3 training products/Atlas A3 inference products</term>, the APIs listed in [Table 1](#api-list-for-enabling-deterministic-computation-1) may produce non-deterministic results during computation. Enabling deterministic computation ensures that the computation results are deterministic.

**Table 1** API list<a id="api-list-for-enabling-deterministic-computation-1"></a>

| API |
|-----|
| `torch_npu.npu_convolution_transpose` |
| `torch_npu.npu_linear` |
| `torch_npu.npu_deformable_conv2d` |

When using <term>Ascend 950DT</term>, the APIs listed in [Table 2](#api-list-for-enabling-deterministic-computation-2) may produce non-deterministic results during computation. Enabling deterministic computation ensures that the computation results are deterministic.

**Table 2** API list<a id="api-list-for-enabling-deterministic-computation-2"></a>

| API |
|-----|
| `torch_npu.npu_scatter_nd_update` |
| `torch_npu.npu_scatter_nd_update_` |
| `torch_npu.scatter_update` |
| `torch_npu.scatter_update_` |
| `torch_npu.npu_fusion_attention_grad` |

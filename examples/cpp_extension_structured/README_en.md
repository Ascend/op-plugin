# Adaptation Development and Usage (Structured)

<!-- md-trans-meta sourceCommit=c6d785c77c92c77d7fc800bb1620ff277b5750a9 translatedAt=2026-08-11T01:29:43.701Z pushedAt=2026-08-11T12:00:45.552Z -->

This document describes the complete process of custom NPU operator adaptation development using the TorchNPU single-operator API through C++ extensions. This process covers operator definition, operator adaptation, and ATen IR registration and binding. This sample focuses on the structured kernel adaptation method, which is applicable to scenarios where the ACLNN API semantics are consistent with the ATen IR and the adaptation layer logic is only responsible for output tensor allocation.

## Operator Adaptation Development

### Prerequisites

Before getting started, ensure that you have completed the installation of the following environments:

1. Install the NPU driver, firmware, and CANN software (Toolkit, ops, and NNAL) by referring to CANN Software Installation.

2. Install the PyTorch framework by referring to [Installation Guide](https://gitcode.com/Ascend/pytorch/tree/v2.7.1-26.1.0/docs/en/installation_guide/installation_description.md).
<!-- [CANN Software Installation](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum) -->
### Adaptation File Structure

```text
cpp_extension_structured/
├── cpp_extension_structured/
│   └── __init__.py                   # Build initialization file
├── build_and_run.sh                  # Script for quick building, installation, and testing
├── deprecated.yaml                   # Deprecated API configuration
├── gen.sh                            # Quick generation script: calls torchnpugen to generate operator adaptation code
├── setup.py                          # Project build script used to build the .whl package
├── npu_custom.yaml                   # Custom operator YAML (containing forward/backward ATen IR and ACLNN mapping)
├── npu_custom_derivatives.yaml       # Forward/backward binding configuration
├── test_native_functions.yaml        # NPU backend declarations (used during stub generation)
├── test/
│   └── test_npu_fast_gelu_custom.py  # Custom operator test script
└── README.md
```

### Procedure

> [!NOTE]
>
> Structured adaptation does not support forward and backward binding. You can bind the operator using Python by referring to [cpp_extension_full/module](../cpp_extension_full/module/README.md).

1. In the operator adaptation layer C++ directory (`csrc`), structured adaptation configuration is defined in the `npu_custom.yaml` file.

    - `func`: The operator signature exposed on the PyTorch side (ATen IR format).

    - `gen_opapi`: Input tensor (such as `self` or `grad`) used to deduce the shape (`size`) and data type (`dtype`) of the output tensor.

    - `exec`: Name of the underlying ACLNN call to be called.

    The code sample is as follows:

    ```yaml
    custom:
      - func: npu_fast_gelu_custom(Tensor self) -> Tensor
        op_api: all_version
        gen_opapi:
          out:
            size: self
            dtype: self
          exec: aclnnFastGelu
      - func: npu_fast_gelu_custom_backward(Tensor grad, Tensor self) -> Tensor
        op_api: all_version
        gen_opapi:
          out:
            size: grad
            dtype: grad
          exec: aclnnFastGeluBackward
    ```

2. Load the `.so` file in the `__init__.py` file under the `cpp_extension_structured` directory.

    ```Python
    import pathlib
    import torch
    # Load the custom operator library
    def _load_opextension_so():
        so_dir = pathlib.Path(__file__).parents[0]
        so_files = list(so_dir.glob('custom_cpp_extension_structured_lib*.so'))
        if not so_files:
            raise FileNotFoundError(f"not find custom_cpp_extension_structured_lib*.so in {so_dir}")
        so_path = str(so_files[0])
        torch.ops.load_library(so_path)
    _load_opextension_so()
    ```

## Usage Example

After completing the operator adaptation development, you can call the custom operator through C++ extensions.

1. Complete the creation, development, compilation, and deployment workflow for the custom operator project. For details, see CANN Ascend C Operator Development.

2. Download the code sample.

    ```bash
    # Download the code sample
    git clone https://gitcode.com/Ascend/op-plugin
    # Go to the code directory
    cd examples/cpp_extension_structured
    ```

3. Complete operator adaptation. For details, see [Operator Adaptation Development](#operator-adaptation-development).

4. Run the following command to compile, install, and execute the test script:

    ```bash
    bash build_and_run.sh
    ```

    The following output indicates successful execution:

    ```bash
    Ran xx tests in xx s
    OK
    ```
<!-- [CANN Ascend C Operator Development](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/programug/Ascendcopdevg/atlas_ascendc_map_10_0002.html) -->
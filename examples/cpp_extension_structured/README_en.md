# Adaptation Development and Usage (Structured)

This example project demonstrates the complete process of custom NPU operator adaptation development using the `torch_npu` single-operator API through C++ extensions. This process covers operator definition, operator adaptation, and ATen IR registration and binding. This project focuses on the structured kernel adaptation method, which is applicable to scenarios where the ACLNN API semantics are consistent with the ATen IR and the adaptation layer logic is only responsible for output tensor allocation.

## Operator Adaptation Development

### Prerequisites

Before getting started, ensure that you have completed the installation of the following environments:

1. Install the NPU driver, firmware, and CANN software (including the Toolkit, ops, and NNAL packages) by referring to [CANN Software Installation](https://www.hiascend.com/document/detail/zh/canncommercial/900/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum) (Commercial Edition) or [CANN Software Installation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum) (Community Edition).
2. Install the PyTorch framework by referring to [Ascend Extension for PyTorch Software Installation Guide](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.0.0/docs/en/installation_guide/installation_description.md).

### Adaptation File Structure

```text
cpp_extension_structured/
├── cpp_extension_structured/
│   └── __init__.py                   # Build initialization file
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

1. Create the custom operator project and complete the operator development, compilation, and deployment process. For details, see the [CANN Ascend C Operator Development Guide](https://www.hiascend.com/document/detail/en/canncommercial/900/programug/Ascendcopdevg/atlas_ascendc_map_10_0002.html).

2. Download the sample code.

    ```bash
    # Downloading sample code
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

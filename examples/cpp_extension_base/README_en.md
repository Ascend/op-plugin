# Adaptation Development and Usage (Basic Example)

This example project demonstrates the complete adaptation development workflow for calling custom operators through the `torch_npu` framework by using C++ extensions and `single-op` APIs. The workflow covers operator definition, operator adaptation, and ATen IR registration and binding, ultimately enabling calls to custom operators.

## Operator Adaptation Development

### Prerequisites

Before getting started, ensure that you have completed the installation of the following environments:

1. Install the NPU driver, firmware, and CANN software (including the Toolkit, ops, and NNAL packages) by referring to [CANN Software Installation](https://www.hiascend.com/document/detail/zh/canncommercial/900/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum) (Commercial Edition) or [CANN Software Installation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum) (Community Edition).
2. Install the PyTorch framework by referring to [Ascend Extension for PyTorch Software Installation Guide](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.0.0/docs/zh/installation_guide/installation_description.md).

### Adaptation File Structure

```text
├── build_and_run.sh                // Script for compiling and installing the custom operator wheel package and executing the test case
├── csrc                            // Directory of the C++ code at the operator adaptation layer
│   └── add_custom.cpp              // Forward and backward adaptation code, ATen IR registration, and binding for the custom operator
├── cpp_extension_base              // Python-side code for the custom operator package
│   ├── ops.py                      // Defines operator APIs
│   └── __init__.py                 // Python initialization file
├── setup.py                        // Compilation file of the wheel package
└── test                            // Directory for test cases
    └── test_add_custom.py          // Script for executing operator test cases in eager mode
```

### Procedure

1. Implement the C++ operator code, adaptation layer, custom operator schema registration, and implementation binding in the `add_custom.cpp` file under the operator adaptation layer C++ directory (`csrc`). PyTorch provides the `TORCH_LIBRARY` macro to define a globally unique namespace and register the operator schema within it. The code sample is as follows:

    > [!NOTE]
    > 
    > In multi-rank scenarios, you must add `const c10::OptionalDeviceGuard device_guard(device_of(Tensor))` to the adaptation code to ensure proper cross-device access.

    ```cpp
    // Register forward implementation for NPU devices
    at::Tensor add_custom_impl_npu(const at::Tensor& self, const at::Tensor& other)
    {
        const c10::OptionalDeviceGuard device_guard(device_of(self));
        // Allocate output memory
        at::Tensor result = at::empty_like(self);

        at::Scalar alpha = 1.0;

        // Call the ACLNN API for computation
        EXEC_NPU_CMD_EXT(aclnnAdd, self, other, alpha, result);
        return result;
    }

    // Forward and backward registration for NPU devices
    // NPU devices use PrivateUse1 in PyTorch 2.1 and later versions, and XLA in earlier versions (change PrivateUse1 to XLA for earlier versions)
    TORCH_LIBRARY_IMPL(cpp_extension_base, PrivateUse1, m) {
        m.impl("add_custom", &add_custom_impl_npu);
    }

    TORCH_LIBRARY(cpp_extension_base, m) {
        m.def("add_custom(Tensor self, Tensor other) -> Tensor");
    }
    ```

2. Add the operator call logic and load the `.so` file in the `__init__.py` and `ops.py` files under the `cpp_extension_base` directory. The code sample is as follows:

    ```Python
    # __init__.py
    __all__ = ['ops', 'add_custom']
    from .ops import add_custom
    import pathlib
    import torch
    def _load_opextension_so():
        so_dir = pathlib.Path(__file__).parents[0]
        so_files = list(so_dir.glob('custom_ops_lib*.so'))

        if not so_files:
            raise FileNotFoundError(f"not find custom_ops_lib*.so in {so_dir}")

        so_path = str(so_files[0])
        torch.ops.load_library(so_path)
    _load_opextension_so()

    # ops.py
    import torch
    def add_custom(self, other):
        return torch.ops.cpp_extension_base.add_custom(self, other)
    ```

## Usage Example

After completing the operator adaptation development, you can call the custom operator through C++ extensions.

1. Create the custom operator project and complete the operator development, compilation, and deployment process. For details, see the [CANN Ascend C Operator Development Guide](https://www.hiascend.com/document/detail/en/canncommercial/900/programug/Ascendcopdevg/atlas_ascendc_map_10_0002.html).
2. Download the sample code.

    ```bash
    # Downloading sample code
    git clone https://gitcode.com/Ascend/op-plugin
    # Go to the code directory
    cd examples/cpp_extension_base
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

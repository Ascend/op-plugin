# Adaptation Development and Usage (pybind-based)

This example project demonstrates the complete adaptation development workflow for calling custom operators through the `torch_npu` framework by using C++ extensions and `single-op` APIs. The workflow covers operator definition, operator adaptation, and ATen IR registration and binding, ultimately enabling calls to custom operators. Unlike the common `TORCH_LIBRARY` method, this example project uses `pybind` for binding and registration to achieve more flexible input type support.

## Operator Adaptation Development

### Prerequisites

Before getting started, ensure that you have completed the installation of the following environments:

1. Install the NPU driver, firmware, and CANN software (including the Toolkit, ops, and NNAL packages) by referring to [CANN Software Installation](https://www.hiascend.com/document/detail/zh/canncommercial/900/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum) (Commercial Edition) or [CANN Software Installation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum) (Community Edition).
2. Install the PyTorch framework by referring to [Ascend Extension for PyTorch Software Installation Guide](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.0.0/docs/en/installation_guide/installation_description.md).

### Adaptation File Structure

```text
├── build_and_run.sh                // Script for compiling and installing the custom operator wheel package and executing the test case
├── csrc                            // Directory of the C++ code at the operator adaptation layer
│   └── add_custom.cpp              // Forward and backward adaptation code, ATen IR registration, and binding for the custom operator
├── cpp_extension_pybind            // Python-side code for the custom operator package
│   ├── __init__.py                 // Python initialization file
│   └── ops                         // Defines operator APIs
│       └── __init__.py             // Python initialization file
├── setup.py                        // Compilation file of the wheel package
└── test                            // Directory for test cases
    └── test_add_custom.py          // Script for executing operator test cases in eager mode
```

### Procedure

1. Implement the C++ operator code, adaptation layer, custom operator schema registration, and implementation binding in the `add_custom.cpp` file under the operator adaptation layer C++ directory (`csrc`). The code sample is as follows:

    > [!NOTE]
    > 
    > The following is the code sample for single-device scenarios. In these scenarios, the `const c10::OptionalDeviceGuard device_guard(device_of(self));` statement is optional. In multi-device scenarios, this statement must be included in the adaptation code.  

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

    PYBIND11_MODULE(custom_ops_lib, m)
    {
        m.def("add_custom", &add_custom_impl_npu, "");
    }
    ```

2. Add the operator call logic and load the `.so` file in the `__init__.py` and `ops.py` files under the `cpp_extension_base` directory.

    ```Python
    # __init__.py
    __all__ = ['ops', 'add_custom']
    from .ops import add_custom

    # ops/__init__.py
    __all__ = ["add_custom"]
    from cpp_extension_pybind.custom_ops_lib import add_custom
    ```

## Usage Example

After completing the operator adaptation development, you can call the custom operator through C++ extensions.

1. Create the custom operator project and complete the operator development, compilation, and deployment process. For details, see the [CANN Ascend C Operator Development Guide](https://www.hiascend.com/document/detail/en/canncommercial/900/programug/Ascendcopdevg/atlas_ascendc_map_10_0002.html).

2. Download the sample code.

    ```bash
    # Downloading sample code
    git clone https://gitcode.com/Ascend/op-plugin
    # Go to the code directory
    cd examples/cpp_extension_pybind
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
    
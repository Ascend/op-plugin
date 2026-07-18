# Adaptation Development and Usage (Complete Module Example)

This example project demonstrates how to adapt and call a single-operator API through `torch_npu` using C++ extensions. The forward and backward bindings are registered through the `forward` and `backward` methods of a Python class.

## Operator Adaptation Development

### Prerequisites

Before getting started, ensure that you have completed the installation of the following environments:

1. Install the NPU driver, firmware, and CANN software (including the Toolkit, ops, and NNAL packages) by referring to [CANN Software Installation](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum).
2. Install the PyTorch framework by referring to [Ascend Extension for PyTorch Software Installation Guide](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.0.0/docs/en/installation_guide/installation_description.md).

### Adaptation File Structure

```text
├── build_and_run.sh                // Script for compiling and installing the custom operator wheel package and executing the test case
├── csrc                            // Directory of the C++ code at the operator adaptation layer
│   └── add_custom.cpp              // Forward and backward adaptation code, ATen IR registration, and binding for the custom operator
├── cpp_extension_full              // Python-side code for the custom operator package
│   ├── ops.py                      // Defines operator APIs
│   └── __init__.py                 // Python initialization file
├── setup.py                        // Compilation file of the wheel package
└── test                            // Directory for test cases
    └── test_add_custom.py          // Script for executing operator test cases
```

### Procedure

1. Implement the C++ operator adaptation code, register the custom operator schema, and bind the concrete implementations in `add_custom.cpp` under the operator adaptation layer C++ directory (`csrc`). PyTorch provides the `TORCH_LIBRARY` macro to define a new namespace and register the schema within the namespace. Note that the namespace name must be unique. The code sample is as follows:

    > [!NOTE]
    > 
    > In multi-device scenarios, you must add `const c10::OptionalDeviceGuard device_guard(device_of(Tensor))` to the adaptation code to ensure proper cross-device access.

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

        // Register backward implementation for NPU devices
        std::tuple<at::Tensor, at::Tensor> add_custom_backward_impl_npu(const at::Tensor& grad)
        {
            const c10::OptionalDeviceGuard device_guard(device_of(grad));
            at::Tensor result = grad; // Allocate output memory

            return {result, result};
        }

        // Register forward implementation for Meta devices
        at::Tensor add_custom_impl_meta(const at::Tensor& self, const at::Tensor& other)
        {
            return at::empty_like(self);
        }

        // Register backward implementation for Meta devices
        std::tuple<at::Tensor, at::Tensor> add_custom_backward_impl_meta(const at::Tensor& self)
        {
            auto result = at::empty_like(self);
            return std::make_tuple(result, result);
        }

        // Register forward and backward implementations for NPU devices
        // NPU devices use PrivateUse1 in PyTorch 2.1 and later versions, and XLA in earlier versions (change PrivateUse1 to XLA for earlier versions)
        TORCH_LIBRARY_IMPL(cpp_extension_full, PrivateUse1, m) {
            m.impl("add_custom", &add_custom_impl_npu);
            m.impl("add_custom_backward", &add_custom_backward_impl_npu);
        }

        // Register forward and backward implementations for Meta devices
        TORCH_LIBRARY_IMPL(cpp_extension_full, Meta, m) {
            m.impl("add_custom", &add_custom_impl_meta);
            m.impl("add_custom_backward", &add_custom_backward_impl_meta);
        }

        TORCH_LIBRARY(cpp_extension_full, m) {
            m.def("add_custom(Tensor self, Tensor other) -> Tensor");
            m.def("add_custom_backward(Tensor self) -> (Tensor, Tensor)");
        }

    ```

2. Add the operator call logic and load the `.so` file in the `__init__.py` and `ops.py` files under the `cpp_extension_full` directory. The code sample is as follows:

    ```Python
    # __init__.py
    __all__ = ['ops', 'add_custom', 'add_custom_backward']
    from .ops import add_custom, add_custom_backward
    import pathlib
    import torch
    # Load the custom operator library
    def _load_opextension_so():
        so_dir = pathlib.Path(__file__).parents[0]
        so_files = list(so_dir.glob('custom_ops_lib*.so'))
        if not so_files:
            raise FileNotFoundError(f"not find custom_ops_lib*.so in {so_dir}")
        atb_so_path = str(so_files[0])
        torch.ops.load_library(atb_so_path)
    _load_opextension_so()

    # ops.py
    import torch
    def add_custom(self, other):
        return torch.ops.cpp_extension_full.add_custom(self, other)
    def add_custom_backward(grad):
        return torch.ops.cpp_extension_full.add_custom_backward(grad)
    ```

3. Implement the forward and backward bindings in the test script `test_add_custom.py` to define the forward computation and backward gradient computation of the operator using Python methods. The code sample is as follows:

    ```Python
    class AddCustomFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, y):
            # Save tensors required for gradient computation
            ctx.save_for_backward(x, y)
            # Execute forward computation
            return torch.ops.cpp_extension_full.add_custom(x, y)
        
        @staticmethod
        def backward(ctx, grad_output):
            # Retrieve the saved tensors
            x, y = ctx.saved_tensors
            return torch.ops.cpp_extension_full.add_custom_backward(grad_output)
    ```

## Usage Example

After completing the operator adaptation development, you can call the custom operator through C++ extensions.

1. Create the custom operator project and complete the operator development, compilation, and deployment process. For details, see the [CANN Ascend C Operator Development Guide](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_map_10_0002.html).
2. Download the sample code.

    ```bash
    # Download sample code
    git clone https://gitcode.com/Ascend/op-plugin
    # Go to the code directory
    cd examples/cpp_extension_full/module
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

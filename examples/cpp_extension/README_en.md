# Adaptation Development and Usage

This sample demonstrates how to implement a custom operator kernel using Ascend C, integrate it into PyTorch through a C++ extension (`cpp_extension`), and finally call it from Python. It covers the complete workflow of project structure, compilation, installation, and testing.

## Directory Structure

```text
├── examples
|   ├── cpp_extension
|   │   ├── csrc
|   │   │   ├── add_custom.asc          # Add operator implementation (comparison of five correct launch methods)
|   │   │   ├── trig_inplace_custom.asc # In-place trigonometric operator implementation
|   │   │   └── pybind11.asc            # pybind binding for custom operators
|   |   ├── op_extension/
|   |   │   ├── __init__.py             # Extension loading logic
|   |   │   └── ops/
|   |   │       └── __init__.py         # Python API definition and extension loading logic
|   |   ├── test
|   │   |   └── test.py                 # Test script
|   |   ├── setup.py                    # Build configuration
|   |   └── README.md                   # Documentation
```

## Adding a New Custom Operator

### 1. Kernel implementation

Create an `.asc` file in `./csrc/` and implement the operator kernel based on Ascend C. For Ascend C development, see the [Ascend Community documentation](https://www.hiascend.com/ascend-c).

Taking `add_custom.asc` as an example, the file contains three parts.

| Component | Description |
|------|------|
| `KernelAdd` class | Device-side implementation of the operator, including `Init` / `Process` / `CopyIn` / `Compute` / `CopyOut` |
| `add_custom` kernel function | Device entry point decorated with `__global__ __vector__` |
| `ascendc_add1/2/3/4/5` | Five correct host-side launch modes, comparing different stream acquisition and queue management strategies |

**Comparison of the five launch modes**

| Mode | Function | stream acquisition | Queue management | Applicable scenario |
|------|------|------------|---------|---------|
| 1 | `ascendc_add1` | `NPUStream` object | `<<<>>>` clears the queue internally | Simple synchronization scenario |
| 2 | `ascendc_add2` | `stream(true)` | Starts directly after clearing the queue | Equivalent to mode 1 |
| 3 | `ascendc_add3` | `stream(false)` | OpCommand is queued | **Recommended**; preserves pipeline performance |
| 4 | `ascendc_add4` | `stream(true)` | Clears the queue and then queues OpCommand | Intuitive semantics |
| 5 | `ascendc_add5` | `stream()` | Starts after the queue is complete | Equivalent to mode 2 |

> **Recommended: mode 3**: `stream(false)` together with `OpCommand::RunOpApiV2` preserves TaskQueue pipeline performance and behaves the same as the built-in TorchNPU operators.

### 2. Python module binding

The `pybind11.asc` file uses `pybind11` to expose C++ functions as a Python interface. The Python side exposes only recommended mode 3.

```cpp
namespace ascendc_ops {
// Mode 3 (recommended): stream(false) + OpCommand::RunOpApiV2
at::Tensor ascendc_add3(const at::Tensor &x, const at::Tensor &y);
at::Tensor run_trig_custom(const at::Tensor &x, const at::Tensor &out_sin, const at::Tensor &out_cos);
}

PYBIND11_MODULE(custom_ops_lib, m)
{
    m.def("custom_add", &ascendc_ops::ascendc_add3, "");
    m.def("custom_trig", &ascendc_ops::run_trig_custom, "");
}
```

### 3. Aten IR implementation

In this project, the `at_npu::native::OpCommand::RunOpApiV2` method is used to enqueue operator execution into the TorchNPU task queue. The code sample is as follows:

```cpp
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace ascendc_ops {
at::Tensor ascendc_add3(const at::Tensor &x, const at::Tensor &y)
{
    // stream(false) returns the ACL stream but does not clear the queue
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    at::Tensor z = at::empty_like(x);
    uint32_t blockDim = 8;
    uint32_t totalLength = 1;
    for (uint32_t size : x.sizes()) {
        totalLength *= size;
    }
    // Launch the custom kernel use <<<>>>
    auto acl_call = [=]() -> int {
        add_custom<<<blockDim, nullptr, acl_stream>>>(
            (uint8_t *)(x.mutable_data_ptr()),
            (uint8_t *)(y.mutable_data_ptr()),
            (uint8_t *)(z.mutable_data_ptr()),
            totalLength);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("ascendc_add", acl_call);
    return z;
}
}  // namespace ascendc_ops
```

The preceding section describes the mandatory workflow for integrating a custom operator kernel. For the complete set of five correct launch methods, see `./csrc/add_custom.asc`. The Python side exposes only recommended mode 3 through `./csrc/pybind11.asc`.

Finally, by creating the `ops` path and defining the Python interface, you can call the custom operator through `module_name.ops.custom_add`. The test code sample is as follows:

```python
import torch
x = torch.randint(low=1, high=100, size=length, device='cpu', dtype=torch.int)
y = torch.randint(low=1, high=100, size=length, device='cpu', dtype=torch.int)

x_npu = x.npu()
y_npu = y.npu()
output = op_extension.ops.custom_add(x_npu, y_npu)
```

## Running the Custom Operator

### 1. Compile the whl package

      ```bash
      python setup.py bdist_wheel
      ```

Key logic of `setup.py`

| Step | Implementation | Description |
|------|------|------|
| Source collection | `glob.glob("csrc/*.asc")` | Automatically collects all `.asc` files |
| Architecture detection | `get_npu_arch()` | Parses the chip model through `npu-smi info` and maps it to `dav-2201`/`dav-3510` |
| Dependency paths | `get_dependency_paths()` | Automatically collects the include and lib paths of torch / torch_npu / Python |
| ABI alignment | `torch._C._GLIBCXX_USE_CXX11_ABI` | Keeps consistent with the PyTorch ABI |
| Compiler | `bisheng -x asc` | Uses the bisheng compiler provided by CANN |

The build output is located at `dist/op_extension-0.1-*.whl`.

Optional environment variables:

```bash
USE_NINJA=1 python setup.py bdist_wheel  # enable ninja acceleration
```

### 2. Install the whl package

      ```bash
      cd dist
      pip install *.whl
      ```
  
  3. Run the sample.

      ```bash
      cd test
      python test.py
      ``` 

## Frequently Asked Questions (FAQ)
 
### 1. Compilation Error: "bisheng Command Not Found"

**Cause**: The `bisheng` compiler is not installed on the system, or environment variables are not correctly configured.
**Solution**:

- Ensure that the CANN toolkit is correctly installed.
- Run `source /usr/local/Ascend/ascend-toolkit/set_env.sh` to configure environment variables.
- Verify whether the `bisheng` compiler is available by running `bisheng --version`.
 
### 2. Runtime Error: "ModuleNotFoundError: No module named 'op_extension'"

**Cause**: The custom extension package is not installed correctly.
**Solution**:

- Ensure the wheel package has been successfully built and installed.
- Check whether the installation directory is included in the module search path of Python.
- Try reinstalling the wheel package through `pip install --force-reinstall *.whl`.

## Precautions
 
### 1. Data Type Support

- The current implementation only supports the `int32` data type.
- To support other data types (such as `float32` or `float16`), you must modify the kernel implementation.
- When modifying the implementation, pay attention to the byte size and alignment requirements for data types.
 
### 2. Hardware Compatibility

- This sample is developed based on Ascend NPUs and supports only Ascend hardware.
- Different Ascend NPU models may require adjustments to compilation parameters.
- You must specify the correct NPU architecture during compilation, such as `--npu-arch=dav-2201`.
 
### 3. Version Compatibility

- Ensure that the PyTorch, TorchNPU, and CANN versions are compatible.
- Version mismatch may cause compilation or runtime errors.
- Refer to the [TorchNPU Documentation](https://gitcode.com/Ascend/pytorch#installation) for compatible version information.

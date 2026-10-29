# (beta) struct c10_npu::NPUHooksInterface

## Supported Products

<!-- npu="910" id1 -->
- <term>Atlas training products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas inference products</term>: Supported
<!-- end id4 -->

## Function

Provides NPU hook APIs as a hook interface class.

## Definition File

torch_npu\csrc\core\npu\NPUHooksInterface.h

## Member Functions

- **const at::Generator& c10_npu::NPUHooksInterface::getDefaultGenerator(c10::DeviceIndex device_index)**

Obtains the default random number generator for `NPUHooksInterface`. This function is identical to `const at::Generator& at::CUDAHooksInterface::getDefaultCUDAGenerator(c10::DeviceIndex device_index = -1)`.

**`device_index`** (`DeviceIndex`): NPU device ID.

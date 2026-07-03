# (beta) struct c10_npu::NPUHooksInterface

## Definition File

torch_npu\csrc\core\npu\NPUHooksInterface.h

## Function

Provides NPU hook APIs as a hook interface class.

## Member Functions

**const at::Generator& c10_npu::NPUHooksInterface::getDefaultGenerator(c10::DeviceIndex**

**device_index)**

Obtains the default random number generator for `NPUHooksInterface`. This function is identical to `const at::Generator& at::CUDAHooksInterface::getDefaultCUDAGenerator(c10::DeviceIndex device_index = -1)`.

**`device_index`** (`DeviceIndex`): NPU device ID.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>

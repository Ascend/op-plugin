# (beta) class at_npu::NPUGeneratorImpl

## Definition File

torch_npu\csrc\aten\NPUGeneratorImpl.h

## Function

Acts as a random number generator class that implements NPU device random number functionality and supports various random-dependent methods.

## Member Functions

- **at_npu::NPUGeneratorImpl::NPUGeneratorImpl(c10::DeviceIndex device_index = -1)**

    `NPUGeneratorImpl` constructor, which constructs a generator by specifying an NPU device ID. This function is identical to `at::CUDAGeneratorImpl::CUDAGeneratorImpl(c10::DeviceIndex device_index)`.

    **`device_index`** (`DeviceIndex`): NPU device ID.

- **std::shared_ptr\<NPUGeneratorImpl> at_npu::NPUGeneratorImpl::clone()**

    `NPUGeneratorImpl` copy function. The return type is `std::shared_ptr<NPUGeneratorImpl>`, which represents the obtained `NPUGeneratorImpl` copy. This function is identical to `std::shared_ptr<CUDAGeneratorImpl> at::CUDAGeneratorImpl::clone()`.

- **void at_npu::NPUGeneratorImpl::set_current_seed(uint64_t seed)**

    Sets the current random number seed for `NPUGeneratorImpl`. This function is identical to `at::CUDAGeneratorImpl::set_current_seed(uint64_t seed)`.

    **`seed`** (`uint64_t`): Random number seed to be set.

- **void at_npu::NPUGeneratorImpl::set_offset(uint64_t offset)**

    Sets the offset value for `NPUGeneratorImpl`. This function is identical to `at::CUDAGeneratorImpl::set_offset(uint64_t offset)`.

    **`offset`** (`uint64_t`): Offset value to be set.

- **uint64_t at_npu::NPUGeneratorImpl::current_seed()**

    Obtains the current random number seed for `NPUGeneratorImpl`. The return type is `uint64_t`, which represents the current random number seed. This function is identical to `at::CUDAGeneratorImpl::current_seed()`.

- **uint64_t at_npu::NPUGeneratorImpl::get_offset()**

    Obtains the current offset value for `NPUGeneratorImpl`. The return type is `uint64_t`, which represents this offset value. This function is identical to `at::CUDAGeneratorImpl::get_offset()`.

- **uint64_t at_npu::NPUGeneratorImpl::seed()**

    Updates the random number seed for `NPUGeneratorImpl`. The return type is `uint64_t`, which generates and returns a new random number seed. This function is identical to `at::CUDAGeneratorImpl::seed()`.

- **void at_npu::NPUGeneratorImpl::set_state(const c10::TensorImpl& new_state)**

    Sets the specified state for `NPUGeneratorImpl`. This function is identical to `void at::CUDAGeneratorImpl::set_state(const c10::TensorImpl& new_state)`.

    **`new_state`** (`TensorImpl`): State to be set, which must be validated using `at::detail::check_rng_state`.

- **c10::intrusive_ptr\<c10::TensorImpl> c10::TensorImpl at_npu::NPUGeneratorImpl::get_state()**

    Obtains the generator state for `NPUGeneratorImpl`. The return type is `c10::intrusive_ptr<c10::TensorImpl>`, which represents the obtained generator state. This function is identical to `c10::intrusive_ptr<c10::TensorImpl> at::CUDAGeneratorImpl::get_state()`.

- **void at_npu::NPUGeneratorImpl::set_philox_offset_per_thread(uint64_t offset)**

    Sets the Philox offset value for each thread for `NPUGeneratorImpl`, which is used for `curandStatePhilox4_32_10`. This function is identical to `void at::CUDAGeneratorImpl::set_philox_offset_per_thread(uint64_t offset)`.

    **`offset`** (`uint64_t`): Required. Philox offset value to be set.

- **uint64_t at_npu::NPUGeneratorImpl::philox_offset_per_thread()**

    Obtains the Philox offset for each thread for `NPUGeneratorImpl`. The return type is `uint64_t`. This function is identical to `uint64_t at::CUDAGeneratorImpl::philox_offset_per_thread()`.

- **at_npu::PhiloxNpuState at_npu::NPUGeneratorImpl::philox_npu_state(uint64_t increment)**

    Captures the Philox NPU state for `NPUGeneratorImpl`. The return type is `PhiloxNpuState`. This function is identical to `at::PhiloxCudaState at::CUDAGeneratorImpl::philox_cuda_state(uint64_t increment)`.

    **`increment`** (`uint64_t`): Required. Philox offset increment.

- **std::pair<uint64_t, uint64_t> at_npu::NPUGeneratorImpl::philox_engine_inputs(uint64_t increment)**

    Obtains the Philox engine inputs for `NPUGeneratorImpl`. The return type is `std::pair<uint64_t, uint64_t>`, which contains the random number seed and the Philox offset value. This function is identical to `std::pair<uint64_t, uint64_t> at::CUDAGeneratorImpl::philox_engine_inputs(uint64_t increment)`.

    **`increment`** (`uint64_t`): Required. Philox offset increment.

- **c10::DeviceType at_npu::NPUGeneratorImpl::device_type()**

    Obtains the device type for `NPUGeneratorImpl`. The return type is `DeviceType`. This function is identical to `c10::DeviceType at::CUDAGeneratorImpl::device_type()`.

In PyTorch 2.5.1 and later, the following member functions are removed. In versions earlier than PyTorch 2.5.1, these member functions are still available:

- **void at_npu::NPUGeneratorImpl::capture_prologue()**

    Sets `offset_extragraph` for `NPUGeneratorImpl`, which is used by `NPUGraph` to reserve a graph capture region and enable graph capture. This function is identical to `void at::CUDAGeneratorImpl::capture_prologue()`.

- **uint64_t at_npu::NPUGeneratorImpl::capture_epilogue()**

    Ends graph capture for `NPUGeneratorImpl`. The return type is `uint64_t`, which disables graph capture and returns the reserved region size `offset_extragraph`. This function is identical to `uint64_t at::CUDAGeneratorImpl::capture_epilogue()`.

In PyTorch 2.5.1 and later, the following member functions are added:

- **void graphsafe_set_state(const c10::intrusive_ptr& state)**

    Sets the expected random number generator state for `aclgraph` during graph capture for `NPUGeneratorImpl`. This function is identical to `void at::CUDAGeneratorImpl::graphsafe_set_state(const c10::intrusive_ptr<c10::GeneratorImpl>& state)`.
    
    **`state`** (`c10::intrusive_ptr<c10::GeneratorImpl>`): Random number generator state.
- **c10::intrusive_ptr\<c10::GeneratorImpl> graphsafe_get_state()**

    Queries the random number generator object for `aclgraph` during graph capture for `NPUGeneratorImpl`. The return value is a `c10::GeneratorImpl` object. This function is identical to `c10::intrusive_ptr<c10::GeneratorImpl> at::CUDAGeneratorImpl::graphsafe_get_state()`.
    
    The return value is a `c10::GeneratorImpl` object.
- **void register_graph(c10_npu::NPUGraph* graph)**

    Registers an `aclgraph` object with `NPUGeneratorImpl` for unified management. This function is identical to `void at::CUDAGeneratorImpl::register_graph(CUDAGraph* graph)`.
- **void unregister_graph(c10_npu::NPUGraph* graph)**

    Removes an `aclgraph` object from `NPUGeneratorImpl`, which is called when the graph object is destroyed. This function is identical to `void at::CUDAGeneratorImpl::unregister_graph(CUDAGraph* graph)`.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>

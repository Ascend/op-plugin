# (beta) struct c10_npu::NPUEvent

## Definition File

torch_npu\csrc\core\npu\NPUEvent.h

## Function

Implements NPU event management functionality as an event class. It can be used to monitor device progress, accurately measure elapsed time, or synchronize NPU streams.

## Member Functions

- **c10_npu::NPUEvent::NPUEvent()**

    Default constructor for `NPUEvent`. This function is identical to `at::cuda::CUDAEvent::CUDAEvent()`.

- **c10_npu::NPUEvent::NPUEvent(unsigned int flags)**

    Constructs an `NPUEvent` with flags. This function is identical to `at::cuda::CUDAEvent::CUDAEvent(unsigned int flags)`.

    **`flags`** (`unsigned int`): Event type to be constructed.

- **c10_npu::NPUEvent::\~NPUEvent()**

    Destructor for `NPUEvent`. This function is identical to `at::cuda::CUDAEvent::~CUDAEvent()`.

- **c10_npu::NPUEvent::NPUEvent(c10_npu::NPUEvent&& other)**

    Move constructor for `NPUEvent`. This function is identical to `at::cuda::CUDAEvent::CUDAEvent(at::cuda::CUDAEvent&& other)`.

    `other` (`NPUEvent`): Object used to move-construct a new `NPUEvent` object.

- **c10_npu::NPUEvent::operator aclrtEvent()**

    Performs type conversion from `NPUEvent` to `aclrtEvent`. This function is identical to `at::cuda::CUDAEvent::operator cudaEvent_t()`.

- **c10::optional\<at::Device> c10_npu::NPUEvent::device()**

    Obtains the device type for `NPUEvent`. This function is identical to `c10::optional<at::Device> at::cuda::CUDAEvent::device()`.

- **bool c10_npu::NPUEvent::isCreated()**

    Queries whether the event is created for `NPUEvent`. The return type is `bool`. Valid values are `True` (the event is created) or `False` (the event is not created). This function is identical to `bool at::cuda::CUDAEvent::isCreated()`.

- **c10::DeviceIndex c10_npu::NPUEvent::device_index()**

    Obtains the device ID for `NPUEvent`. This function is identical to `c10::DeviceIndex at::cuda::CUDAEvent::device_index()`.

- **aclrtEvent c10_npu::NPUEvent::event()**

    Queries the underlying ACL event for `NPUEvent`. The return type is `aclrtEvent`. This function is identical to `cudaEvent_t at::cuda::CUDAEvent::event()`.

- **bool c10_npu::NPUEvent::query()**

    Queries whether the event is complete for `NPUEvent`. The return type is `bool`. Valid values are `True` (all submitted work is complete) or `False` (the work is not complete). This function is identical to `bool at::cuda::CUDAEvent::query()`.

- **void c10_npu::NPUEvent::record()**

    Records the event for `NPUEvent`, which records the event on the current stream. This function is identical to `void at::cuda::CUDAEvent::record()`.

- **void c10_npu::NPUEvent::record(const c10_npu::NPUStream& stream)**

    Records the event for `NPUEvent`, which records the event on a specified stream. This function is identical to `void at::cuda::CUDAEvent::record(const c10::cuda::CUDAStream& stream)`.

    **`stream`** (`NPUStream`): Stream on which the event is recorded.

- **void c10_npu::NPUEvent::recordOnce(const c10_npu::NPUStream& stream)**

    Records the event for `NPUEvent`. Given a specified stream, the event is recorded once if it has not been recorded previously. This function is identical to `void at::cuda::CUDAEvent::recordOnce(const c10::cuda::CUDAStream& stream)`.

    **`stream`** (`NPUStream`): Stream on which the event is recorded.

- **void c10_npu::NPUEvent::block(const c10_npu::NPUStream& stream)**

    Blocks a specified event for `NPUEvent`. This function is identical to `void at::cuda::CUDAEvent::block(const c10::cuda::CUDAStream& stream)`.

    **`stream`** (`NPUStream`): Stream to block.

- **float c10_npu::NPUEvent::elapsed_time(const c10_npu::NPUEvent& other)**

    Blocks the event for `NPUEvent`. The return type is `float`, which represents the elapsed time in milliseconds from the recorded event to the current event. This function is identical to `float at::cuda::CUDAEvent::elapsed_time(const at::cuda::CUDAEvent& other)`.

    **`other`** (`NPUEvent`): Destination event used to compute the elapsed time.

- **void c10_npu::NPUEvent::synchronize()**

    Synchronizes the event for `NPUEvent`, which blocks execution until the event completes execution. This function is identical to `void at::cuda::CUDAEvent::synchronize()`.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>

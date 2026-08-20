# (beta) class c10_npu::NPUStream

## Definition File

torch_npu\csrc\core\npu\NPUStream.h

## Function

Implements NPU stream management functionality as an NPU stream class. An NPU stream is a linear execution sequence belonging to an NPU device.

## Member Functions

- **c10_npu::NPUStream::NPUStream(c10::Stream stream)**

    `NPUStream` constructor, which constructs an NPU stream from a specified stream. This function is identical to `c10::cuda::CUDAStream::CUDAStream(c10::Stream stream)`.

    **`stream`** (`Stream`): Specified input stream, which must be an NPU stream.

- **c10_npu::NPUStream::NPUStream(Unchecked, c10::Stream stream)**

    `NPUStream` constructor, which constructs an NPU stream from a specified stream without checking whether it is an NPU stream. This function is identical to `c10::cuda::CUDAStream::CUDAStream(Unchecked, c10::Stream stream)`.

    **`stream`** (`Stream`): Specified input stream.

- **c10_npu::NPUStream::\~NPUStream()**

    `NPUStream` destructor, which is identical to `c10::cuda::CUDAStream::~CUDAStream()`.

- **bool c10_npu::NPUStream::operator==(const c10_npu::NPUStream& other)**

    Overloads the `==` operator for `NPUStream`. Returns `True` if the two streams are equal after conversion to `c10::Stream`. This function is identical to `bool c10::cuda::CUDAStream::operator==(const c10::cuda::CUDAStream& other)`.

    **`other`** (`NPUStream`): NPU stream to be compared.

- **bool c10_npu::NPUStream::operator!=(const c10_npu::NPUStream& other)**

    Overloads the `!=` operator for `NPUStream`. Returns `True` if the two streams are not equal after conversion to `c10::Stream`. This function is identical to `bool c10::cuda::CUDAStream::operator!=(const c10::cuda::CUDAStream& other)`.

    **`other`** (`NPUStream`): NPU stream to be compared.

- **c10_npu::NPUStream::operator aclrtStream()**

    Performs `aclrtStream` type conversion for `NPUStream`. This function is identical to `c10::cuda::CUDAStream::operator cudaStream_t()`.

- **c10_npu::NPUStream::operator c10::Stream()**

    Performs `c10::Stream` type conversion for `NPUStream`. This function is identical to `c10::cuda::CUDAStream::operator c10::Stream()`.

- **c10::DeviceType c10_npu::NPUStream::device_type()**

    Obtains the device type for `NPUStream`. The return type is `DeviceType`. This function is identical to `c10::DeviceType c10::cuda::CUDAStream::device_type()`.

- **c10::DeviceIndex c10_npu::NPUStream::device_index()**

    Obtains the device ID for `NPUStream`. The return type is `DeviceIndex`. This function is identical to `c10::DeviceIndex c10::cuda::CUDAStream::device_index()`.

- **c10::Device c10_npu::NPUStream::device()**

    Obtains the device for `NPUStream`. The return type is `Device`, which is guaranteed to represent an NPU device. This function is identical to `c10::Device c10::cuda::CUDAStream::device()`.

- **c10::StreamId c10_npu::NPUStream::id()**

    Obtains the stream ID for `NPUStream`. The return type is `StreamId`. This function is identical to `c10::StreamId c10::cuda::CUDAStream::id()`.

- **bool c10_npu::NPUStream::query()**

    Queries whether the stream has completed for `NPUStream`. The return type is `bool`. A return value of `True` indicates that all submitted work has completed. This function is identical to `bool c10::cuda::CUDAStream::query()`.

- **void c10_npu::NPUStream::synchronize()**

    Synchronizes the stream for `NPUStream`, which blocks execution until all submitted work completes. This function is identical to `void c10::cuda::CUDAStream::synchronize()`.

- **aclrtStream c10_npu::NPUStream::stream()**

    Queries the `aclrtStream` stream for `NPUStream`. The return type is `aclrtStream`. This function is identical to `cudaStream_t c10::cuda::CUDAStream::stream()`.

- **c10::Stream c10_npu::NPUStream::unwrap()**

    Queries the `Stream` stream for `NPUStream`. The return type is `Stream`. This function is identical to `c10::Stream c10::cuda::CUDAStream::unwrap()`.

- **struct c10::StreamData3 c10_npu::NPUStream::pack3()**

    Packs the stream for `NPUStream`. The return type is `StreamData3`. This function is identical to `struct c10::StreamData3 c10::cuda::CUDAStream::pack3()`.

- **c10_npu::NPUStream c10_npu::NPUStream::unpack3(c10::StreamId stream_id, c10::DeviceIndex device_index, c10::DeviceType device_type)**

    Unpacks a stream from a `StreamData3` structure for `NPUStream`. The return type is `NPUStream`. This function is identical to `c10::cuda::CUDAStream c10::cuda::CUDAStream::unpack3(c10::StreamId stream_id, c10::DeviceIndex device_index, c10::DeviceType device_type)`.

    **`stream_id`** (`StreamId`): Stream ID stored in the `StreamData3` structure.

    **`device_index`** (`DeviceIndex`): Device ID stored in the `StreamData3` structure.

    **`device_type`** (`DeviceType`): Device type stored in the `StreamData3` structure.

- **void c10_npu::NPUStream::setDataPreprocessStream(bool is_data_preprocess_stream)**

    Sets `NPUStream` as a data preprocessing stream.

    **`is_data_preprocess_stream`** (`bool`): `True` specifies the stream as a data preprocessing stream.

- **bool c10_npu::NPUStream::isDataPreprocessStream()**

    Queries whether the stream is a data preprocessing stream for `NPUStream`. The return type is `bool`. A return value of `True` indicates that it is a data preprocessing stream.

- **aclrtStream c10_npu::NPUStream::stream(const bool need_empty)**

    Queries the `aclrtStream` stream for `NPUStream`, allowing the input parameter `need_empty`. The return type is `aclrtStream`.

    **`need_empty`** (`bool`): `False` specifies to return the current stream directly instead of an empty stream.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>

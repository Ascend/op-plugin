// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_utils = at_npu::native::NpuUtils;
using npu_compile_type = at_npu::native::CompileType;
using npu_preparation = at_npu::native::OpPreparation;

namespace {
const static int64_t LENGTH_DATA_ALIGN = 128;
const static int64_t LENGTH_BOLCK_ALIGN = 8;
const static int64_t INCREMENT = 10;
at::Tensor& dropout_do_mask(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& mask,
    at::Scalar prob) {
  at_npu::native::OpCommand cmd;
  cmd.Name("DropOutDoMask")
      .Input(self)
      .Input(mask)
      .Input(prob, self.scalar_type(), npu_compile_type::MEMORY_HOST_COMPILE_DEPENDENT)
      .Output(result)
      .Run();
  return result;
}

std::tuple<at::Tensor, at::Tensor> dropout_do_mask_npu(
    const at::Tensor& self,
    const at::Tensor& mask,
    double p) {
  at::Scalar prob = at::Scalar(1. - p);
  at::Tensor result = npu_preparation::apply_tensor(self);
  at_npu::native::OpCommand cmd;
  cmd.Name("DropOutDoMask")
      .Input(self)
      .Input(mask)
      .Input(prob, self.scalar_type(), npu_compile_type::MEMORY_HOST_COMPILE_DEPENDENT)
      .Output(result)
      .Run();
  return std::tie(result, mask);
}

at::Tensor dropout_gen_mask(const at::Tensor& self, at::Scalar prob) {
  bool is_not_jit_compile = at_npu::native::env::CheckJitDisable();
  auto desc_ = torch_npu::NPUBridge::GetNpuStorageImpl(self)->get_npu_desc();
  int64_t numels = is_not_jit_compile ? c10::multiply_integers(desc_.storage_sizes_) : self.numel();

  uint64_t length = (static_cast<uint64_t>(numels) + LENGTH_DATA_ALIGN - 1) / LENGTH_DATA_ALIGN * LENGTH_DATA_ALIGN;
  at::Tensor mask = npu_preparation::apply_tensor_with_format(
      {length / LENGTH_BOLCK_ALIGN},
      self.options().dtype(at::kByte),
      ACL_FORMAT_ND);

  at::IntArrayRef self_shape = is_not_jit_compile ? desc_.storage_sizes_ : self.sizes();

  at_npu::native::OpCommand cmd;
  // DropOutGenMask use seed and seed1 to generator a seed, like this:
  //  seed1   seed
  // 127~64   63~0
  // so, we set seed1 = 0 to ensure the seed which user set is equal to the seed
  // used by the operator DropOutGenMask
  const auto gen = at_npu::detail::getDefaultNPUGenerator();
  auto pair = at::check_generator<at_npu::NPUGeneratorImpl>(gen)->philox_engine_inputs(INCREMENT);
  // At present, the default value of random number may be very large,
  // which will cause overflow in graph mode, so we set seed = 0 to avoid it.
  const int64_t seed = static_cast<int64_t>(pair.first);
  const int64_t offset = static_cast<int64_t>(pair.second);
  at::SmallVector<int64_t, N> offset_list = {0, offset};

  const int64_t seed1 = 0;
  cmd.Name("StatelessDropOutGenMask")
      .Input(self_shape)
      .Input(prob, self.scalar_type(), npu_compile_type::MEMORY_HOST_COMPILE_DEPENDENT)
      .Input(at::Scalar(seed), at::ScalarType::Int)
      .Input(at::Scalar(seed1), at::ScalarType::Int)
      .Input(offset_list, at::kLong, npu_compile_type::MEMORY_HOST_COMPILE_INDEPENDENT)
      .Output(mask)
      .Run();
  return mask;
}

std::tuple<at::Tensor, at::Tensor> dropout_v1_out_nocheck(
    at::Tensor result,
    const at::Tensor& self,
    double p) {
  at::Tensor self_cp = npu_utils::format_contiguous(self);
  TORCH_CHECK(
      p >= 0 && p <= 1,
      "dropout probability has to be between 0 and 1, but got ",
      p, OPS_ERROR(ErrCode::VALUE));
  TORCH_CHECK(
      at::isFloatingType(self_cp.scalar_type()),
      "dropout only supports floating-point dtypes" + OPS_ERROR(ErrCode::TYPE));

  double retain = 1. - p;
  at::Scalar prob = at::Scalar(retain);
  at::Tensor mask;
  auto original_stream = c10_npu::getCurrentNPUStream();
  {
    // During the life cycle of this raii instance, the calcu stream is set as the
    // secondary stream, and tasks are distributed to the secondary stream. At the
    // same time, according to the one-stream-one-pool principle, memory is also
    // alloced from the pool of the secondary stream.
    c10_npu::SecondaryStreamGuard guard(c10_npu::getCurrentSecondaryStream());
    mask = dropout_gen_mask(self_cp, prob);
  }
  // When tasks on multiple streams read and write the same block of memory,
  // recordStream needs to be called to ensure the correctness of memory reuse.
  c10_npu::NPUCachingAllocator::recordStream(mask.storage().data_ptr(), original_stream);
  dropout_do_mask(result, self_cp, mask, prob);

  return std::tie(result, mask);
}
} // namespace

std::tuple<at::Tensor, at::Tensor> _npu_dropout(
    const at::Tensor& self,
    double p) {
  at::Tensor result = npu_preparation::apply_tensor(self);
  return dropout_v1_out_nocheck(result, self, p);
}

at::Tensor npu_dropout_gen_mask(
    at::IntArrayRef size, double p,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<c10::Layout> layout_opt,
    c10::optional<c10::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  c10::TensorOptions options = c10::TensorOptions().dtype(dtype_opt)
      .device(device_opt)
      .layout(layout_opt)
      .pinned_memory(pin_memory_opt);

  at::Scalar prob = at::Scalar(1. - p);
  int64_t numels = c10::multiply_integers(size);

  uint64_t length = (static_cast<uint64_t>(numels) + 128 - 1) / 128 * 128;
  at::Tensor mask = npu_preparation::apply_tensor_with_format(
      at::IntArrayRef{length / 8},
      options.dtype(at::kByte),
      ACL_FORMAT_ND);

  at_npu::native::OpCommand cmd;
  // If either seed or seed1 are set to be non-zero, the random number generator
  // is seeded by the given seed. Otherwise, it is seeded by a random seed.
  const auto gen = at_npu::detail::getDefaultNPUGenerator();
  auto pair = at::check_generator<at_npu::NPUGeneratorImpl>(gen)->philox_engine_inputs(10);
  // At present, the default value of random number may be very large,
  // which will cause overflow in graph mode, so we set seed = 0 to avoid it.
  const int64_t seed = static_cast<int64_t>(pair.first);
  const int64_t offset = static_cast<int64_t>(pair.second);
  at::SmallVector<int64_t, N> offset_list = {0, offset};

  const int64_t seed1 = 0;
  cmd.Name("StatelessDropOutGenMask")
      .Input(size)
      .Input(prob, c10::typeMetaToScalarType(options.dtype()), npu_compile_type::MEMORY_HOST_COMPILE_DEPENDENT)
      .Input(at::Scalar(seed), at::ScalarType::Int)
      .Input(at::Scalar(seed1), at::ScalarType::Int)
      .Input(offset_list, at::kLong, npu_compile_type::MEMORY_HOST_COMPILE_INDEPENDENT)
      .Output(mask)
      .Run();
  return mask;
}

at::Tensor npu_dropout_backward(
    const at::Tensor& grad_output,
    const at::Tensor& mask,
    double scale) {
  TORCH_CHECK(
      at::isFloatingType(grad_output.scalar_type()),
      "dropoutbackward only supports floating-point dtypes" + OPS_ERROR(ErrCode::TYPE));
  TORCH_CHECK(
      mask.scalar_type() == at::ScalarType::Byte,
      "mask should be torch.uint8 dtype" + OPS_ERROR(ErrCode::TYPE));
  double retain = 1. - scale;
  at::Tensor result = npu_preparation::apply_tensor(grad_output);

  at_npu::native::OpCommand cmd;
  cmd.Name("DropOutDoMask")
      .Input(grad_output)
      .Input(mask)
      .Input(at::Scalar(retain), grad_output.scalar_type(), npu_compile_type::MEMORY_HOST_COMPILE_DEPENDENT)
      .Output(result)
      .Run();

  return result;
}

std::tuple<at::Tensor, at::Tensor> npu_dropout_do_mask(
    const at::Tensor& self,
    const at::Tensor& mask,
    double p) {
  return dropout_do_mask_npu(self, mask, p);
}

at::Tensor dropout(const at::Tensor& self, double p, bool train) {
  if (p == 0 || !train || self.numel() == 0) {
    return self;
  }
  if (p == 1) {
    return self.mul(at::zeros(self.sizes(), self.options()));
  }
  auto results = at_npu::native::custom_ops::_npu_dropout(self, p);
  return std::get<0>(results);
}
} // namespace acl_op

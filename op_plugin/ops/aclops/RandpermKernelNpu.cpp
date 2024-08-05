// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& randperm_out_nocheck(at::Tensor& result, int64_t n, c10::optional<at::Generator> gen) {
  auto gen_val = at::get_generator_or_default<at_npu::NPUGeneratorImpl>(gen, at_npu::detail::getDefaultNPUGenerator());
  auto pair = gen_val->philox_engine_inputs(10);
  const int64_t seed = static_cast<int64_t>(pair.first);
  const int64_t offset = static_cast<int64_t>(pair.second);
  const int64_t layout = 1;
  at_npu::native::OpCommand cmd;
  cmd.Name("StatelessRandperm")
      .Input(at::Scalar(n), at::kLong)
      .Input(at::Scalar(seed), at::kLong)
      .Input(at::Scalar(offset), at::kLong)
      .Output(result)
      .Attr("layout", layout)
      .Attr("dtype", result.scalar_type())
      .Run();
  return result;
}
} // namespace

at::Tensor& randperm_out(int64_t n, c10::optional<at::Generator> generator, at::Tensor& result) {
    TORCH_CHECK(n >= 0, "n must be non-negative, got", n, OPS_ERROR(ErrCode::VALUE));
    npu_preparation::CheckOut({}, result, result, {n});
    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        randperm_out_nocheck(contiguous_result, n, generator);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        randperm_out_nocheck(result, n, generator);
    }
    return result;
}

at::Tensor& randperm_out(int64_t n, at::Tensor& result) {
  return acl_op::randperm_out(n, static_cast<c10::optional<at::Generator>>(c10::nullopt), result);
}

at::Tensor randperm(
    int64_t n,
    c10::optional<at::Generator> generator,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory) {
    TORCH_CHECK(n >= 0, "n must be non-negative, got", n, OPS_ERROR(ErrCode::VALUE));
    at::TensorOptions options = c10::TensorOptions()
        .dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);
    at::Tensor result = npu_preparation::apply_tensor_with_format(
        {n},
        options,
        ACL_FORMAT_ND);
    randperm_out_nocheck(result, n, generator);
    return result;
}

at::Tensor randperm(
    int64_t n,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory) {
  return acl_op::randperm(n, static_cast<c10::optional<at::Generator>>(c10::nullopt), dtype, layout, device, pin_memory);
}
} // namespace acl_op

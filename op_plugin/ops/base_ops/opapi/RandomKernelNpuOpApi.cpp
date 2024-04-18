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
#include "op_plugin/OpApiInterface.h"
#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {

const int64_t RANDOM_DOUBLE_MAX = 1LL << 53;
const int64_t RANDOM_HALF_MAX = 1LL << 11;
const int64_t RANDOM_FLOAT_MAX = 1LL << 24;
const int64_t RANDOM_BFLOAT16_MAX = 1LL << 8;

}  // namespace
static std::map<at::ScalarType, int64_t> DTYPE_MAX_VALUE_MAP = {
  {at::kHalf, RANDOM_HALF_MAX + 1},
  {at::kFloat, RANDOM_FLOAT_MAX + 1},
  {at::kDouble, RANDOM_DOUBLE_MAX + 1},
  {at::kInt, std::numeric_limits<int>::max()},
  {at::kShort, std::numeric_limits<int16_t>::max()},
  {at::kChar, std::numeric_limits<int8_t>::max()},
  {at::kByte, std::numeric_limits<uint8_t>::max()},
  {at::kLong, std::numeric_limits<long>::max()},
  {at::kBFloat16, RANDOM_BFLOAT16_MAX + 1},
  {at::kBool, 1}
};

int64_t get_dtype_max_value(at::ScalarType dtype)
{
    auto iter = DTYPE_MAX_VALUE_MAP.find(dtype);
    TORCH_CHECK(iter != DTYPE_MAX_VALUE_MAP.end(), "self scalar_type:", dtype, "is not surpported.", OPS_ERROR(ErrCode::TYPE));
    return iter->second;
}

at::Tensor& random_op_api_(at::Tensor& self, int64_t from, int64_t to, c10::optional<at::Generator> gen_)
{
    int64_t delta = to - from;
    auto output_size = op_infer::input_same_output_size(self);
    at::Tensor cache_base = npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(at::ScalarType::Float));
    cache_base = op_api::uniform_(cache_base, 0.0, 1.0, gen_);
    cache_base = op_api::mul_(cache_base, at::Scalar(delta));
    at::Tensor cache = op_api::npu_dtype_cast(cache_base, self.scalar_type());
    self = op_api::add_(op_api::mul_(self, at::Scalar(0)), at::Scalar(from), at::Scalar(1));
    return op_api::add_(self, cache, at::Scalar(1));
}

at::Tensor& random_(at::Tensor& self, int64_t from, c10::optional<int64_t> to, c10::optional<at::Generator> gen_)
{
    DO_COMPATIBILITY(aclnnInplaceRandom, acl_op::random_(self, from, to, gen_));
    int64_t to_ = to.value_or(get_dtype_max_value(self.scalar_type()));
    random_op_api_(self, from, to_, gen_);
    return self;
}

at::Tensor& random_(at::Tensor& self, int64_t to, c10::optional<at::Generator> gen_)
{
    DO_COMPATIBILITY(aclnnInplaceRandom, acl_op::random_(self, to, gen_));
    int64_t from = 0;
    random_op_api_(self, from, to, gen_);
    return self;
}

at::Tensor& random_(at::Tensor& self, c10::optional<at::Generator> gen_)
{
    DO_COMPATIBILITY(aclnnInplaceRandom, acl_op::random_(self, gen_));
    int64_t from = 0;
    int64_t to = get_dtype_max_value(self.scalar_type());
    random_op_api_(self, from, to, gen_);
    return self;
}
}  // namespace op_api

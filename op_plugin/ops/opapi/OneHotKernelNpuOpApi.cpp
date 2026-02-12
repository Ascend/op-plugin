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
#include "op_plugin/utils/op_api_common.h"
#if VERSION_BETWEEN(V2R9, VERSION_NEWEST)
#include <ATen/DTensorState.h>
#endif

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

static const int64_t MIN_DEPTH = 1;
static const int64_t AUTO_DEPTH = -1;
static const int64_t MIN_NUM_CLASSES = 0;

at::Tensor one_hot(const at::Tensor& self, int64_t num_classes)
{
    auto ks = self.key_set();
    bool is_fake_or_meta = ks.has_all(c10::DispatchKeySet(c10::BackendComponent::MetaBit)) ||
        ks.has_all(c10::DispatchKeySet(c10::DispatchKey::Python)) ||
        self.is_meta();
    if (is_fake_or_meta) {
        TORCH_CHECK(num_classes != -1, "FakeTensorMode does not support num_classes == -1.");

#if VERSION_BETWEEN(V2R9, VERSION_NEWEST)
        at::DTensorAllowImplicitReplication guard;
#endif

        auto options = self.options().dtype(at::kLong);
        at::Tensor index = at::arange(num_classes, options);
        return at::eq(self.unsqueeze(-1), index).to(at::kLong);
    }
    DO_COMPATIBILITY(aclnnOneHot, acl_op::one_hot(self, num_classes));
    int64_t depth = num_classes;
    TORCH_CHECK(depth >= AUTO_DEPTH, "NPU error, not yet support negative num_classes, when num_classes less than -1",
                OPS_ERROR(ErrCode::PARAM));
    // when the self is empty, num_classes should be greater than 0
    TORCH_CHECK(self.numel() != 0 || num_classes > MIN_NUM_CLASSES,
                "NPU error, can not infer total number of classes from empty tensor.", OPS_ERROR(ErrCode::PARAM));
    if (depth == AUTO_DEPTH) {
        depth = self.max().item().toLong() + 1;
        if (depth < MIN_DEPTH) {
            depth = MIN_DEPTH;
        }
    }
    // construct on_value tensor
    at::Tensor on_value_tensor = npu_preparation::apply_tensor_without_format({1}, self.options());
    on_value_tensor.fill_(1);
    // construct off_value tensor
    at::Tensor off_value_tensor = npu_preparation::apply_tensor_without_format({1}, self.options());
    off_value_tensor.fill_(0);
    auto output_size = op_infer::array_to_small_vector(self.sizes());
    output_size.emplace_back(depth);
    // construct the output tensor of the NPU
    at::Tensor result = npu_preparation::apply_tensor(output_size, self.options(), self);
    int64_t axis = -1;
    EXEC_NPU_CMD(aclnnOneHot, self, depth, on_value_tensor, off_value_tensor, axis, result);
    return result;
}

at::Tensor npu_one_hot(
    const at::Tensor& self,
    int64_t num_classes,
    int64_t depth,
    const at::Scalar& on_value,
    const at::Scalar& off_value)
{
    if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend950) {
        return acl_op::npu_one_hot(self, num_classes, depth, on_value, off_value);
    }

    auto ks = self.key_set();
    bool is_fake_or_meta = ks.has_all(c10::DispatchKeySet(c10::BackendComponent::MetaBit)) ||
        ks.has_all(c10::DispatchKeySet(c10::DispatchKey::Python)) ||
        self.is_meta();
    if (is_fake_or_meta) {
        TORCH_CHECK(depth != -1, "FakeTensorMode does not support depth == -1.");

#if VERSION_BETWEEN(V2R9, VERSION_NEWEST)
        at::DTensorAllowImplicitReplication guard;
#endif

        auto options = self.options().dtype(at::kLong);
        at::Tensor index = at::arange(depth, options);
        return at::eq(self.unsqueeze(-1), index).to(at::kLong);
    }

    DO_COMPATIBILITY(aclnnOneHot, acl_op::npu_one_hot(self, num_classes, depth, on_value, off_value));

    TORCH_CHECK(depth >= AUTO_DEPTH, "NPU error, not yet support negative depth, when depth less than -1",
                OPS_ERROR(ErrCode::PARAM));
    // when the self is empty, num_classes should be greater than 0
    TORCH_CHECK(self.numel() != 0 || depth > MIN_NUM_CLASSES,
                "NPU error, can not infer total number of classes from empty tensor.", OPS_ERROR(ErrCode::PARAM));
    if (depth == AUTO_DEPTH) {
        depth = self.max().item().toLong() + 1;
        if (depth < MIN_DEPTH) {
            depth = MIN_DEPTH;
        }
    }
    // construct on_value tensor
    at::Tensor on_value_tensor = npu_preparation::apply_tensor_without_format({1}, self.options());
    op_api::fill_(on_value_tensor, on_value);
    // construct off_value tensor
    at::Tensor off_value_tensor = npu_preparation::apply_tensor_without_format({1}, self.options());
    op_api::fill_(off_value_tensor, off_value);
    auto output_size = op_infer::array_to_small_vector(self.sizes());
    int64_t max_num_classes = static_cast<int64_t>(output_size.size());
    TORCH_CHECK(num_classes >= AUTO_DEPTH,
                "NPU error: num_classes cannot be less than -1",
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(num_classes < max_num_classes,
                "NPU error: num_classes must be less than ", max_num_classes,
                OPS_ERROR(ErrCode::PARAM));
    int64_t axis = (num_classes + max_num_classes) % max_num_classes;

    c10::SmallVector<int64_t, SIZE> output_shape;
    for (int64_t i = 0; i < axis; i++) {
        output_shape.emplace_back(self.size(i));
    }
    output_shape.emplace_back(depth);
    for (int64_t i = axis+1; i < max_num_classes; i++) {
        output_shape.emplace_back(self.size(i));
    }

    // construct the output tensor of the NPU
    at::Tensor result = npu_preparation::apply_tensor(output_shape, self.options(), self);

    EXEC_NPU_CMD(aclnnOneHot, self, depth, on_value_tensor, off_value_tensor, axis, result);
    return result;
}
}

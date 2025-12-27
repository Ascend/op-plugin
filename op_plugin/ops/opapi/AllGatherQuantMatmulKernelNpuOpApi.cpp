// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/framework/utils/InternalFormatOpAdapter.h"

namespace op_api {
const int32_t MIN_SUPPORT_WORLD_SIZE = 2;
const int32_t MAX_SUPPORT_WORLD_SIZE = 64;
using npu_preparation = at_npu::native::OpPreparation;
static c10::SmallVector<int64_t, op_infer::SIZE> get_y_size(const at::Tensor& x1, const at::Tensor& x2,
                                                            int64_t world_size, int64_t gather_index)
{
    auto out_x = gather_index == 0 ? x1.size(0) * world_size : x1.size(0);
    auto out_y = x2.size(1);
    return {out_x, out_y};
}

static c10::SmallVector<int64_t, op_infer::SIZE> get_gather_out_size(const at::Tensor& x1, const at::Tensor& x2,
                                                                     int64_t world_size, int64_t gather_index)
{
    const at::Tensor& gather_out = gather_index == 0 ? x1 : x2;
    return {gather_out.size(0) * world_size, gather_out.size(1)};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_all_gather_quant_mm(
    const at::Tensor& self, const at::Tensor& x2, c10::string_view hcom, int64_t world_size,
    const c10::optional<at::Tensor>& bias, const c10::optional<at::Tensor>& x1_scale,
    const c10::optional<at::Tensor>& x2_scale, const c10::optional<at::Tensor>& quant_scale, int64_t block_size,
    int64_t gather_index, bool gather_output, int64_t comm_turn, c10::OptionalIntArrayRef group_sizes,
    bool amax_output, c10::optional<int64_t> y_dtype, c10::optional<int64_t> x1_dtype, c10::optional<int64_t> x2_dtype,
    c10::optional<int64_t> x1_scale_dtype, c10::optional<int64_t> x2_scale_dtype)
{
    TORCH_CHECK(world_size >= MIN_SUPPORT_WORLD_SIZE && world_size <= MAX_SUPPORT_WORLD_SIZE &&
                (world_size & (world_size - 1)) == 0,
                "world_size should be in [2, 4, 8, 16, 32, 64], but actual value is ", world_size,
                OPS_ERROR(ErrCode::PARAM));
    at::IntArrayRef group_size_list = group_sizes.value_or(at::IntArrayRef{});
    int64_t group_size = op_plugin::utils::check_and_get_group_size(group_size_list);
    TORCH_CHECK(group_size != -1, "Invalid group_sizes.", OPS_ERROR(ErrCode::PARAM));
    int64_t stream_mode = 1;
    const char* hcom_value = (char*)hcom.data();
    const at::Tensor& bias_value = bias.value_or(at::Tensor());
    const at::Tensor& x1_scale_value = x1_scale.value_or(at::Tensor());
    const at::Tensor& x2_scale_value = x2_scale.value_or(at::Tensor());
    const at::Tensor& quant_scale_value = quant_scale.value_or(at::Tensor());
    c10::SmallVector<int64_t, op_infer::SIZE> y_size = get_y_size(self, x2, world_size, gather_index);
    auto gather_out_size = gather_output ? get_gather_out_size(self, x2, world_size, gather_index)
                                         : c10::SmallVector<int64_t, op_infer::SIZE>{0};
    auto amax_size =
        amax_output ? c10::SmallVector<int64_t, op_infer::SIZE>{1} : c10::SmallVector<int64_t, op_infer::SIZE>{0};
    auto gather_out_dtype = gather_index == 0 ? x1_dtype : x2_dtype;
    auto gather_out_scalar_type = gather_index == 0 ? self.scalar_type() : x2.scalar_type();
    auto amax_dtype = at::kFloat;
    auto y_scalar_type = gather_index == 0 ? self.scalar_type() : x2.scalar_type();
    if (y_scalar_type == at::kBFloat16 || y_scalar_type == at::kHalf) {
        if (y_dtype.has_value()) {
            y_scalar_type = npu_preparation::convert_to_scalar_type(c10_npu::GetAclDataType(y_dtype.value()));
            std::string selfType = c10::getDtypeNames(self.scalar_type()).first;
            std::string yType = c10::getDtypeNames(y_scalar_type).first;
            TORCH_CHECK(y_scalar_type == self.scalar_type(), "When input is float16 or bfloat16, output should",
                        "be the same as input dtype. Expected output dtype:", selfType,
                        ", but got:", yType,
                        OPS_ERROR(ErrCode::PARAM));
        }
    } else {
        TORCH_CHECK(y_dtype.has_value(), "y_dtype should be provided when input dtype is not bf16 or fp16",
                    OPS_ERROR(ErrCode::PARAM));
        y_scalar_type = npu_preparation::convert_to_scalar_type(c10_npu::GetAclDataType(y_dtype.value()));
    }
    at::Tensor y =
        npu_preparation::apply_tensor_without_format(y_size, self.options().dtype(y_scalar_type));
    at::Tensor gather_out = gather_output ? npu_preparation::apply_tensor_without_format(
        gather_out_size, self.options().dtype(gather_out_scalar_type)) : at::empty({0}, self.options());
    at::Tensor amax = amax_output ? npu_preparation::apply_tensor_without_format(
        amax_size, self.options().dtype(amax_dtype)) : at::Tensor();
    TensorWrapper x1_wrapper = {
        self, (x1_dtype.has_value()) ? c10_npu::GetAclDataType(x1_dtype.value())
                                     : npu_preparation::convert_to_acl_data_type(self.scalar_type())};
    TensorWrapper x2_wrapper = {x2, (x2_dtype.has_value())
                                        ? c10_npu::GetAclDataType(x2_dtype.value())
                                        : npu_preparation::convert_to_acl_data_type(x2.scalar_type())};
    TensorWrapper gather_out_wrapper = {
        gather_out, (gather_out_dtype.has_value())
                        ? c10_npu::GetAclDataType(gather_out_dtype.value())
                        : npu_preparation::convert_to_acl_data_type(gather_out_scalar_type)};
    auto x1_scale_scalar_dtype = x1_scale.has_value() ? x1_scale.value().scalar_type() : at::kFloat;
    auto x2_scale_scalar_dtype = x2_scale.has_value() ? x2_scale.value().scalar_type() : at::kFloat;
    TensorWrapper x1_scale_wrapper = {x1_scale.value_or(at::Tensor()),
                                      (x1_scale_dtype.has_value())
                                          ? c10_npu::GetAclDataType(x1_scale_dtype.value())
                                          : npu_preparation::convert_to_acl_data_type(x1_scale_scalar_dtype)};
    TensorWrapper x2_scale_wrapper = {x2_scale.value_or(at::Tensor()),
                                      (x2_scale_dtype.has_value())
                                          ? c10_npu::GetAclDataType(x2_scale_dtype.value())
                                          : npu_preparation::convert_to_acl_data_type(x2_scale_scalar_dtype)};
    const char* comm_mode = "ccu";
    EXEC_NPU_CMD(aclnnAllGatherMatmulV2, x1_wrapper, x2_wrapper, bias_value, x1_scale_wrapper, x2_scale_wrapper,
                 quant_scale_value, block_size, hcom_value, gather_index, comm_turn, stream_mode, group_size, comm_mode,
                 y, gather_out_wrapper, amax);
    return std::tie(y, gather_out, amax);
}
}  // namespace op_api

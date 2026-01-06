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

#include <set>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/framework/utils/InternalFormatOpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
const std::set<int> SUPPORT_WORLD_SIZE_LIST{2, 4, 8, 16, 32, 64};
static const int DIM_TWO = 2;
std::tuple<at::Tensor, at::Tensor> npu_quant_mm_reduce_scatter(
    const at::Tensor& self, const at::Tensor& x2, c10::string_view hcom, int64_t world_size, c10::string_view reduce_op,
    const c10::optional<at::Tensor>& bias, const c10::optional<at::Tensor>& x1_scale,
    const c10::optional<at::Tensor>& x2_scale, const c10::optional<at::Tensor>& quant_scale, int64_t block_size,
    int64_t comm_turn, c10::OptionalIntArrayRef group_sizes, bool amax_output, c10::optional<int64_t> y_dtype,
    c10::optional<int64_t> x1_dtype, c10::optional<int64_t> x2_dtype, c10::optional<int64_t> x1_scale_dtype,
    c10::optional<int64_t> x2_scale_dtype)
{
    TORCH_CHECK(SUPPORT_WORLD_SIZE_LIST.find(world_size) != SUPPORT_WORLD_SIZE_LIST.end(),
                "world_size should be in [2, 4, 8, 16, 32, 64], but the actual value is ", world_size,
                OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(self.dim() == DIM_TWO && x2.dim() == DIM_TWO,
                "Both inputs of mm are required to be 2D, but the actual inputs are ", self.dim(), "D and ", x2.dim(),
                "D", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(self.size(1) == x2.size(0),
                "The K-axis in the two inputs of Matmul must be equal, but in reality, the K-axis of x1 is ",
                self.size(1), " and the K-axis of x2 is ", x2.size(0), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(world_size != 0, "world_size cannot be zero", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(self.size(0) % world_size == 0, "The M-axis in input of Matmul should be be divisible by world_size",
                OPS_ERROR(ErrCode::PARAM));
    at::IntArrayRef group_size_list = group_sizes.value_or(at::IntArrayRef{});
    int64_t group_size = op_plugin::utils::check_and_get_group_size(group_size_list);
    TORCH_CHECK(group_size != -1, "Invalid group_sizes.", OPS_ERROR(ErrCode::PARAM));
    auto output_size = {self.size(0) / (world_size != 0 ? world_size : 1), x2.size(1)};
    auto output_scalar_type = self.scalar_type();
    bool is_fp16_or_bf16 = ((output_scalar_type == at::kBFloat16) || (output_scalar_type == at::kHalf));
    if (is_fp16_or_bf16) {
        if (y_dtype.has_value()) {
            auto y_scalar_type = npu_preparation::convert_to_scalar_type(c10_npu::GetAclDataType(y_dtype.value()));
            std::string selfType = c10::getDtypeNames(self.scalar_type()).first;
            std::string yType = c10::getDtypeNames(y_scalar_type).first;
            TORCH_CHECK(y_scalar_type == self.scalar_type(), "When input is float16 or bfloat16, output should ",
                        "be the same as input dtype. Expected output dtype:", selfType,
                        ", but got:", yType,
                        OPS_ERROR(ErrCode::PARAM));
        }
    } else {
        TORCH_CHECK(y_dtype.has_value(), "input dtype is not bf16 or fp16, but no input y_dtype",
                    OPS_ERROR(ErrCode::PARAM));
        auto output_acltype = c10_npu::GetAclDataType(y_dtype.value());
        output_scalar_type = npu_preparation::convert_to_scalar_type(output_acltype);
    }
    c10::TensorOptions options = self.options().dtype(output_scalar_type);
    auto result = npu_preparation::apply_tensor_without_format(output_size, options);
    char* reduce_op_ptr = const_cast<char*>(reduce_op.data());
    char* hcom_ptr = const_cast<char*>(hcom.data());
    const at::Tensor& bias_real = bias.value_or(at::Tensor());
    const at::Tensor& quant_scale_real = quant_scale.value_or(at::Tensor());
    int64_t stream_mode = ACL_STOP_ON_FAILURE;
    auto amax_output_result = at::Tensor();
    if (amax_output) {
        amax_output_result = npu_preparation::apply_tensor_without_format({1}, self.options().dtype(at::kFloat));
    }
    TensorWrapper x1_wrapper = {self, (x1_dtype.has_value())
                                          ? c10_npu::GetAclDataType(x1_dtype.value())
                                          : npu_preparation::convert_to_acl_data_type(self.scalar_type())};
    TensorWrapper x2_wrapper = {x2, (x2_dtype.has_value())
                                        ? c10_npu::GetAclDataType(x2_dtype.value())
                                        : npu_preparation::convert_to_acl_data_type(x2.scalar_type())};
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
    EXEC_NPU_CMD(aclnnMatmulReduceScatterV2, x1_wrapper, x2_wrapper, bias_real, x1_scale_wrapper, x2_scale_wrapper,
                 quant_scale_real, block_size, hcom_ptr, reduce_op_ptr, comm_turn, stream_mode, group_size, comm_mode,
                 result, amax_output_result);

    FLOP_COUNT(FlopCounter::mm_flop, self, x2);
    return std::tie(result, amax_output_result);
}
}  // namespace op_api

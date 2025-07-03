// Copyright (c) 2024 Huawei Technologies Co., Ltd
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

namespace op_api {
const static int64_t IN_NOT_SPLIT_OUT_NOT_SPLIT = 0;
const static int64_t IN_SPLIT_OUT_NOT_SPLIT = 1;
const static int64_t IN_NOT_SPLIT_OUT_SPLIT = 2;
const static int64_t IN_SPLIT_OUT_SPLIT = 3;
const static int64_t INT4_NUMS_IN_INT32 = 8;
using npu_preparation = at_npu::native::OpPreparation;

void check_dims(int64_t split_item, size_t num_x, size_t num_weight, size_t num_group_list)
{
    TORCH_CHECK(num_x > 0 && num_weight > 0,
                "Invalid inputs: neither x nor weight could be empty." + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(split_item == IN_NOT_SPLIT_OUT_NOT_SPLIT || split_item == IN_SPLIT_OUT_NOT_SPLIT ||
                split_item == IN_NOT_SPLIT_OUT_SPLIT || split_item == IN_SPLIT_OUT_SPLIT,
                "Invalid value of split_item [", split_item, "], which should only be one of 0/1/2/3."
                + OPS_ERROR(ErrCode::PARAM));
    if (split_item == IN_NOT_SPLIT_OUT_NOT_SPLIT || split_item == IN_SPLIT_OUT_NOT_SPLIT) {
        if (num_group_list > 0) {
            TORCH_CHECK(num_x == 1 && num_weight == num_group_list, "Invalid inputs. "
                        "When split_item = 0 or 1 and input group_list is not None, "
                        "the following two conditions are supposed to be satisfied: "
                        "(1) length of x equals 1; (2) length of weight equals that of group_list. "
                        "Actual lengths: x [", num_x, "], weight [", num_weight, "], "
                        "group_list [", num_group_list, "]." + OPS_ERROR(ErrCode::PARAM));
        } else {
            TORCH_CHECK(num_x == num_weight,
                        "When split_item = 0 or 1 and input group_list is None, "
                        "the num of x tensors must equal the num of weight tensors."
                        "Actual lengths: x [", num_x, "], weight [", num_weight, "]." + OPS_ERROR(ErrCode::PARAM));
        }
    }
}

void create_new_tensor_multi_dim(std::vector<at::Tensor> &y, const at::Tensor &x_i, size_t n,
                                 c10::TensorOptions options)
{
    auto x_sizes = x_i.sizes();
    std::vector<int64_t> y_sizes(x_sizes.begin(), x_sizes.end());
    y_sizes.at(x_sizes.size() - 1) = static_cast<int64_t>(n);

    auto output_size = op_infer::array_to_small_vector(y_sizes);
    y.emplace_back(npu_preparation::apply_tensor_without_format(output_size, options));
}

void create_new_tensor(std::vector<at::Tensor> &y, size_t dim_m, size_t dim_n, c10::TensorOptions options)
{
    auto output_size = op_infer::array_to_small_vector({dim_m, dim_n});
    y.emplace_back(npu_preparation::apply_tensor_without_format(output_size, options));
}

void calculate_dim_m(size_t& dim_m, size_t num_x, const at::TensorList x)
{
    for (size_t i = 0; i < num_x; i++) {
        dim_m += x[i].sizes()[0];
    }
}

#if VERSION_BETWEEN(V1R11, V1R11)
// Motivation for adapting this interface for each Torch version separately:
// 1. Optional TensorList is only supported in Torch2.1 and later versions.
//    Thus, "Tensor[] bias" is used in Torch1.11 and Torch2.0, while
//    "Tensor[]? bias=None" is used in Torch2.1 and later versions.
// 2. Even if "Int[]? group_list=None" is used for all Torch versions, the
//    auto-generated data type for optional IntList group_list in Torch1.11
//    is different from those in Torch2.0 and later versions.
std::vector<at::Tensor> npu_grouped_matmul(const at::TensorList x,
                                           const at::TensorList weight,
                                           const at::TensorList bias,
                                           const at::TensorList scale,
                                           const at::TensorList offset,
                                           const at::TensorList antiquant_scale,
                                           const at::TensorList antiquant_offset,
                                           c10::optional<at::IntArrayRef> group_list,
                                           c10::optional<int64_t> split_item,
                                           c10::optional<at::ScalarType> output_dtype)
{
    auto num_x = x.size();
    auto num_weight = weight.size();
    auto group_list_real = group_list.value_or(at::IntArrayRef{});
    auto num_group_list = group_list_real.size();
    int64_t split_item_value = split_item.value_or(0);
    check_dims(split_item_value, num_x, num_weight, num_group_list);

    std::vector<at::Tensor> y;
    c10::TensorOptions options = x[0].options().dtype(output_dtype.value_or(x[0].scalar_type()));

    if (IN_NOT_SPLIT_OUT_NOT_SPLIT == split_item_value || IN_SPLIT_OUT_NOT_SPLIT == split_item_value) {
        if (num_group_list > 0) {
            y.reserve(num_group_list);
            TORCH_CHECK(group_list_real[0] >= 0,
                "group_list[0] should be larger than or equal to 0, but now is ", group_list_real[0], "." +
                OPS_ERROR(ErrCode::VALUE));
            create_new_tensor(y, group_list_real[0], weight[0].sizes()[1], options);
            for (size_t i = 1; i < num_group_list; i++) {
                TORCH_CHECK(group_list_real[i] - group_list_real[i - 1] >= 0,
                    "group_list[", i, "] - group_list[", i - 1, "] should be larger than or equal to 0, but now is ",
                    group_list_real[i] - group_list_real[i - 1], "." + OPS_ERROR(ErrCode::VALUE));
                create_new_tensor(y, group_list_real[i] - group_list_real[i - 1], weight[i].sizes()[1], options);
            }
        } else {
            y.reserve(num_x);
            for (size_t i = 0; i < num_x; i++) {
                create_new_tensor_multi_dim(y, x[i], weight[i].size(1), options);
            }
        }  // 校验NO_SPLIT时为特殊场景（groupList为空）或num_x > 1
    } else if (IN_NOT_SPLIT_OUT_SPLIT == split_item_value || IN_SPLIT_OUT_SPLIT == split_item_value) {
        if (num_x > 1) {
            size_t dim_m = 0;
            for (size_t i = 0; i < num_x; i++) {
                dim_m += x[i].sizes()[0];
            }
            create_new_tensor(y, dim_m, weight[0].sizes()[1], options);
        } else if (num_x == 1) {
            create_new_tensor(y, x[0].sizes()[0], weight[0].sizes()[1], options);
        }
    }
    at::TensorList result = at::TensorList(y);

    EXEC_NPU_CMD(aclnnGroupedMatmul, x, weight, bias, scale, offset, antiquant_scale,
                 antiquant_offset, group_list_real, split_item_value, result);

    return y;
}
#endif

#if VERSION_BETWEEN(V2R0, V2R0)
// Motivation for adapting this interface for each Torch version separately:
// 1. Optional TensorList is only supported in Torch2.1 and later versions.
//    Thus, "Tensor[] bias" is used in Torch1.11 and Torch2.0, while
//    "Tensor[]? bias=None" is used in Torch2.1 and later versions.
// 2. Even if "Int[]? group_list=None" is used for all Torch versions, the
//    auto-generated data type for optional IntList group_list in Torch2.0
//    is different from those in Torch1.11, Torch2.1, and later versions.
std::vector<at::Tensor> npu_grouped_matmul(const at::TensorList x,
                                           const at::TensorList weight,
                                           const at::TensorList bias,
                                           const at::TensorList scale,
                                           const at::TensorList offset,
                                           const at::TensorList antiquant_scale,
                                           const at::TensorList antiquant_offset,
                                           at::OptionalIntArrayRef group_list,
                                           c10::optional<int64_t> split_item,
                                           c10::optional<at::ScalarType> output_dtype)
{
    auto num_x = x.size();
    auto num_weight = weight.size();
    auto group_list_real = group_list.value_or(at::IntArrayRef{});
    auto num_group_list = group_list_real.size();
    int64_t split_item_value = split_item.value_or(0);
    check_dims(split_item_value, num_x, num_weight, num_group_list);

    std::vector<at::Tensor> y;
    c10::TensorOptions options = x[0].options().dtype(output_dtype.value_or(x[0].scalar_type()));

    if (IN_NOT_SPLIT_OUT_NOT_SPLIT == split_item_value || IN_SPLIT_OUT_NOT_SPLIT == split_item_value) {
        if (num_group_list > 0) {
            y.reserve(num_group_list);
            create_new_tensor(y, group_list_real[0], weight[0].sizes()[1], options);
            for (int i = 1; i < num_group_list; i++) {
                create_new_tensor(y, group_list_real[i] - group_list_real[i - 1], weight[i].sizes()[1], options);
            }
        } else {
            y.reserve(num_x);
            for (int i = 0; i < num_x; i++) {
                create_new_tensor_multi_dim(y, x[i], weight[i].size(1), options);
            }
        }  // 校验NO_SPLIT时为特殊场景（groupList为空）或num_x > 1
    } else if (IN_NOT_SPLIT_OUT_SPLIT == split_item_value || IN_SPLIT_OUT_SPLIT == split_item_value) {
        if (num_x > 1) {
            size_t dim_m = 0;
            for (int i = 0; i < num_x; i++) {
                dim_m += x[i].sizes()[0];
            }
            create_new_tensor(y, dim_m, weight[0].sizes()[1], options);
        } else if (num_x == 1) {
            create_new_tensor(y, x[0].sizes()[0], weight[0].sizes()[1], options);
        }
    }
    at::TensorList result = at::TensorList(y);

    EXEC_NPU_CMD(aclnnGroupedMatmul, x, weight, bias, scale, offset, antiquant_scale,
                 antiquant_offset, group_list_real, split_item_value, result);

    return y;
}
#endif

#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
// Motivation for adapting this interface for each Torch version separately:
// 1. Optional TensorList is only supported in Torch2.1 and later versions.
//    Thus, "Tensor[] bias" is used in Torch1.11 and Torch2.0, while
//    "Tensor[]? bias=None" is used in Torch2.1 and later versions.
// 2. Even if "Int[]? group_list=None" is used for all Torch versions, the
//    auto-generated data type for optional IntList group_list in Torch2.1
//    is different from those in Torch1.11 and Torch2.0.
std::vector<at::Tensor> npu_grouped_matmul(const at::TensorList x,
                                           const at::TensorList weight,
                                           const c10::optional<at::TensorList> bias,
                                           const c10::optional<at::TensorList> scale,
                                           const c10::optional<at::TensorList> offset,
                                           const c10::optional<at::TensorList> antiquant_scale,
                                           const c10::optional<at::TensorList> antiquant_offset,
                                           const c10::optional<at::TensorList> per_token_scale,
                                           const c10::optional<at::Tensor>& group_list,
                                           const c10::optional<at::TensorList> activation_input,
                                           const c10::optional<at::TensorList> activation_quant_scale,
                                           const c10::optional<at::TensorList> activation_quant_offset,
                                           c10::optional<int64_t> split_item,
                                           c10::optional<int64_t> group_type,
                                           c10::optional<int64_t> group_list_type,
                                           c10::optional<int64_t> act_type,
                                           const c10::OptionalIntArrayRef tuning_config,
                                           c10::optional<at::ScalarType> output_dtype)
// func: npu_grouped_matmul(Tensor[] x, Tensor[] weight, *, Tensor[]? bias=None, Tensor[]? scale=None,
// Tensor[]? offset=None, Tensor[]? antiquant_scale=None, Tensor[]? antiquant_offset=None,
// Tensor[]? per_token_scale=None, Tensor? group_list=None, Tensor[]? activation_input,
// Tensor[]? activation_quant_offset, Tensor[]? activation_quant_offset, int? split_item=0,
// int? group_type=-1, int? group_list_type=0, int? act_type=0, ScalarType? output_dtype=None) -> Tensor[]
{
    TORCH_CHECK(group_type.has_value(),
                "Requires manual passing group_type, current is None.", OPS_ERROR(ErrCode::VALUE));
    int64_t group_type_value = group_type.value();
    TORCH_CHECK(group_type_value == -1 || group_type_value == 0,
                "group_type currently only support -1 and 0, current value is ",
                group_type_value, OPS_ERROR(ErrCode::VALUE));
    static const bool is_grouped_matmul_V4_available = check_aclnn_kernel_available("aclnnGroupedMatmulV4");
    if (C10_UNLIKELY(!is_grouped_matmul_V4_available)) {
        TORCH_CHECK(!group_list.has_value(),
                    "group_list don't support Tensor input with current cann version. "
                    "Please update cann version to 8.0.RC3 or higher, or use List[int] as input.",
                    OPS_ERROR(ErrCode::VALUE));
        auto num_x = x.size();
        auto num_weight = weight.size();
        auto group_list_real = at::IntArrayRef{};
        size_t num_group_list = 0;
        int64_t split_item_value = split_item.value_or(0);
        check_dims(split_item_value, num_x, num_weight, num_group_list);

        std::vector<at::Tensor> y;
        c10::TensorOptions options = x[0].options().dtype(output_dtype.value_or(x[0].scalar_type()));

        if (split_item_value == IN_NOT_SPLIT_OUT_NOT_SPLIT || split_item_value == IN_SPLIT_OUT_NOT_SPLIT) {
            y.reserve(num_x);
            for (size_t i = 0; i < num_x; i++) {
                create_new_tensor_multi_dim(y, x[i], weight[i].size(1), options);
            }
        } else if (split_item_value == IN_NOT_SPLIT_OUT_SPLIT || split_item_value == IN_SPLIT_OUT_SPLIT) {
            if (num_x > 1) {
                size_t dim_m = 0;
                calculate_dim_m(dim_m, num_x, x);
                create_new_tensor(y, dim_m, weight[0].sizes()[1], options);
            } else if (num_x == 1) {
                create_new_tensor(y, x[0].sizes()[0], weight[0].sizes()[1], options);
            }
        }
        at::TensorList result = at::TensorList(y);

        auto bias_real = bias.value_or(at::TensorList());
        auto scale_real = scale.value_or(at::TensorList());
        auto offset_real = offset.value_or(at::TensorList());
        auto antiquant_scale_real = antiquant_scale.value_or(at::TensorList());
        auto antiquant_offset_real = antiquant_offset.value_or(at::TensorList());
        EXEC_NPU_CMD(aclnnGroupedMatmul, x, weight, bias_real, scale_real, offset_real, antiquant_scale_real,
                     antiquant_offset_real, group_list_real, split_item_value, result);

        return y;
    }

    auto num_x = x.size();
    bool singleWeight = weight.size() == 1 && weight[0].sizes().size() == 3;
    auto num_weight = singleWeight ? static_cast<size_t>(weight[0].size(0)) : static_cast<size_t>(weight.size());
    auto group_list_real = group_list.value_or(at::Tensor());
    auto num_group_list = group_list_real.size(0);
    int64_t split_item_value = split_item.value_or(0);
    check_dims(split_item_value, num_x, num_weight, num_group_list);

    std::vector<at::Tensor> y;
    c10::TensorOptions options = x[0].options().dtype(output_dtype.value_or(x[0].scalar_type()));

    size_t dim_num_w = weight[0].sizes().size();
    size_t n0 = static_cast<size_t>(weight[0].size(dim_num_w - 1));

    if (split_item_value == IN_NOT_SPLIT_OUT_NOT_SPLIT || split_item_value == IN_SPLIT_OUT_NOT_SPLIT) {
        if (num_group_list > 0) {
            y.reserve(num_group_list);
            int64_t glr_value_0 = group_list_real[0].item<int64_t>();
            TORCH_CHECK(glr_value_0 >= 0,
                "group_list[0] should be larger than or equal to 0, but now is ", glr_value_0, "." +
                OPS_ERROR(ErrCode::VALUE));
            create_new_tensor(y, glr_value_0, n0, options);
            int64_t glr_value_pre = glr_value_0;
            for (int i = 1; i < num_group_list; i++) {
                int64_t glr_value_cur = group_list_real[i].item<int64_t>();
                TORCH_CHECK(glr_value_cur - glr_value_pre >= 0,
                    "group_list[", i, "] - group_list[", i - 1, "] should be larger than or equal to 0, but now is ",
                    glr_value_cur - glr_value_pre, "." + OPS_ERROR(ErrCode::VALUE));
                size_t ni = singleWeight ? n0 : weight[i].size(dim_num_w - 1);
                create_new_tensor(y, glr_value_cur - glr_value_pre, ni, options);
                glr_value_pre = glr_value_cur;
            }
        } else {
            y.reserve(num_x);
            for (size_t i = 0; i < num_x; i++) {
                size_t ni = singleWeight ? n0 : weight[i].size(dim_num_w - 1);
                create_new_tensor_multi_dim(y, x[i], ni, options);
            }
        }  // 校验NO_SPLIT时为特殊场景（groupList为空）或num_x > 1
    } else if (split_item_value == IN_NOT_SPLIT_OUT_SPLIT || split_item_value == IN_SPLIT_OUT_SPLIT) {
        if (num_x > 1) {
            size_t dim_m = 0;
            for (size_t i = 0; i < num_x; i++) {
                dim_m += static_cast<size_t>(x[i].size(0));
            }
            weight[0].dtype() == at::ScalarType::Int ?
                create_new_tensor(y, dim_m, n0 * INT4_NUMS_IN_INT32, options) :
                create_new_tensor(y, dim_m, n0, options);
        } else if (num_x == 1) {
            weight[0].dtype() == at::ScalarType::Int ?
                create_new_tensor(y, x[0].size(0), n0 * INT4_NUMS_IN_INT32, options) :
                create_new_tensor(y, x[0].size(0), n0, options);
        }
    }
    at::TensorList result = at::TensorList(y);

    auto bias_real = bias.value_or(at::TensorList());
    auto scale_real = scale.value_or(at::TensorList());
    auto offset_real = offset.value_or(at::TensorList());
    auto antiquant_scale_real = antiquant_scale.value_or(at::TensorList());
    auto antiquant_offset_real = antiquant_offset.value_or(at::TensorList());
    auto per_token_scale_real = per_token_scale.value_or(at::TensorList());
    auto activation_input_real = activation_input.value_or(at::TensorList());
    auto activation_quant_scale_real = activation_quant_scale.value_or(at::TensorList());
    auto activation_quant_offset_real = activation_quant_offset.value_or(at::TensorList());
    auto act_out = at::TensorList();
    auto dynamic_quant_scale_out = at::TensorList();
    int64_t group_list_type_value = group_list_type.value_or(0);
    int64_t act_type_value = act_type.value_or(0);
    auto tuning_config_real = tuning_config.value_or(at::IntArrayRef{});

    const bool is_weight_nz = at_npu::native::custom_ops::get_npu_format(weight[0]) == ACL_FORMAT_FRACTAL_NZ;
    if (is_weight_nz) {
        static const bool is_weight_nz_available = check_aclnn_kernel_available("aclnnGroupedMatmulWeightNz");
        TORCH_CHECK(is_weight_nz_available,
                    "Format of weight in npu_grouped_matmul is FRACTAL_NZ, current CANN version "
                    "do not support with this format. Please try to update the version of CANN."
                    + OPS_ERROR(ErrCode::PARAM));
        int64_t quant_per_group_size = 0;
        EXEC_NPU_CMD(aclnnGroupedMatmulWeightNz, x, weight, bias_real, scale_real, offset_real, antiquant_scale_real,
            antiquant_offset_real, per_token_scale_real, group_list_real, activation_input_real,
            activation_quant_scale_real, activation_quant_offset_real, split_item_value, group_type_value,
            group_list_type_value, act_type_value, tuning_config_real, quant_per_group_size,
            result, act_out, dynamic_quant_scale_out);
        return y;
    }
    static const bool is_grouped_matmul_V5_available = check_aclnn_kernel_available("aclnnGroupedMatmulV5");
    if (!is_grouped_matmul_V5_available) {
        EXEC_NPU_CMD(aclnnGroupedMatmulV4, x, weight, bias_real, scale_real, offset_real, antiquant_scale_real,
            antiquant_offset_real, per_token_scale_real, group_list_real, activation_input_real,
            activation_quant_scale_real, activation_quant_offset_real, split_item_value, group_type_value,
            group_list_type_value, act_type_value, result, act_out, dynamic_quant_scale_out);
    } else {
        EXEC_NPU_CMD(aclnnGroupedMatmulV5, x, weight, bias_real, scale_real, offset_real, antiquant_scale_real,
            antiquant_offset_real, per_token_scale_real, group_list_real, activation_input_real,
            activation_quant_scale_real, activation_quant_offset_real, split_item_value, group_type_value,
            group_list_type_value, act_type_value, tuning_config_real, result, act_out, dynamic_quant_scale_out);
    }
    return y;
}

// Motivation for adapting this interface for each Torch version separately:
// 1. Optional TensorList is only supported in Torch2.1 and later versions.
//    Thus, "Tensor[] bias" is used in Torch1.11 and Torch2.0, while
//    "Tensor[]? bias=None" is used in Torch2.1 and later versions.
// 2. Even if "Int[]? group_list=None" is used for all Torch versions, the
//    auto-generated data type for optional IntList group_list in Torch2.1
//    is different from those in Torch1.11 and Torch2.0.
std::vector<at::Tensor> npu_grouped_matmul(const at::TensorList x,
                                           const at::TensorList weight,
                                           const c10::optional<at::TensorList> bias,
                                           const c10::optional<at::TensorList> scale,
                                           const c10::optional<at::TensorList> offset,
                                           const c10::optional<at::TensorList> antiquant_scale,
                                           const c10::optional<at::TensorList> antiquant_offset,
                                           const c10::optional<at::TensorList> per_token_scale,
                                           c10::OptionalIntArrayRef group_list,
                                           const c10::optional<at::TensorList> activation_input,
                                           const c10::optional<at::TensorList> activation_quant_scale,
                                           const c10::optional<at::TensorList> activation_quant_offset,
                                           c10::optional<int64_t> split_item,
                                           c10::optional<int64_t> group_type,
                                           c10::optional<int64_t> group_list_type,
                                           c10::optional<int64_t> act_type,
                                           c10::optional<at::ScalarType> output_dtype)
{
    auto num_x = x.size();
    auto num_weight = weight.size();
    auto group_list_real = group_list.value_or(at::IntArrayRef{});
    auto num_group_list = group_list_real.size();
    int64_t split_item_value = split_item.value_or(0);
    check_dims(split_item_value, num_x, num_weight, num_group_list);

    std::vector<at::Tensor> y;
    c10::TensorOptions options = x[0].options().dtype(output_dtype.value_or(x[0].scalar_type()));

    if (split_item_value == IN_NOT_SPLIT_OUT_NOT_SPLIT || split_item_value == IN_SPLIT_OUT_NOT_SPLIT) {
        if (num_group_list > 0) {
            y.reserve(num_group_list);
            TORCH_CHECK(group_list_real[0] >= 0,
                "group_list[0] should be larger than or equal to 0, but now is ", group_list_real[0], "." +
                OPS_ERROR(ErrCode::VALUE));
            create_new_tensor(y, group_list_real[0], weight[0].sizes()[1], options);
            for (size_t i = 1; i < num_group_list; i++) {
                TORCH_CHECK(group_list_real[i] - group_list_real[i - 1] >= 0,
                    "group_list[", i, "] - group_list[", i - 1, "] should be larger than or equal to 0, but now is ",
                    group_list_real[i] - group_list_real[i - 1], "." + OPS_ERROR(ErrCode::VALUE));
                create_new_tensor(y, group_list_real[i] - group_list_real[i - 1], weight[i].sizes()[1], options);
            }
        } else {
            y.reserve(num_x);
            for (size_t i = 0; i < num_x; i++) {
                create_new_tensor_multi_dim(y, x[i], weight[i].size(1), options);
            }
        }  // 校验NO_SPLIT时为特殊场景（groupList为空）或num_x > 1
    } else if (split_item_value == IN_NOT_SPLIT_OUT_SPLIT || split_item_value == IN_SPLIT_OUT_SPLIT) {
        if (num_x > 1) {
            size_t dim_m = 0;
            for (size_t i = 0; i < num_x; i++) {
                dim_m += x[i].sizes()[0];
            }
            create_new_tensor(y, dim_m, weight[0].sizes()[1], options);
        } else if (num_x == 1) {
            create_new_tensor(y, x[0].sizes()[0], weight[0].sizes()[1], options);
        }
    }
    at::TensorList result = at::TensorList(y);

    auto bias_real = bias.value_or(at::TensorList());
    auto scale_real = scale.value_or(at::TensorList());
    auto offset_real = offset.value_or(at::TensorList());
    auto antiquant_scale_real = antiquant_scale.value_or(at::TensorList());
    auto antiquant_offset_real = antiquant_offset.value_or(at::TensorList());
    EXEC_NPU_CMD(aclnnGroupedMatmul, x, weight, bias_real, scale_real, offset_real, antiquant_scale_real,
                 antiquant_offset_real, group_list_real, split_item_value, result);

    return y;
}
#endif
}

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
using npu_preparation = at_npu::native::OpPreparation;

void check_dims(int64_t split_item, size_t num_x, size_t num_weight, size_t num_group_list)
{
    TORCH_CHECK(num_x > 0 && num_weight > 0,
                "Invalid inputs: neither x nor weight could be empty." + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(IN_NOT_SPLIT_OUT_NOT_SPLIT == split_item || IN_SPLIT_OUT_NOT_SPLIT == split_item ||
                IN_NOT_SPLIT_OUT_SPLIT == split_item || IN_SPLIT_OUT_SPLIT == split_item,
                "Invalid value of split_item [", split_item, "], which should only be one of 0/1/2/3."
                + OPS_ERROR(ErrCode::PARAM));
    if (IN_NOT_SPLIT_OUT_NOT_SPLIT == split_item || IN_SPLIT_OUT_NOT_SPLIT == split_item) {
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

void create_new_tensor_multi_dim(std::vector<at::Tensor> &y, const at::Tensor &x_i, const at::Tensor &weight_i,
                                 c10::TensorOptions options)
{
    auto x_sizes = x_i.sizes();
    std::vector<int64_t> y_sizes(x_sizes.begin(), x_sizes.end());
    y_sizes.at(x_sizes.size() - 1) = weight_i.sizes()[1];

    auto output_size = op_infer::array_to_small_vector(y_sizes);
    y.emplace_back(npu_preparation::apply_tensor_without_format(output_size, options));
}

void create_new_tensor(std::vector<at::Tensor> &y, size_t dim_m, size_t dim_n, c10::TensorOptions options)
{
    auto output_size = op_infer::array_to_small_vector({dim_m, dim_n});
    y.emplace_back(npu_preparation::apply_tensor_without_format(output_size, options));
}

// Motivation for adapting this interface for each Torch version separately:
// 1. Optional TensorList is only supported in Torch2.1 and later versions.
//    Thus, "Tensor[] bias" is used in Torch1.11 and Torch2.0, while
//    "Tensor[]? bias=None" is used in Torch2.1 and later versions.
// 2. Even if "Int[]? group_list=None" is used for all Torch versions, the
//    auto-generated data type for optional IntList group_list in Torch2.2
//    is different from those in Torch1.11 and Torch2.0.
std::vector<at::Tensor> npu_grouped_matmul(const at::TensorList x,
                                           const at::TensorList weight,
                                           const c10::optional<at::TensorList> bias,
                                           const c10::optional<at::TensorList> scale,
                                           const c10::optional<at::TensorList> offset,
                                           const c10::optional<at::TensorList> antiquant_scale,
                                           const c10::optional<at::TensorList> antiquant_offset,
                                           c10::OptionalIntArrayRef group_list,
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
            for (int i = 1; i < num_group_list; i++) {
                TORCH_CHECK(group_list_real[i] - group_list_real[i - 1] >= 0,
                    "group_list[", i, "] - group_list[", i - 1, "] should be larger than or equal to 0, but now is ",
                    group_list_real[i] - group_list_real[i - 1], "." + OPS_ERROR(ErrCode::VALUE));
                create_new_tensor(y, group_list_real[i] - group_list_real[i - 1], weight[i].sizes()[1], options);
            }
        } else {
            y.reserve(num_x);
            for (int i = 0; i < num_x; i++) {
                create_new_tensor_multi_dim(y, x[i], weight[i], options);
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

    auto bias_real = bias.value_or(at::TensorList());
    auto scale_real = scale.value_or(at::TensorList());
    auto offset_real = offset.value_or(at::TensorList());
    auto antiquant_scale_real = antiquant_scale.value_or(at::TensorList());
    auto antiquant_offset_real = antiquant_offset.value_or(at::TensorList());
    EXEC_NPU_CMD(aclnnGroupedMatmul, x, weight, bias_real, scale_real, offset_real, antiquant_scale_real,
                 antiquant_offset_real, group_list_real, split_item_value, result);

    return y;
}
}  // namespace op_api


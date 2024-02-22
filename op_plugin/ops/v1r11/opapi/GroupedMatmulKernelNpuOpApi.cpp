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

bool check_dims(int64_t split_item, size_t num_x, size_t num_weight, size_t num_group_list)
{
    if (IN_NOT_SPLIT_OUT_NOT_SPLIT == split_item || IN_NOT_SPLIT_OUT_SPLIT == split_item) { // When split_item_value = 0 or 2: (1) length of x equals that of weight; (2) length of group_list equals 0
        return num_x == num_weight && 0 == num_group_list;
    } else if (IN_SPLIT_OUT_NOT_SPLIT == split_item || IN_SPLIT_OUT_SPLIT == split_item) { // When split_item_value = 1 or 3: (1) length of x equals 1; (2) length of weight equals that of group_list
        return num_x == 1 && num_weight == num_group_list;
    } else { // split_item must be one of 0/1/2/3
        return false;
    }
}

void creat_new_tensor(std::vector<at::Tensor> &y, size_t dim_m, size_t dim_n, c10::TensorOptions options)
{
    auto output_size = op_infer::array_to_small_vector({dim_m, dim_n});
    y.push_back(npu_preparation::apply_tensor_without_format(output_size, options));
}

std::vector<at::Tensor> npu_grouped_matmul(const at::TensorList x,
                                           const at::TensorList weight,
                                           const at::TensorList bias,
                                           c10::optional<at::IntArrayRef> group_list,
                                           c10::optional<int64_t> split_item)
{
    auto num_x = x.size();
    auto num_weight = weight.size();
    auto group_list_real = group_list.value_or(at::IntArrayRef{});
    auto num_group_list = group_list_real.size();
    int64_t split_item_value = split_item.value_or(0);

    std::vector<at::Tensor> y;
    c10::TensorOptions options = x[0].options().dtype(x[0].scalar_type());

    TORCH_CHECK(check_dims(split_item_value, num_x, num_weight, num_group_list),
                "Invalid value of split_item or invalid dims of inputs: split_item = ", split_item_value,
                ", num_x = ", num_x, ", num_weight = ", num_weight, ", num_group_list = ", num_group_list);

    if (IN_NOT_SPLIT_OUT_NOT_SPLIT == split_item_value) {
        y.reserve(num_x);
        for (int i = 0; i < num_x; i++) {
            creat_new_tensor(y, x[i].sizes()[0], weight[i].sizes()[1], options);
        }
    } else if (IN_SPLIT_OUT_NOT_SPLIT == split_item_value) {
        y.reserve(num_weight);
        for (int i = 0; i < num_weight; i++) {
            creat_new_tensor(y, group_list_real[i], weight[i].sizes()[1], options);
        }
    } else if (IN_NOT_SPLIT_OUT_SPLIT == split_item_value) {
        size_t dim_m = 0;
        for (int i = 0; i < num_x; i++) {
            dim_m += x[i].sizes()[0];
        }
        creat_new_tensor(y, dim_m, weight[0].sizes()[1], options);
    } else if (IN_SPLIT_OUT_SPLIT == split_item_value) {
        creat_new_tensor(y, x[0].sizes()[0], weight[0].sizes()[1], options);
    }
    at::TensorList result = at::TensorList(y);

    EXEC_NPU_CMD(aclnnGroupedMatmul, x, weight, bias, group_list_real, split_item_value, result);

    return y;
}
}  // namespace op_api


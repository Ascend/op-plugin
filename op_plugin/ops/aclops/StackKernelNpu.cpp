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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"
#include "op_plugin/third_party/acl/inc/op_proto/all_ops.h"


namespace acl_op {
using DyNumAndIndex = std::vector<std::pair<uint32_t, uint32_t>>;
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
at_npu::native::DynamicInputRegFunc stack_func = [](DyNumAndIndex num_and_index,
                                                    std::string op_name) -> ge::OperatorPtr {
    auto ge_op = std::make_shared<ge::op::Pack>(op_name.c_str());
    ge_op->create_dynamic_input_byindex_x(num_and_index.front().first, num_and_index.front().second);
    return ge_op;
};

at::SmallVector<int64_t, SIZE> stack_npu_output_size(at::TensorList tensors, int64_t dim)
{
    dim = op_plugin::utils::make_warp_dim(dim, tensors[0].dim() + 1);
    at::SmallVector<int64_t, SIZE> shape;
    for (int i = 0; i < dim; i++) {
        shape.emplace_back(tensors[0].size(i));
    }
    shape.emplace_back(tensors.size());
    for (int i = dim; i < tensors[0].dim(); i++) {
        shape.emplace_back(tensors[0].size(i));
    }

    return shape;
}

at::Tensor &stack_out_nocheck(at::Tensor &result, at::TensorList tensors, int64_t dim)
{
    c10::SmallVector<at::Tensor, N> input_tensors;
    for (uint i = 0; i < tensors.size(); i++) {
        input_tensors.emplace_back(tensors[i]);
    }

    auto dynamic_num = input_tensors.size();
    at_npu::native::OpCommand cmd;
    cmd.Name("Pack").DynamicInputReg(stack_func, {{dynamic_num, 0}});
    for (uint i = 0; i < dynamic_num; i++) {
        string input_name = "x" + std::to_string(i);
        cmd.Input(input_tensors[i], input_name);
    }
    cmd.Output(result).Attr("N", static_cast<int64_t>(tensors.size())).Attr("axis", dim).Run();

    return result;
}
} // namespace

at::Tensor &stack_out(at::TensorList tensors, int64_t dim, at::Tensor &out)
{
    auto output_size = stack_npu_output_size(tensors, dim);

    npu_preparation::CheckOut({tensors[0]}, out, ACL_FORMAT_ND, tensors[0].scalar_type(), output_size);
    if (!npu_utils::check_match(&out)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(out);
        stack_out_nocheck(contiguous_result, tensors, dim);
        npu_utils::format_fresh_view(out, contiguous_result);
    } else {
        stack_out_nocheck(out, tensors, dim);
    }

    return out;
}

at::Tensor stack(at::TensorList tensors, int64_t dim)
{
    auto output_size = stack_npu_output_size(tensors, dim);

    at::Tensor result = npu_preparation::apply_tensor_with_format(output_size, tensors[0].options(), ACL_FORMAT_ND);

    stack_out_nocheck(result, tensors, dim);

    return result;
}
} // namespace acl_op

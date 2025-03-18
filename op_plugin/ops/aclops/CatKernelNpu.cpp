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

#include <ATen/native/TypeProperties.h>
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"
#include "op_plugin/third_party/acl/inc/op_proto/all_ops.h"
#include "op_plugin/utils/custom_functions/aclops/inner_compute.h"

namespace acl_op {
using DyNumAndIndex = std::vector<std::pair<uint32_t, uint32_t>>;
using npu_preparation = at_npu::native::OpPreparation;
using calcu_op_util = at_npu::native::CalcuOpUtil;
using npu_utils = at_npu::native::NpuUtils;

namespace {
template <typename ge_op_type>
at_npu::native::DynamicInputRegFunc concat_func =
    [](DyNumAndIndex num_and_index, std::string op_name) -> ge::OperatorPtr {
    auto ge_op = std::make_shared<ge_op_type>(op_name.c_str());
    ge_op->create_dynamic_input_byindex_x(num_and_index.front().first, num_and_index.front().second);
    return ge_op;
};

#if VERSION_BETWEEN(V1R11, V1R11)
c10::SmallVector<at::Tensor, N> cat_dest_tensor_list(at::TensorList tensors)
{
    at::ScalarType high_type = at::native::result_type(tensors);
    c10::SmallVector<at::Tensor, N> dst_tensor_list;
    // pytorch supports empty tensors, which needs to be removed from the NPU.
    for (at::Tensor tensor : tensors) {
        if (tensor.dim() == 1 && tensor.sizes()[0] == 0) {
            continue;
        }
        if (tensor.scalar_type() != high_type) {
            tensor = at_npu::native::custom_ops::npu_dtype_cast(tensor, high_type);
        }
        dst_tensor_list.emplace_back(tensor);
    }
    return dst_tensor_list;
}

at::Tensor& cat_out_nocheck(at::Tensor& result, at::TensorList tensors, int64_t dim)
{
    c10::SmallVector<at::Tensor, N> input_tensors = cat_dest_tensor_list(tensors);
    int64_t dim_post_expr = 0;
    if (input_tensors.size() > 0) {
        dim_post_expr = input_tensors[0].dim();
    } else {
        return result;
    }
    dim = op_plugin::utils::make_warp_dim(dim, dim_post_expr);
    int64_t input_number = 0;
    at_npu::native::OpCommand cmd;
    cmd.Name("ConcatD");
    for (size_t i = 0; i < input_tensors.size(); i++) {
        if (input_tensors[i].numel() != 0) {
            string input_name = "x" + std::to_string(input_number++);
            cmd.Input(input_tensors[i], input_name);
        }
    }

    cmd.DynamicInputReg(concat_func<ge::op::ConcatD>, {{input_number, 0}})
        .Output(result)
        .Attr("N", input_number)
        .Attr("concat_dim", dim)
        .Run();
    return result;
}
#endif

#if VERSION_BETWEEN(V2R0, VERSION_NEWEST)
c10::SmallVector<at::Tensor, N> cat_dest_tensor_list(const at::MaterializedITensorListRef& tensors)
{
    auto high_type = at::native::result_type(tensors);
    c10::SmallVector<at::Tensor, N> dst_tensor_list;
    // pytorch supports empty tensors, which needs to be removed from the NPU.
    for (const at::Tensor& t : tensors) {
        at::Tensor tensor = t;
        if (tensor.dim() == 1 && tensor.sizes()[0] == 0) {
            continue;
        }
        if (tensor.scalar_type() != high_type) {
            tensor = at_npu::native::custom_ops::npu_dtype_cast(tensor, high_type);
        }
        dst_tensor_list.emplace_back(tensor);
    }
    return dst_tensor_list;
}

at::Tensor& cat_output_nocheck(at::Tensor& result, const at::MaterializedITensorListRef& tensors, int64_t dim)
{
    if (tensors.size() == 1) {
        return result.copy_(tensors[0].get());
    }

    c10::SmallVector<at::Tensor, N> input_tensors = cat_dest_tensor_list(tensors);
    int64_t dim_post_expr = 0;
    if (input_tensors.size() > 0) {
        dim_post_expr = input_tensors[0].dim();
    } else {
        return result;
    }
    dim = op_plugin::utils::make_warp_dim(dim, dim_post_expr);

    int64_t input_number = 0;
    at_npu::native::OpCommand cmd;
    cmd.Name("ConcatD");
    for (size_t i = 0; i < input_tensors.size(); i++) {
        if (input_tensors[i].numel() != 0) {
            string input_name = "x" + std::to_string(input_number++);
            cmd.Input(input_tensors[i], input_name);
        }
    }

    cmd.DynamicInputReg(concat_func<ge::op::ConcatD>, {{input_number, 0}})
        .Output(result)
        .Attr("N", input_number)
        .Attr("concat_dim", dim)
        .Run();
    return result;
}
#endif
} // namespace

at::Tensor& cat_out(at::TensorList tensors, at::Dimname dim, at::Tensor& result)
{
    TORCH_CHECK(tensors.size() > 0, "cat inputs should not be empty." + OPS_ERROR(ErrCode::PARAM));
    return at::cat_out(result, tensors, dimname_to_position(tensors[0], dim));
}

at::Tensor cat(at::TensorList tensors, at::Dimname dim)
{
    TORCH_CHECK(tensors.size() > 0, "cat inputs should not be empty." + OPS_ERROR(ErrCode::PARAM));
    return at::cat(tensors, dimname_to_position(tensors[0], dim));
}

#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor& _cat_out(at::TensorList tensors, int64_t dim, at::Tensor& result)
{
    if (tensors.size() == 1) {
        return result.copy_(tensors[0]);
    }

    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        cat_out_nocheck(contiguous_result, tensors, dim);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        cat_out_nocheck(result, tensors, dim);
    }
    return result;
}

at::Tensor& cat_out(at::TensorList tensors, int64_t dim, at::Tensor& result)
{
    c10::SmallVector<at::Tensor, N> input_tensors = cat_dest_tensor_list(tensors);

    int64_t dim_post_expr = 0;
    if (input_tensors.size() > 0) {
        dim_post_expr = input_tensors[0].dim();
    } else {
        return result;
    }
    dim = op_plugin::utils::make_warp_dim(dim, dim_post_expr);
    auto output_size = op_infer::cat_npu_output_size(input_tensors, dim);
    npu_preparation::CheckOut(
        {tensors[0]},
        result,
        ACL_FORMAT_ND,
        tensors[0].scalar_type(),
        output_size);

    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        at::_cat_out(contiguous_result, tensors, dim);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        at::_cat_out(result, tensors, dim);
    }
    return result;
}

at::Tensor _cat(at::TensorList tensors, int64_t dim)
{
    c10::SmallVector<at::Tensor, N> input_tensors = cat_dest_tensor_list(tensors);

    int64_t dim_post_expr = 0;
    if (input_tensors.size() > 0) {
        dim_post_expr = input_tensors[0].dim();
    } else {
        at::Tensor result = npu_preparation::apply_tensor(tensors[0]);
        return result;
    }
    dim = op_plugin::utils::make_warp_dim(dim, dim_post_expr);
    auto output_size = op_infer::cat_npu_output_size(input_tensors, dim);

    // check tensors_dim for output format setting
    bool tensors_dim_check = true;
    for (at::Tensor t : tensors) {
        if (t.sizes().size() != 4) {
            break;
        }
        int64_t C = t.size(1);
        if (C % 16 != 0) {
            tensors_dim_check = false;
            break;
        }
    }

    at::Tensor result = npu_preparation::apply_tensor(input_tensors[0], output_size);
    if (!tensors_dim_check) {
        result = npu_preparation::apply_tensor_with_format(input_tensors[0], output_size, ACL_FORMAT_ND);
    }
    acl_op::_cat_out(tensors, dim, result);
    return result;
}

at::Tensor cat(at::TensorList tensors, int64_t dim)
{
    return at::_cat(tensors, dim);
}
#endif

#if VERSION_BETWEEN(V2R0, V2R0)
at::Tensor& cat_out(const at::ITensorListRef& tensors, int64_t dim, at::Tensor& result)
{
    auto materialized = tensors.materialize();
    c10::SmallVector<at::Tensor, N> input_tensors = cat_dest_tensor_list(materialized);

    int64_t dim_post_expr = 0;
    if (input_tensors.size() > 0) {
        dim_post_expr = input_tensors[0].dim();
    } else {
        at::Tensor output = npu_preparation::apply_tensor(materialized[0], result.options());
        result.resize_({0}).copy_(output);
        return result;
    }
    dim = op_plugin::utils::make_warp_dim(dim, dim_post_expr);
    auto output_size = op_infer::cat_npu_output_size(input_tensors, dim);
    npu_preparation::CheckOut(
        {materialized[0].get()},
        result,
        ACL_FORMAT_ND,
        materialized[0].get().scalar_type(),
        output_size);

    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        cat_output_nocheck(contiguous_result, materialized, dim);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        cat_output_nocheck(result, materialized, dim);
    }
    return result;
}

at::Tensor cat(const at::ITensorListRef& tensors, int64_t dim)
{
    auto materialized = tensors.materialize();
    c10::SmallVector<at::Tensor, N> input_tensors = cat_dest_tensor_list(materialized);

    int64_t dim_post_expr = 0;
    if (input_tensors.size() > 0) {
        dim_post_expr = input_tensors[0].dim();
    } else {
        at::Tensor result = npu_preparation::apply_tensor(materialized[0]);
        return result;
    }
    dim = op_plugin::utils::make_warp_dim(dim, dim_post_expr);
    auto output_size = op_infer::cat_npu_output_size(input_tensors, dim);

    // check tensors_dim for output format setting
    bool tensors_dim_check = true;
    for (at::Tensor t : materialized) {
        if (t.sizes().size() != 4) {
            break;
        }
        int64_t C = t.size(1);
        if (C % 16 != 0) {
            tensors_dim_check = false;
            break;
        }
    }

    at::Tensor result = tensors_dim_check ?
        npu_preparation::apply_tensor(input_tensors[0], output_size) :
        npu_preparation::apply_tensor_with_format(input_tensors[0], output_size, ACL_FORMAT_ND);
    cat_output_nocheck(result, materialized, dim);
    return result;
}
#endif

#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
at::Tensor& cat_out(const at::ITensorListRef& tensors, int64_t dim, at::Tensor& result)
{
    auto materialized = tensors.materialize();
    c10::SmallVector<at::Tensor, N> input_tensors = cat_dest_tensor_list(materialized);

    int64_t dim_post_expr = 0;
    if (input_tensors.size() > 0) {
        dim_post_expr = input_tensors[0].dim();
    } else {
        at::Tensor output = npu_preparation::apply_tensor(materialized[0], result.options());
        result.resize_({0}).copy_(output);
        return result;
    }
    dim = op_plugin::utils::make_warp_dim(dim, dim_post_expr);
    auto output_size = op_infer::cat_npu_output_size(input_tensors, dim);
    npu_preparation::CheckOut(
        {materialized[0].get()},
        result,
        ACL_FORMAT_ND,
        materialized[0].get().scalar_type(),
        output_size);

    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        cat_output_nocheck(contiguous_result, materialized, dim);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        cat_output_nocheck(result, materialized, dim);
    }
    return result;
}

at::Tensor cat(const at::ITensorListRef& tensors, int64_t dim)
{
    auto materialized = tensors.materialize();
    c10::SmallVector<at::Tensor, N> input_tensors = cat_dest_tensor_list(materialized);

    int64_t dim_post_expr = 0;
    if (input_tensors.size() > 0) {
        dim_post_expr = input_tensors[0].dim();
    } else {
        at::Tensor result = npu_preparation::apply_tensor(materialized[0]);
        return result;
    }
    dim = op_plugin::utils::make_warp_dim(dim, dim_post_expr);
    auto output_size = op_infer::cat_npu_output_size(input_tensors, dim);

    // check tensors_dim for output format setting
    bool tensors_dim_check = true;
    for (at::Tensor t : materialized) {
        if (t.sizes().size() != 4) {
            break;
        }
        int64_t C = t.size(1);
        if (C % 16 != 0) {
            tensors_dim_check = false;
            break;
        }
    }

    at::Tensor result = tensors_dim_check ?
        npu_preparation::apply_tensor(input_tensors[0], output_size) :
        npu_preparation::apply_tensor_with_format(input_tensors[0], output_size, ACL_FORMAT_ND);
    cat_output_nocheck(result, materialized, dim);
    return result;
}
#endif
} // namespace acl_op

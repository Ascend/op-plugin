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
// See the License for the specific

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
static inline int64_t calculate_prod_output_format(const at::Tensor &self, at::IntArrayRef size)
{
    int64_t npu_format = npu_preparation::get_tensor_npu_format(self);
    // scalar scene no support nz
    if (size.empty()) {
        npu_format = ACL_FORMAT_ND;
    }
    return npu_format;
}

at::Tensor &prod_out_npu_nocheck(at::Tensor &result, const at::Tensor &self,
                                 c10::SmallVector<int64_t, at_npu::native::N> dim_list, bool keepdim,
                                 c10::optional<at::ScalarType> dtype)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("ReduceProd").Input(self).Input(dim_list).Output(result).Attr("keep_dims", keepdim).Run();
    return result;
}

at::ScalarType get_cal_type(const at::Tensor &self, const c10::optional<at::ScalarType> &dtype)
{
    at::ScalarType cal_type = dtype.has_value() ? dtype.value() : self.scalar_type();
    if (cal_type == at::ScalarType::Half) {
        cal_type = at::ScalarType::Float;
    } else if (cal_type == at::ScalarType::Bool) {
        cal_type = at::ScalarType::Long;
    }
    return cal_type;
}

at::ScalarType get_dst_type(const at::Tensor &self, const c10::optional<at::ScalarType> &dtype)
{
    if (dtype.has_value()) {
        return dtype.value();
    }
    at::ScalarType dst_type = self.scalar_type();
    if (isIntegralType(dst_type, true)) {
        return at::ScalarType::Long;
    }
    return dst_type;
}
} // namespace

at::Tensor &prod_out(const at::Tensor &self, int64_t dim, bool keepdim, c10::optional<at::ScalarType> dtype,
                     at::Tensor &result)
{
    auto output_size = op_infer::prod_npu_output_size(self, dim, keepdim);
    at::ScalarType dst_type = dtype.has_value() ? dtype.value() : result.scalar_type();

    npu_preparation::CheckOut({self}, result, ACL_FORMAT_ND, dst_type, output_size);

    at::ScalarType cal_type = get_cal_type(self, dtype);
    at::Tensor self_tmp =
        self.scalar_type() != cal_type ? at_npu::native::custom_ops::npu_dtype_cast(self, cal_type) : self;
    at::Tensor result_tmp =
        result.scalar_type() != cal_type ? at_npu::native::custom_ops::npu_dtype_cast(result, cal_type) : result;

    c10::SmallVector<int64_t, N> dim_now = {dim};
    if (self.dim() == 0) {
        dim_now = op_plugin::utils::get_dimlist_for_tensor(self);
    }

    if (!npu_utils::check_match(&result_tmp)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result_tmp);
        prod_out_npu_nocheck(contiguous_result, self_tmp, dim_now, keepdim, dtype);
        npu_utils::format_fresh_view(result_tmp, contiguous_result);
    } else {
        prod_out_npu_nocheck(result_tmp, self_tmp, dim_now, keepdim, dtype);
    }

    if (cal_type != dst_type) {
        result_tmp = at_npu::native::custom_ops::npu_dtype_cast(result_tmp, dst_type);
        result.copy_(result_tmp);
    }
    return result;
}

at::Tensor prod(const at::Tensor &self, int64_t dim, bool keepdim, c10::optional<at::ScalarType> dtype)
{
    at::ScalarType cal_type = get_cal_type(self, dtype);
    at::Tensor self_tmp =
        self.scalar_type() != cal_type ? at_npu::native::custom_ops::npu_dtype_cast(self, cal_type) : self;

    auto output_size = op_infer::prod_npu_output_size(self, dim, keepdim);
    int64_t npu_format = calculate_prod_output_format(self_tmp, output_size);
    at::Tensor result = npu_preparation::apply_tensor_with_format(output_size, self_tmp.options(), npu_format);
    at::ScalarType dst_type = get_dst_type(self, dtype);

    c10::SmallVector<int64_t, N> dim_now = {dim};
    if (self.dim() == 0) {
        dim_now = op_plugin::utils::get_dimlist_for_tensor(self);
    }

    prod_out_npu_nocheck(result, self_tmp, dim_now, keepdim, dtype);
    if (cal_type != dst_type) {
        result = at_npu::native::custom_ops::npu_dtype_cast(result, dst_type);
    }
    return result;
}

at::Tensor prod(const at::Tensor &self, c10::optional<at::ScalarType> dtype)
{
    at::ScalarType cal_type = get_cal_type(self, dtype);
    at::Tensor self_tmp =
        self.scalar_type() != cal_type ? at_npu::native::custom_ops::npu_dtype_cast(self, cal_type) : self;

    auto output_size = op_infer::prod_npu_output_size(self, false);
    int64_t npu_format = calculate_prod_output_format(self, output_size);
    at::Tensor result = npu_preparation::apply_tensor_with_format(output_size, self_tmp.options(), npu_format);
    at::ScalarType dst_type = get_dst_type(self, dtype);

    prod_out_npu_nocheck(result, self_tmp, op_plugin::utils::get_dimlist_for_tensor(self), false, dtype);
    if (cal_type != dst_type) {
        result = at_npu::native::custom_ops::npu_dtype_cast(result, dst_type);
    }
    return result;
}
} // namespace acl_op

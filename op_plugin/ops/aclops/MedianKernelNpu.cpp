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

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
c10::SmallVector<int64_t, SIZE> median_npu_output_size(const at::Tensor &self, int64_t dim, bool keepdim)
{
    dim = op_plugin::utils::make_warp_dim(dim, self.dim());
    at::IntArrayRef dims(dim);
    return op_infer::reduce_ops_npu_output_size(self, dims, keepdim);
}

at::Tensor &median_out_nocheck(at::Tensor &result, const at::Tensor &self)
{
    int64_t size = self.numel();
    if (size <= 0) {
        result = at::full({}, std::numeric_limits<float>::quiet_NaN()).to(self.options());
        return result;
    }

    at::Tensor input = self.has_names() ? self.rename(c10::nullopt).reshape({-1}) : self.reshape({-1});
    int64_t k = input.size(0) / 2;

    auto ret = at::topk(input, k + 1);
    at::Tensor topk_values = std::get<0>(ret);
    at::Tensor value = topk_values[k];
    acl_op::fill_(result, value);
    return result;
}

std::tuple<at::Tensor &, at::Tensor &> median_out_value_nocheck(at::Tensor &values, at::Tensor &indices,
                                                                const at::Tensor &self, int64_t dim, bool keepdim)
{
    dim = op_plugin::utils::make_warp_dim(dim, self.dim());
    int64_t k = self.dim() > 0 ? (self.size(dim) + 1) / 2 : 1;

    at::Tensor self_name = self.has_names() ? self.rename(c10::nullopt) : self;
    auto ret = at::topk(self_name, k, dim, false, true);
    at::Tensor topk_values = std::get<0>(ret);
    at::Tensor topkIndices = std::get<1>(ret);

    // NCHW -> reflush base format
    at::Tensor index =
        npu_preparation::apply_tensor_with_format({1}, self_name.options().dtype(at::kLong), ACL_FORMAT_NCHW);
    acl_op::fill_(index, k - 1);
    at::Tensor values_index_select = acl_op::index_select(topk_values, dim, index);
    at::Tensor indices_index_select = acl_op::index_select(topkIndices, dim, index);
    if (!keepdim) {
        values_index_select.squeeze_(dim);
        indices_index_select.squeeze_(dim);
    }
    values.copy_(values_index_select);
    indices.copy_(indices_index_select);
    at::namedinference::propagate_names_for_reduction(values, self, dim, keepdim);
    at::namedinference::propagate_names_for_reduction(indices, self, dim, keepdim);
    return std::tuple<at::Tensor &, at::Tensor &>(values, indices);
}

std::tuple<at::Tensor &, at::Tensor &> median_out_npu_nocheck(at::Tensor &values, at::Tensor &indices,
                                                              const at::Tensor &self, at::Dimname dim, bool keepdim)
{
    return median_out_value_nocheck(values, indices, self, dimname_to_position(self, dim), keepdim);
}
} // namespace

std::tuple<at::Tensor &, at::Tensor &> median_out(const at::Tensor &self, int64_t dim, bool keepdim, at::Tensor &values,
                                                  at::Tensor &indices)
{
    auto output_size = median_npu_output_size(self, dim, keepdim);
    npu_preparation::CheckOut({self}, values, ACL_FORMAT_ND, self.scalar_type(), output_size);

    npu_preparation::CheckOut({self}, indices, ACL_FORMAT_ND, at::ScalarType::Long, output_size);

    bool values_match = npu_utils::check_match(&values);
    bool indices_match = npu_utils::check_match(&indices);
    if (!(values_match && indices_match)) {
        at::Tensor contiguous_values = values_match ? values : npu_utils::format_contiguous(values);
        at::Tensor contiguous_indices = indices_match ? indices : npu_utils::format_contiguous(indices);
        median_out_value_nocheck(contiguous_values, contiguous_indices, self, dim, keepdim);
        if (!values_match) {
            npu_utils::format_fresh_view(values, contiguous_values);
        }
        if (!indices_match) {
            npu_utils::format_fresh_view(indices, contiguous_indices);
        }
    } else {
        median_out_value_nocheck(values, indices, self, dim, keepdim);
    }
    return std::tuple<at::Tensor &, at::Tensor &>(values, indices);
}

at::Tensor median(const at::Tensor &self)
{
    at::Tensor result =
        npu_preparation::apply_tensor_with_format({}, self.options(), npu_preparation::get_tensor_npu_format(self));
    median_out_nocheck(result, self);
    return result;
}

std::tuple<at::Tensor, at::Tensor> median(const at::Tensor &self, int64_t dim, bool keepdim)
{
    auto output_size = median_npu_output_size(self, dim, keepdim);
    at::Tensor values = npu_preparation::apply_tensor_with_format(output_size, self.options(),
                                                                  npu_preparation::get_tensor_npu_format(self));
    at::Tensor indices =
        npu_preparation::apply_tensor_with_format(output_size, self.options().dtype(at::kLong), ACL_FORMAT_NCHW);
    median_out_value_nocheck(values, indices, self, dim, keepdim);
    return std::tuple<at::Tensor &, at::Tensor &>(values, indices);
}

#if VERSION_BETWEEN(V1R11, V1R11) || VERSION_BETWEEN(V2R1, VERSION_NEWEST)
at::Tensor nanmedian(const at::Tensor &self)
{
    TORCH_NPU_WARN_ONCE(
        "Warning: kernel [nanmedian] is not supported by NPU currently. Now this kernel is running on CPU.");
    at::Tensor self_cpu = self.to("cpu");
    auto result = at::native::nanmedian_cpu(self_cpu);
    at::Tensor output = result.to(self.device());
    return output;
}

std::tuple<at::Tensor, at::Tensor> nanmedian(const at::Tensor &self, int64_t dim, bool keepdim)
{
    TORCH_WARN_ONCE(
        "Warning: kernel [nanmedian.dim] is not supported by NPU currently. Now this kernel is running on CPU.");
    auto output_size = median_npu_output_size(self, dim, keepdim);
    at::Tensor values = npu_preparation::apply_tensor_with_format(output_size, self.options(),
                                                                  npu_preparation::get_tensor_npu_format(self));
    at::Tensor indices =
        npu_preparation::apply_tensor_with_format(output_size, self.options().dtype(at::kLong), ACL_FORMAT_NCHW);

    auto self_cpu = self.cpu();
    auto values_cpu = values.cpu();
    auto indices_cpu = indices.cpu();
    auto result = at::native::nanmedian_out_cpu(self_cpu, dim, keepdim, values_cpu, indices_cpu);
    at::Tensor values_out = values_cpu.to(self.device());
    at::Tensor indices_out = indices_cpu.to(self.device());
    return std::tuple<at::Tensor &, at::Tensor &>(values_out, indices_out);
}
#endif
} // namespace acl_op

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
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
void index_fill_d_check_index(at::IntArrayRef shape, const at::Tensor &index, int64_t dim)
{
    TORCH_CHECK(index.dim() == 1, "Index should be a one-dimensional tensor"
        + OPS_ERROR(ErrCode::PARAM));
    int index_temp = INT_MAX;
    for (int i = 0; i < index.sizes()[0]; i++) {
        index_temp = static_cast<int>(op_plugin::utils::get_scalar_float_value(index[i].item()));
        TORCH_CHECK(shape[dim] > index_temp, "Index out of range, it should be in [0,", shape[dim], ")"
        + OPS_ERROR(ErrCode::VALUE));
    }
}

c10::SmallVector<float, N> index_fill_d_assist_help_init(int64_t dim, at::IntArrayRef sizes, std::vector<int> index,
                                                         bool flag, float value)
{
    int blocksize = 0;
    int blocknum = 1;
    int n = 1;

    for (uint i = 0; i < sizes.size(); i++) {
        if (i <= dim) {
            blocknum *= sizes[i];
        }
        n *= sizes[i];
    }
    blocksize = n / blocknum;

    c10::SmallVector<float, N> ast;
    ast.resize(n);

    if (flag) {
        ast = c10::SmallVector<float, N>(n, 1);
    } else {
        ast = c10::SmallVector<float, N>(n, 0);
    }

    for (uint i = 0; i < index.size(); i++) {
        int start = 0;
        int end = 0;
        int idx = index[i];
        int k = idx;
        int count = 0;
        while (k < blocknum) {
            start = blocksize * k;
            end = start + blocksize;
            for (int j = start; j < end; j++) {
                ast[j] = value;
            }
            count++;
            k = idx + sizes[dim] * count;
        }
    }
    return ast;
}

at::Tensor index_fill_d_assist_help(const at::Tensor &self, const at::Tensor &index, int64_t dim, at::Scalar value,
                                    bool flag)
{
    c10::SmallVector<float, N> assist;
    at::IntArrayRef size = self.sizes();
    std::vector<int> index_vector;
    for (int i = 0; i < index.sizes()[0]; i++) {
        int index_temp = static_cast<int>(op_plugin::utils::get_scalar_float_value(index[i].item()));
        index_vector.push_back(index_temp);
    }
    // input
    // index is a 1-D tensor
    // value is a tensor which has only one item
    float value_float = op_plugin::utils::get_scalar_float_value(value);
    assist = index_fill_d_assist_help_init(dim, size, index_vector, flag, value_float);
    at::Tensor assist_help = at::from_blob(assist.data(), size, dtype(at::ScalarType::Float));
    return assist_help.to(at::device(torch_npu::utils::get_npu_device_type()));
}

at::Tensor &index_fill_d_nocheck(at::Tensor &result, const at::Tensor &self, int64_t dim, const at::Tensor &index,
                                 at::Scalar value)
{
    // Special case
    // There is a zero in shape
    // example : shape = [1,3,4,0] return itself else return
    // processed_data(result)
    if (self.numel() == 0) {
        return result;
    }
    at::Scalar value_zeros = at::Scalar(0.0);
    const at::Tensor *aclInput = &self;
    at::Tensor assist_help1 = index_fill_d_assist_help(self, index, dim, value_zeros, true);
    at::Tensor assist_help2 = index_fill_d_assist_help(self, index, dim, value, false);
    at::ScalarType self_type = self.scalar_type();
    assist_help1 = assist_help1.scalar_type() == self_type ?
                       assist_help1 :
                       at_npu::native::custom_ops::npu_dtype_cast(assist_help1, self_type);
    assist_help2 = assist_help2.scalar_type() == self_type ?
                       assist_help2 :
                       at_npu::native::custom_ops::npu_dtype_cast(assist_help2, self_type);

    at_npu::native::OpCommand cmd;
    cmd.Name("IndexFillD").Input(self).Input(assist_help1).Input(assist_help2).Attr("dim", dim).Output(result).Run();
    return result;
}
} // namespace

at::Tensor index_fill(const at::Tensor &self, int64_t dim, const at::Tensor &index, const at::Scalar &value)
{
    at::Tensor result = npu_preparation::apply_tensor(self);
    index_fill_d_nocheck(result, self, dim, index, value);
    return result;
}

at::Tensor &index_fill_(at::Tensor &self, int64_t dim, const at::Tensor &index, const at::Scalar &value)
{
    at::IntArrayRef shape_self = self.sizes();
    index_fill_d_check_index(shape_self, index, dim);

    if (!npu_utils::check_match(&self)) {
        at::Tensor contiguous_self = npu_utils::format_contiguous(self);
        index_fill_d_nocheck(contiguous_self, contiguous_self, dim, index, value);
        npu_utils::format_fresh_view(self, contiguous_self);
    } else {
        index_fill_d_nocheck(self, self, dim, index, value);
    }
    return self;
}

at::Tensor index_fill(const at::Tensor &self, int64_t dim, const at::Tensor &index, const at::Tensor &value)
{
    at::IntArrayRef shape_self = self.sizes();
    index_fill_d_check_index(shape_self, index, dim);
    TORCH_CHECK(value.dim() == 0, "Value should be a 0-dimensional tensor,but got ", value.dim(),
        OPS_ERROR(ErrCode::PARAM));
    at::Scalar value_scalar = value.item();
    at::Tensor result = npu_preparation::apply_tensor(self);
    index_fill_d_nocheck(result, self, dim, index, value_scalar);
    return result;
}

at::Tensor &index_fill_(at::Tensor &self, int64_t dim, const at::Tensor &index, const at::Tensor &value)
{
    at::IntArrayRef shape_self = self.sizes();
    index_fill_d_check_index(shape_self, index, dim);
    TORCH_CHECK(value.dim() == 0, "Value should be a 0-dimensional tensor,but got ", value.dim(),
        OPS_ERROR(ErrCode::PARAM));
    at::Scalar value_scalar = value.item();

    if (!npu_utils::check_match(&self)) {
        at::Tensor contiguous_self = npu_utils::format_contiguous(self);
        index_fill_d_nocheck(contiguous_self, contiguous_self, dim, index, value_scalar);
        npu_utils::format_fresh_view(self, contiguous_self);
    } else {
        index_fill_d_nocheck(self, self, dim, index, value_scalar);
    }
    return self;
}
} // namespace acl_op

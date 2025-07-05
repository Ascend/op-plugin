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

#include <bitset>
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "op_plugin/utils/AdvancedIndex.h"
#include "op_plugin/utils/OpUtils.h"
#include "op_plugin/utils/KernelNpuOutputSize.h"

namespace op_infer {
using tuple_array_vector = std::tuple<c10::IntArrayRef, c10::IntArrayRef, c10::SmallVector<int64_t, SIZE>>;
using tuple_vector = std::tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>>;
using tuple_vectors =
    std::tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>>;
using small_vector = c10::SmallVector<int64_t, SIZE>;
using int_array_ref_list = std::tuple<c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef>;

const int DIM_4D = 4;
const int DIM_5D = 5;

// Integer division rounding to -Infinity
template <typename T>
static inline T div_rtn(T x, T y)
{
    if (y == 0) {
        AT_ERROR("div_rtn: Division by zero!");
    }
    int q = x / y;
    int r = x % y;
    if ((r != 0) && ((r < 0) != (y < 0))) {
        --q;
    }
    return q;
}

int64_t CeilDiv(int64_t value, int64_t factor)
{
    int64_t value_num = 0;
    if (factor == 0) {
        return value_num;
    }
    if (value % factor == 0) {
        value_num = value / factor;
    } else {
        value_num = value / factor + 1;
    }

    return value_num;
}

int64_t make_wrap_dim(int64_t dim, int64_t dim_post_expr)
{
    // this will make range [-1, 0]
    if (dim_post_expr <= 0) {
        dim_post_expr = 1;
    }

    if (dim < 0) {
        dim += dim_post_expr;
    }

    return dim;
}

std::bitset<64> make_dim_mask(c10::IntArrayRef dims, int64_t ndim)
{
    std::bitset<64> mask = std::bitset<64>();
    if (ndim <= 0) {
        ndim = 1;
    }
    if (dims.empty()) {
        mask.flip();
    } else {
        for (int64_t dim : dims) {
            int64_t positive_dim = make_wrap_dim(dim, ndim);
            int64_t min = ndim * -1;
            int64_t max = ndim - 1;
            TORCH_CHECK(min <= dim && dim <= max, "Dimension out of range (expected to be in range of [",
                min, ", ", max, "], but got ", dim, ")", OPS_ERROR(ErrCode::PARAM));
            mask.set(positive_dim);
        }
    }

    return mask;
}

c10::SmallVector<int64_t, SIZE> array_to_small_vector(c10::IntArrayRef shape)
{
    c10::SmallVector<int64_t, SIZE> shape_small_vec;
    for (uint64_t i = 0; i < shape.size(); i++) {
        shape_small_vec.emplace_back(shape[i]);
    }

    return shape_small_vec;
}

c10::IntArrayRef input_same_output_size(const at::Tensor &input)
{
    return input.sizes();
}

c10::SmallVector<int64_t, SIZE> broadcast_ops_npu_output_size(c10::IntArrayRef shape1_, c10::IntArrayRef shape2_)
{
    return c10::SmallVector<int64_t, SIZE>(at::infer_size(shape1_, shape2_));
}

c10::SmallVector<int64_t, SIZE> broadcast_ops_npu_output_size(const at::Tensor &self, const at::Tensor &other)
{
    return broadcast_ops_npu_output_size(self.sizes(), other.sizes());
}

c10::SmallVector<int64_t, SIZE> reduce_ops_npu_output_size(const at::Tensor &self, c10::IntArrayRef dim, bool keepdim)
{
    int64_t ndim = self.dim();
    std::bitset<64> mask = make_dim_mask(dim, ndim);
    auto shape = array_to_small_vector(self.sizes());
    for (int dim = static_cast<int64_t>(shape.size()) - 1; dim >= 0; dim--) {
        if (mask[dim]) {
            if (keepdim) {
                shape[dim] = 1;
            } else {
                shape.erase(shape.begin() + dim);
            }
        }
    }

    return shape;
}

c10::SmallVector<int64_t, SIZE> mse_loss_npu_output_size(const at::Tensor &self, const at::Tensor &target,
                                                         int64_t reduction)
{
    auto shape = broadcast_ops_npu_output_size(self, target);
    if (reduction == at::Reduction::None) {
        return shape;
    } else {
        c10::SmallVector<int64_t, SIZE> output_size;
        for (uint64_t i = 1; i < shape.size(); i++) {
            output_size.emplace_back(shape[i]);
        }
        return output_size;
    }
}

c10::SmallVector<int64_t, SIZE> adaptive_avg_pool3d_npu_output_size(const at::Tensor &self,
                                                                    c10::IntArrayRef output_size)
{
    for (const auto i : c10::irange(1, self.ndimension())) {
        TORCH_CHECK(
            self.size(i) > 0,
            "adaptive_avg_pool3d(): Expected input to have non-zero size for non-batch dimensions, "
            "but input has sizes ",
            self.sizes(),
            " with dimension ",
            i,
            " being "
            "empty",
            OPS_ERROR(ErrCode::PARAM));
    }

    TORCH_CHECK(
        (self.ndimension() == DIM_4D || self.ndimension() == DIM_5D),
        "adaptive_avg_pool3d(): Expected 4D or 5D tensor, but got ",
        self.sizes(), OPS_ERROR(ErrCode::PARAM));

    TORCH_CHECK(output_size.size() > 2, "output_size length should greater than 2, "
        "but got the output_size length is ", output_size.size(), OPS_ERROR(ErrCode::PARAM));

    auto shape = array_to_small_vector(self.sizes());
    auto iter = shape.rbegin();
    *iter = output_size[2];
    *(iter + 1) = output_size[1];
    *(iter + 2) = output_size[0];
    return shape;
}

c10::SmallVector<int64_t, SIZE> addmm_npu_output_size(const at::Tensor &self, const at::Tensor &mat1,
                                                      const at::Tensor &mat2)
{
    return broadcast_ops_npu_output_size(self.sizes(), {mat1.size(0), mat2.size(1)});
}

c10::SmallVector<int64_t, SIZE> addbmm_npu_output_size(const at::Tensor &self, const at::Tensor &batch1,
                                                       const at::Tensor &batch2)
{
    return broadcast_ops_npu_output_size(self.sizes(), {batch1.size(1), batch2.size(2)});
}

c10::SmallVector<int64_t, SIZE> addmv_npu_output_size(const at::Tensor &self, const at::Tensor &mat)
{
    return broadcast_ops_npu_output_size(self.sizes(), {mat.size(0)});
}

c10::SmallVector<int64_t, SIZE> addr_npu_output_size(const at::Tensor &self, const at::Tensor &vec1,
                                                     const at::Tensor &vec2)
{
    return broadcast_ops_npu_output_size(self.sizes(), {vec1.size(0), vec2.size(0)});
}

c10::SmallVector<int64_t, SIZE> avg_pool2d_npu_output_size(const at::Tensor &self, c10::IntArrayRef kernel_size,
                                                           c10::IntArrayRef stride, c10::IntArrayRef padding,
                                                           bool ceil_mode)
{
    TORCH_CHECK(self.dim() == 3 || self.dim() == 4, "tensor self's dimension must be 3 or 4",
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(kernel_size.size() == 2, "kernel_size length should be 2", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(stride.size() == 2, "stride length should be 2", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(stride[0] * stride[1] != 0, "stride should not contain zero", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(padding.size() == 2, "padding length should be 2", OPS_ERROR(ErrCode::PARAM));

    int self_h = self.size(-2);
    int self_w = self.size(-1);

    int64_t kernel_h = ceil_mode ? (CeilDiv(self_h + 2 * padding[0] - kernel_size[0], stride[0]) + 1) :
                                   ((self_h + 2 * padding[0] - kernel_size[0]) / stride[0] + 1);
    int64_t kernel_w = ceil_mode ? (CeilDiv(self_w + 2 * padding[1] - kernel_size[1], stride[1]) + 1) :
                                   ((self_w + 2 * padding[1] - kernel_size[1]) / stride[1] + 1);
    TORCH_CHECK(kernel_h > 0, "kernel_h has to be positive, but got ", kernel_h, OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(kernel_w > 0, "kernel_w has to be positive, but got ", kernel_w, OPS_ERROR(ErrCode::VALUE));

    if (ceil_mode) {
        if ((kernel_h - 1) * stride[0] >= self_h + padding[0]) {
            --kernel_h;
        }

        if ((kernel_w - 1) * stride[1] >= self_w + padding[1]) {
            --kernel_w;
        }
    }

    c10::SmallVector<int64_t, SIZE> output_size;
    if (self.dim() == 3) {
        output_size = {self.size(0), kernel_h, kernel_w};
    } else {
        output_size = {self.size(0), self.size(1), kernel_h, kernel_w};
    }

    return output_size;
}

c10::SmallVector<int64_t, SIZE> avg_pool3d_npu_output_size(const at::Tensor &self, c10::IntArrayRef kernel_size,
                                                           c10::IntArrayRef stride, c10::IntArrayRef padding,
                                                           bool ceil_mode)
{
    TORCH_CHECK(self.dim() == 4 || self.dim() == 5, "tensor self's dimension must be 4 or 5",
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(kernel_size.size() == 3, "kernel_size length should be 3", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(stride.size() == 3, "stride length should be 3", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(stride[0] * stride[1] * stride[2] != 0, "stride should not contain zero", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(padding.size() == 3, "padding length should be 3", OPS_ERROR(ErrCode::PARAM));

    int self_d = self.size(-3);
    int self_h = self.size(-2);
    int self_w = self.size(-1);

    int64_t kernel_d = ceil_mode ? (CeilDiv(self_d + 2 * padding[0] - kernel_size[0], stride[0]) + 1) :
                                   ((self_d + 2 * padding[0] - kernel_size[0]) / stride[0] + 1);
    int64_t kernel_h = ceil_mode ? (CeilDiv(self_h + 2 * padding[1] - kernel_size[1], stride[1]) + 1) :
                                   ((self_h + 2 * padding[1] - kernel_size[1]) / stride[1] + 1);
    int64_t kernel_w = ceil_mode ? (CeilDiv(self_w + 2 * padding[2] - kernel_size[2], stride[2]) + 1) :
                                   ((self_w + 2 * padding[2] - kernel_size[2]) / stride[2] + 1);
    TORCH_CHECK(kernel_d > 0, "kernel_d has to be positive, but got ", kernel_d, OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(kernel_h > 0, "kernel_h has to be positive, but got ", kernel_h, OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(kernel_w > 0, "kernel_w has to be positive, but got ", kernel_w, OPS_ERROR(ErrCode::VALUE));

    if (ceil_mode) {
        if ((kernel_d - 1) * stride[0] >= self_d + padding[0]) {
            --kernel_d;
        }

        if ((kernel_h - 1) * stride[1] >= self_h + padding[1]) {
            --kernel_h;
        }

        if ((kernel_w - 1) * stride[2] >= self_w + padding[2]) {
            --kernel_w;
        }
    }

    c10::SmallVector<int64_t, SIZE> output_size;
    if (self.dim() == 4) {
        output_size = {self.size(0), kernel_d, kernel_h, kernel_w};
    } else {
        output_size = {self.size(0), self.size(1), kernel_d, kernel_h, kernel_w};
    }

    return output_size;
}

small_vector avg_pool2d_backward_npu_output_size(const at::Tensor &self)
{
    TORCH_CHECK(self.dim() == 3 || self.dim() == 4, "tensor self's dimension must be 3 or 4",
        OPS_ERROR(ErrCode::PARAM));
    c10::SmallVector<int64_t, SIZE> output_size;
    if (self.dim() == 3) {
        output_size = {self.size(0), self.size(1), self.size(2)};
    } else {
        output_size = {self.size(0), self.size(1), self.size(2), self.size(3)};
    }
    return output_size;
}

c10::SmallVector<int64_t, SIZE> baddbmm_npu_output_size(const at::Tensor &self, const at::Tensor &mat2)
{
    TORCH_CHECK(self.dim() > 1, "tensor self's dimension must be greater than 1, "
        "but got Tensor of dimension ", self.dim(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(mat2.dim() > 2, "tensor mat2's dimension must be greater than 2, "
        "but got Tensor of dimension ", mat2.dim(), OPS_ERROR(ErrCode::PARAM));

    return {self.size(0), self.size(1), mat2.size(2)};
}

c10::SmallVector<int64_t, SIZE> cdist_npu_output_size(const at::Tensor &x1, const at::Tensor &x2)
{
    int64_t r1 = x1.size(-2);
    int64_t r2 = x2.size(-2);
    int64_t dim1 = static_cast<int64_t>(x1.dim());
    int64_t dim2 = static_cast<int64_t>(x2.dim());
    TORCH_CHECK(dim1 >= 2, "Dim of x1 should be grater than 2, but now is ", dim1, OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dim2 >= 2, "Dim of x2 should be grater than 2, but now is ", dim2, OPS_ERROR(ErrCode::PARAM));
    c10::IntArrayRef batch_tensor1(x1.sizes().data(), dim1 - 2);
    c10::IntArrayRef batch_tensor2(x2.sizes().data(), dim2 - 2);
    c10::SmallVector<int64_t, SIZE> expand_batch_portion(at::infer_size(batch_tensor1, batch_tensor2));
    c10::SmallVector<int64_t, SIZE> output_shape(expand_batch_portion);
    output_shape.insert(output_shape.end(), {r1, r2});
    return output_shape;
}

c10::SmallVector<int64_t, SIZE> conv1d_npu_output_size(const at::Tensor &input, const at::Tensor &weight,
                                                       c10::IntArrayRef padding, c10::IntArrayRef stride,
                                                       c10::IntArrayRef dilation)
{
    TORCH_CHECK(input.dim() > 2, "tensor input's dimension must be greater than 2, "
        "but got Tensor of dimension ", input.dim(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(weight.dim() > 2, "tensor weight's dimension must be greater than 2, "
        "but got Tensor of dimension ", weight.dim(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(stride.size() > 0, "stride length should be greater than 0, "
        "but got the stride length is ", stride.size(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(stride[0] != 0, "stride should not contain zero", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(padding.size() > 0, "padding length should be greater than 0, "
        "but got the padding length is ", padding.size(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dilation.size() > 0, "dilation length should be greater than 0, "
        "but got the dilation length is ", dilation.size(), OPS_ERROR(ErrCode::PARAM));

    int64_t N_ = input.size(0);
    int64_t L = input.size(2);
    int64_t C_out = weight.size(0);
    C_out = (weight.size(1) != 0) ? C_out : 0;

    auto kernel_size = weight.sizes().slice(2);
    int64_t L_out = (L + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1;
    c10::SmallVector<int64_t, SIZE> output_size = {N_, C_out, L_out};
    return output_size;
}

c10::SmallVector<int64_t, SIZE> conv2d_npu_output_size(const at::Tensor &input, const at::Tensor &weight,
                                                       c10::IntArrayRef padding, c10::IntArrayRef stride,
                                                       c10::IntArrayRef dilation)
{
    TORCH_CHECK(input.dim() > 3, "tensor input's dimension must be greater than 3, "
        "but got Tensor of dimension ", input.dim(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(weight.dim() > 3, "tensor weight's dimension must be greater than 3, "
        "but got Tensor of dimension ", weight.dim(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(stride.size() > 1, "stride length should be greater than 1, "
        "but got the stride length is ", stride.size(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(padding.size() > 1, "padding length should be greater than 1, "
        "but got the padding length is ", padding.size(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dilation.size() > 1, "dilation length should be greater than 1, "
        "but got the dilation length is ", dilation.size(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(stride[0] * stride[1] != 0, "Stride cannot contain 0" + OPS_ERROR(ErrCode::PARAM));

    int64_t N_ = input.size(0);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    int64_t C_out = weight.size(0);

    auto kernel_size = weight.sizes().slice(2);

    int64_t H_out = (H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1;
    int64_t W_out = (W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1;
    c10::SmallVector<int64_t, SIZE> output_size = {N_, C_out, H_out, W_out};
    return output_size;
}

c10::SmallVector<int64_t, SIZE> conv_transpose1d_npu_output_size(const at::Tensor &input, const at::Tensor &weight,
                                                                 c10::IntArrayRef padding,
                                                                 c10::IntArrayRef output_padding,
                                                                 c10::IntArrayRef stride, c10::IntArrayRef dilation,
                                                                 int64_t groups)
{
    TORCH_CHECK(input.dim() > 2, "tensor input's dimension must be greater than or equal to 2, "
        "but got Tensor of dimension ", input.dim(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(weight.dim() > 2, "tensor weight's dimension must be greater than or equal to 2, "
        "but got Tensor of dimension ", weight.dim(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(stride.size() > 0, "stride length should be greater than 0, "
        "but got the stride length is ", stride.size(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(padding.size() > 0, "padding length should be greater than 0, "
        "but got the padding length is ", padding.size(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dilation.size() > 0, "dilation length should be greater than 0, "
        "but got the dilation length is ", dilation.size(), OPS_ERROR(ErrCode::PARAM));

    int64_t N_ = input.size(0);
    int64_t L = input.size(2);
    int64_t C_out = weight.size(1) * groups;
    C_out = (weight.size(0) != 0) ? C_out : 0;

    auto kernel_size = weight.sizes().slice(2);

    int64_t L_out = (L - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1;
    c10::SmallVector<int64_t, SIZE> output_size = {N_, C_out, L_out};
    return output_size;
}

c10::SmallVector<int64_t, SIZE> conv3d_npu_output_size(const at::Tensor &input, const at::Tensor &weight,
                                                       c10::IntArrayRef padding,
                                                       c10::IntArrayRef output_padding, c10::IntArrayRef stride,
                                                       c10::IntArrayRef dilation, int64_t groups, bool transposed)
{
    TORCH_CHECK(input.dim() >= 5, "input has to be more than 5D, but got Tensor of dimension ", input.dim(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(weight.dim() >= 5, "weight has to be more than 5D, but got Tensor of dimension ", weight.dim(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(padding.size() >= 3, "padding has to contain more than 3 elements, but got ", padding.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(stride.size() >= 3, "stride has to contain more than 3 elements, but got ", stride.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dilation.size() >= 3, "dilation has to contain more than 3 elements, but got ", dilation.size(),
        OPS_ERROR(ErrCode::PARAM));
    int64_t N_ = input.size(0);
    int64_t D = input.size(2);
    int64_t H = input.size(3);
    int64_t W = input.size(4);
    int64_t Co;
    int64_t Do;
    int64_t Ho;
    int64_t Wo;

    if (!transposed) {
        TORCH_CHECK(stride[0] * stride[1] * stride[2] != 0, "Stride cannot contain 0" + OPS_ERROR(ErrCode::PARAM));
        Co = weight.size(0);
        auto kernel_size = weight.sizes().slice(2);
        Do = (D + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1;
        Ho = (H + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1;
        Wo = (W + 2 * padding[2] - dilation[2] * (kernel_size[2] - 1) - 1) / stride[2] + 1;
    } else {
        Co = weight.size(1) * groups;
        auto kernel_size = weight.sizes().slice(2);
        Do = (D - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1;
        Ho = (H - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + output_padding[1] + 1;
        Wo = (W - 1) * stride[2] - 2 * padding[2] + dilation[2] * (kernel_size[2] - 1) + output_padding[2] + 1;
    }
    TORCH_CHECK(Do > 0, "Do has to be positive, but got ", Do, OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(Ho > 0, "Ho has to be positive, but got ", Ho, OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(Wo > 0, "Wo has to be positive, but got ", Wo, OPS_ERROR(ErrCode::VALUE));

    c10::SmallVector<int64_t, SIZE> output_size = {N_, Co, Do, Ho, Wo};
    return output_size;
}

c10::SmallVector<int64_t, SIZE> conv_npu_output_size(const at::Tensor &input, const at::Tensor &weight,
                                                     const c10::optional<at::Tensor> &bias, c10::IntArrayRef padding,
                                                     c10::IntArrayRef output_padding, c10::IntArrayRef stride,
                                                     c10::IntArrayRef dilation, int64_t groups, bool transposed)
{
    int64_t dim = weight.ndimension() - 2; // Subtract nonspatial dimensions: 2
    if (!transposed) {
        if (dim == 1) {
            return conv1d_npu_output_size(input, weight, padding, stride, dilation);
        } else if (dim == 2) {
            return conv2d_npu_output_size(input, weight, padding, stride, dilation);
        } else {
            return conv3d_npu_output_size(input, weight, padding, output_padding, stride,
                                          dilation, groups, transposed);
        }
    } else {
        const at::Tensor &bias_tensor = c10::value_or_else(bias, [] { return at::Tensor(); });
        if (dim == 1) {
            return conv_transpose1d_npu_output_size(input, weight, padding, output_padding, stride,
                                                    dilation, groups);
        } else if (dim == 2) {
            // input dim = 2
            if (input.ndimension() == 3) {
                c10::SmallVector<int64_t, SIZE> unsqueeze_size = {1, input.size(0), input.size(1), input.size(2)};
                input.resize_(unsqueeze_size);
            }
            return conv_transpose2d_npu_output_size(input, weight, padding, output_padding, stride,
                                                    dilation, groups);
        } else {
            return conv3d_npu_output_size(input, weight, padding, output_padding, stride,
                                          dilation, groups, transposed);
        }
    }
}

std::tuple<c10::IntArrayRef, c10::IntArrayRef, c10::SmallVector<int64_t, SIZE>>
conv2d_backward_npu_output_size(const at::Tensor &input, const at::Tensor &grad, const at::Tensor &weight)
{
    c10::SmallVector<int64_t, SIZE> gradBiasSize = {grad.size(1)};
    // input dim = 3, grad dim = 3, weight dim = 4
    if (input.ndimension() == 3 && grad.ndimension() == 3 && weight.ndimension() == 4) {
        c10::SmallVector<int64_t, SIZE> input_unsqueeze_size = {1, input.size(0), input.size(1), input.size(2)};
        c10::SmallVector<int64_t, SIZE> grad_unsqueeze_size = {1, grad.size(0), grad.size(1), grad.size(2)};
        input.resize_(input_unsqueeze_size);
        grad.resize_(grad_unsqueeze_size);
        gradBiasSize = {grad.size(1)};
    }
    return std::tuple<c10::IntArrayRef, c10::IntArrayRef, c10::SmallVector<int64_t, SIZE>>(
        input.sizes(), weight.sizes(), gradBiasSize);
}

std::tuple<c10::IntArrayRef, c10::IntArrayRef, c10::SmallVector<int64_t, SIZE>>
conv2d_backward_tbc_output_size(const at::Tensor &input, const at::Tensor &grad, const at::Tensor &weight)
{
    c10::SmallVector<int64_t, SIZE> gradBiasSize = {grad.size(2)};
    return std::tuple<c10::IntArrayRef, c10::IntArrayRef, c10::SmallVector<int64_t, SIZE>>(
        input.sizes(), weight.sizes(), gradBiasSize);
}

c10::SmallVector<int64_t, SIZE> cosine_similarity_npu_output_size(const at::Tensor &x1, int64_t dim, bool keepdim)
{
    c10::IntArrayRef dims(dim);
    return reduce_ops_npu_output_size(x1, dims, keepdim);
}

tuple_array_vector conv_transpose2d_backward_npu_output_size(const at::Tensor &input, const at::Tensor &grad_output,
                                                             const at::Tensor &weight)
{
    TORCH_CHECK(grad_output.dim() > 1, "tensor grad_output's dimension must be greater than 1, "
        "but got Tensor of dimension ", grad_output.dim(), OPS_ERROR(ErrCode::PARAM));

    c10::SmallVector<int64_t, SIZE> gradBiasSize = {grad_output.size(1)};
    return std::tuple<c10::IntArrayRef, c10::IntArrayRef, c10::SmallVector<int64_t, SIZE>>(
        input.sizes(), weight.sizes(), gradBiasSize);
}

c10::SmallVector<int64_t, SIZE> conv_transpose2d_npu_output_size(const at::Tensor &input, const at::Tensor &weight,
                                                                 c10::IntArrayRef padding,
                                                                 c10::IntArrayRef output_padding,
                                                                 c10::IntArrayRef stride, c10::IntArrayRef dilation,
                                                                 int64_t groups)
{
    TORCH_CHECK(input.dim() > 3, "tensor input's dimension must be greater than 3, "
        "but got Tensor of dimension ", input.dim(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(weight.dim() > 3, "tensor weight's dimension must be greater than 3, "
        "but got Tensor of dimension ", weight.dim(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(stride.size() > 1, "stride length should be greater than 1, "
        "but got the stride length is ", stride.size(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(padding.size() > 1, "padding length should be greater than 1, "
        "but got the padding length is ", padding.size(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dilation.size() > 1, "dilation length should be greater than 1, "
        "but got the dilation length is ", dilation.size(), OPS_ERROR(ErrCode::PARAM));

    int64_t N_ = input.size(0);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    int64_t Co = weight.size(1) * groups;
    auto kernel_size = weight.sizes().slice(2);

    int64_t Ho = (H - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1;
    int64_t Wo = (W - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + output_padding[1] + 1;

    c10::SmallVector<int64_t, SIZE> outputSize = {N_, Co, Ho, Wo};

    return outputSize;
}

c10::SmallVector<int64_t, SIZE> deformable_conv2d_npu_output_size(const at::Tensor &input, const at::Tensor &offset,
                                                                  c10::IntArrayRef kernel_size)
{
    int64_t No = input.size(0);
    int64_t Co = input.size(1);
    int64_t Ho = offset.size(2) * kernel_size[0];
    int64_t Wo = offset.size(3) * kernel_size[1];

    c10::SmallVector<int64_t, SIZE> outputSize = {No, Co, Ho, Wo};

    return outputSize;
}

std::tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>>
ctc_loss_npu_output_size(const at::Tensor &log_probs, int64_t max_length)
{
    const int64_t dim_num_two = 2;
    int64_t time_size = log_probs.size(0);
    int64_t batch_size = log_probs.size(1);

    if (log_probs.dim() == dim_num_two) {
        batch_size = 1;
    }
    c10::SmallVector<int64_t, SIZE> neg_log_likelihood_size = {batch_size};
    int64_t alpha_tail_size = 2 * max_length + 1;
    // Apply for a 32 byte aligned space to avoid address shifting in the OP.
    int64_t alpha_tail_size_align = (alpha_tail_size + 7) / 8 * 8;
    c10::SmallVector<int64_t, SIZE> log_alpha_size = {batch_size, time_size, alpha_tail_size_align};

    if (log_probs.dim() == dim_num_two) {
        return std::tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>>(at::ArrayRef<int64_t>(),
                                                                                            log_alpha_size);
    }

    return std::tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>>(neg_log_likelihood_size,
                                                                                        log_alpha_size);
}

c10::SmallVector<int64_t, SIZE> dot_npu_output_size()
{
    c10::SmallVector<int64_t, SIZE> outputSize = {1};
    return outputSize;
}

c10::SmallVector<int64_t, SIZE> embedding_dense_backward_npu_output_size(const at::Tensor &grad_output,
                                                                         int64_t num_weights)
{
    return {num_weights, grad_output.size(-1)};
}

int_array_ref_list layer_norm_backward_npu_output_size(const at::Tensor &X, const at::Tensor &gamma)
{
    return std::tuple<c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef>(X.sizes(), gamma.sizes(), gamma.sizes());
}

static bool hasContiguousSubspace(at::TensorList tl)
{
    // true if all the non-null tensors are adjacent
    auto isDefined = [](const at::Tensor &tensor) { return tensor.defined(); };
    auto isNull = [](const at::Tensor &tensor) { return !tensor.defined(); };
    auto start = std::find_if(tl.begin(), tl.end(), isDefined);
    auto stop = std::find_if(tl.rbegin(), tl.rend(), isDefined);
    auto it = std::find_if(start, stop.base(), isNull);
    return it == stop.base();
}

std::vector<at::Tensor> index_expand_outplace(at::TensorList to_expand)
{
    // expands a list of Tensors; ignores undefined (null) tensors
    bool first = true;
    std::vector<int64_t> sizes;
    for (size_t i = 0; i < to_expand.size(); ++i) {
        if (!to_expand[i].defined()) {
            continue;
        } else if (first) {
            sizes = to_expand[i].sizes().vec();
            first = false;
        } else {
            sizes = at::infer_size(sizes, to_expand[i].sizes());
        }
    }
    std::vector<at::Tensor> result(to_expand.size());
    for (size_t i = 0; i < to_expand.size(); ++i) {
        if (!to_expand[i].defined()) {
            continue;
        } else if (to_expand[i].sizes().equals(sizes)) {
            result[i] = to_expand[i];
        } else {
            result[i] = to_expand[i].expand(sizes, true);
        }
    }
    return result;
}

c10::SmallVector<int64_t, SIZE> index_reshape(std::vector<at::Tensor> end_indices, int64_t dims_before,
                                              int64_t dims_after)
{
    c10::SmallVector<int64_t, SIZE> index_shape;
    for (auto &index : end_indices) {
        if (index.defined()) {
            auto shape = at::DimVector();
            shape.append(dims_before, 1);
            shape.append(index.sizes().begin(), index.sizes().end());
            shape.append(dims_after, 1);
            if (index_shape.empty()) {
                index_shape = shape;
            } else if (index_shape != shape) {
                index_shape = at::infer_size(index_shape, shape);
            }
        }
    }
    return index_shape;
}

c10::SmallVector<int64_t, SIZE> index_npu_output_size(const at::Tensor &self, at::TensorList indices)
{
    std::vector<at::Tensor> mid_indices = index_expand_outplace(indices);

    while (mid_indices.size() < static_cast<size_t>(self.dim())) {
        mid_indices.emplace_back();
    }
    at::Tensor src = self;
    std::vector<at::Tensor> end_indices = mid_indices;
    if (!hasContiguousSubspace(mid_indices)) {
        end_indices.clear();
        std::tie(src, end_indices) = at::native::transposeToFront(self, mid_indices);
    }

    int64_t dims_before = 0;
    int64_t dims_after = 0;
    int64_t dims_indexed = 0;
    c10::SmallVector<int64_t, SIZE> replacement_shape;
    at::DimVector indexed_sizes;
    for (size_t dim = 0; dim < end_indices.size(); dim++) {
        if (!end_indices[dim].defined()) {
            if (dims_indexed == 0) {
                dims_before++;
            } else {
                dims_after++;
            }
        } else {
            dims_indexed++;
            replacement_shape = end_indices[dim].sizes();
            indexed_sizes.push_back(src.size(dim));
        }
    }
    if (std::find(indexed_sizes.begin(), indexed_sizes.end(), 0) != indexed_sizes.end() &&
        std::find(replacement_shape.begin(), replacement_shape.end(), 0) == replacement_shape.end()) {
        TORCH_CHECK_INDEX(false, "index is out of bounds for dimension with size 0", OPS_ERROR(ErrCode::PARAM));
    }
    auto self_shape = at::DimVector(src.sizes());
    int64_t end = dims_before + dims_indexed;
    self_shape.erase(self_shape.begin() + dims_before, self_shape.begin() + end);
    self_shape.insert(self_shape.begin() + dims_before, replacement_shape.begin(), replacement_shape.end());

    c10::SmallVector<int64_t, SIZE> index_shape = index_reshape(end_indices, dims_before, dims_after);
    c10::SmallVector<int64_t, SIZE> outputSize = index_shape;
    if (index_shape != self_shape) {
        outputSize = at::infer_size(index_shape, self_shape);
    }
    return outputSize;
}

c10::SmallVector<int64_t, SIZE> index_select_npu_output_size(const at::Tensor &self, int64_t dim,
                                                             const at::Tensor &index)
{
    at::Tensor indexTmp(index);
    if (indexTmp.ndimension() == 0) {
        indexTmp = index.unsqueeze(0);
    }
    int64_t indexSize = indexTmp.size(0);

    int64_t selfDim = self.ndimension() > 0 ? self.ndimension() : 1;
    bool dim_valid = dim >= -selfDim && dim < selfDim;
    TORCH_CHECK(dim_valid, "Dimension out of range (expected to be in range of [", -selfDim, ", ", selfDim - 1,
                "], but got ", dim, ")", OPS_ERROR(ErrCode::PARAM));
    if (dim < 0) {
        dim += selfDim;
    }

    c10::SmallVector<int64_t, SIZE> outputSize;
    for (int64_t i = 0; i < static_cast<int64_t>(self.sizes().size()); ++i) {
        if (i == dim) {
            outputSize.push_back(indexSize);
        } else {
            outputSize.push_back(self.size(i));
        }
    }

    return outputSize;
}

c10::SmallVector<int64_t, SIZE> nnpack_spatial_convolution_npu_output_size(const at::Tensor &input,
                                                                           const at::Tensor &weight,
                                                                           c10::IntArrayRef padding,
                                                                           c10::IntArrayRef stride)
{
    TORCH_CHECK(input.dim() >= 4, "The input should be at least 4D, but got: ", input.dim(), "D",
        OPS_ERROR(ErrCode::PARAM));
    int64_t N_ = input.size(0);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    int64_t Co = weight.size(0);
    auto kernel_size = weight.sizes().slice(2);

    int64_t Ho = 0;
    int64_t Wo = 0;
    if (padding.size() == 1 && stride.size() == 1) {
        TORCH_CHECK(stride[0] != 0, "stride should not contain zero", OPS_ERROR(ErrCode::PARAM));
        Ho = (H + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1;
        Wo = (W + 2 * padding[0] - (kernel_size[1] - 1) - 1) / stride[0] + 1;
    }
    if (padding.size() != 1 && stride.size() == 1) {
        TORCH_CHECK(stride[0] != 0, "stride should not contain zero", OPS_ERROR(ErrCode::PARAM));
        Ho = (H + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1;
        Wo = (W + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[0] + 1;
    }
    if (padding.size() != 1 && stride.size() != 1) {
        TORCH_CHECK(stride[0] * stride[1] != 0, "stride should not contain zero", OPS_ERROR(ErrCode::PARAM));
        Ho = (H + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1;
        Wo = (W + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1;
    }
    c10::SmallVector<int64_t, SIZE> outputSize = {N_, Co, Ho, Wo};
    return outputSize;
}

tuple_vectors nms_with_mask_npu_output_size(const at::Tensor &self)
{
    c10::SmallVector<int64_t, SIZE> boxesSize = {self.size(0), 5};
    c10::SmallVector<int64_t, SIZE> idxSize = {
        self.size(0),
    };
    c10::SmallVector<int64_t, SIZE> maskSize = {
        self.size(0),
    };

    return std::tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>,
                      c10::SmallVector<int64_t, SIZE>>(boxesSize, idxSize, maskSize);
};

c10::SmallVector<int64_t, SIZE> nonzero_npu_max_output_size(const at::Tensor &self)
{
    int64_t selfNumEl = self.numel();
    int64_t selfDim = self.dim();
    at::SmallVector<int64_t, SIZE> maxOutputSize;
    if (selfNumEl == 1 && selfDim == 0) {
        if (self.is_nonzero()) {
            maxOutputSize = {1, 0};
        } else {
            maxOutputSize = {0, 0};
        }
    } else {
        maxOutputSize = {selfNumEl, selfDim};
    }
    return maxOutputSize;
}

c10::SmallVector<int64_t, SIZE> prelu_backward_npu_grad_weight_output_size(const at::Tensor &weight)
{
    int64_t weight_num = weight.numel();
    if (weight_num == 1) {
        return array_to_small_vector(weight.sizes());
    }

    c10::SmallVector<int64_t, SIZE> output_size = {weight_num};
    return output_size;
}

c10::SmallVector<int64_t, SIZE> pad_npu_output_size(const at::Tensor &input, c10::IntArrayRef paddings)
{
    c10::SmallVector<int64_t, SIZE> outputSize;
    for (uint64_t i = 0; i < static_cast<uint64_t>(input.dim()); i++) {
        if (i * 2 + 1 < paddings.size()) {
            outputSize.emplace_back(input.size(i) + paddings[i * 2] + paddings[i * 2 + 1]);
        } else if (i * 2 < paddings.size()) {
            outputSize.emplace_back(input.size(i) + paddings[i * 2]);
        } else {
            outputSize.emplace_back(input.size(i));
        }
    }
    return outputSize;
}

c10::SmallVector<int64_t, SIZE> pdist_npu_output_size(const at::Tensor &self)
{
    c10::SmallVector<int64_t, SIZE> outputSize;
    int64_t n = self.size(0);
    int64_t resultSize = n * (n - 1) / 2;
    outputSize.emplace_back(resultSize);
    return outputSize;
}

c10::SmallVector<int64_t, SIZE> prod_npu_output_size(const at::Tensor &self, int64_t dim, bool keepdim)
{
    c10::IntArrayRef dims(dim);
    return reduce_ops_npu_output_size(self, dims, keepdim);
}

c10::SmallVector<int64_t, SIZE> prod_npu_output_size(const at::Tensor &self, bool keepdim)
{
    c10::IntArrayRef dims;
    return reduce_ops_npu_output_size(self, dims, keepdim);
}

c10::SmallVector<int64_t, SIZE> reflection_pad1d_npu_out_size(const at::Tensor &self, at::IntArrayRef padding)
{
    uint64_t padding_num = padding.size();
    int64_t self_num = self.dim();
    TORCH_CHECK(padding_num == 2, "padding length should be 2", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(self_num == 2 || self_num == 3, "self should be 2D or 3D", OPS_ERROR(ErrCode::PARAM));
    // 0, 1, -2, -1 are indexes
    int64_t padding_l = padding[0];
    int64_t padding_r = padding[1];
    int64_t C = self.size(-2);
    int64_t W = self.size(-1);
    int64_t Wo = W + padding_l + padding_r;
    c10::SmallVector<int64_t, SIZE> output_size = {C, Wo};
    // 3 is dim
    if (self_num == 3) {
        // -3 is index
        int64_t N_ = self.size(-3);
        output_size = {N_, C, Wo};
    }
    return output_size;
}

c10::SmallVector<int64_t, SIZE> reflection_pad2d_npu_out_size(const at::Tensor &self, at::IntArrayRef padding)
{
    uint64_t padding_num = padding.size();
    int64_t self_num = self.dim();
    TORCH_CHECK(padding_num == 4, "padding length should be 4", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(self_num == 3 || self_num == 4, "self should be 3D or 4D", OPS_ERROR(ErrCode::PARAM));
    // -3, -2, -1, 0, 1, 2, 3 are indexes
    int64_t padding_l = padding[0];
    int64_t padding_r = padding[1];
    int64_t padding_t = padding[2];
    int64_t padding_b = padding[3];
    int64_t C = self.size(-3);
    int64_t H = self.size(-2);
    int64_t W = self.size(-1);
    int64_t Ho = H + padding_t + padding_b;
    int64_t Wo = W + padding_l + padding_r;
    c10::SmallVector<int64_t, SIZE> output_size = {C, Ho, Wo};
    // 4 is dim
    if (self_num == 4) {
        // -4 is index
        int64_t N_ = self.size(-4);
        output_size = {N_, C, Ho, Wo};
    }
    return output_size;
}

c10::SmallVector<int64_t, SIZE> conv_depthwise2d_npu_output_size(const at::Tensor &self, const at::Tensor &weight,
                                                                 at::IntArrayRef kernel_size, at::IntArrayRef stride,
                                                                 at::IntArrayRef padding, at::IntArrayRef dilation)
{
    int64_t self_num = self.dim();
    int64_t weight_num = weight.dim();
    uint64_t kernel_size_num = kernel_size.size();
    uint64_t stride_num = stride.size();
    uint64_t padding_num = padding.size();
    uint64_t dilation_num = dilation.size();
    TORCH_CHECK(self_num == 4 && weight_num == 4, "self and weight should be 4D", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(kernel_size_num == 2 && stride_num == 2 && padding_num == 2 && dilation_num == 2,
                "Attr length should be 2", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(kernel_size == weight.sizes().slice(2), "kernel size should be equal to the last 2 dim of weight",
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(stride[0] * stride[1] != 0, "stride should not contain zero", OPS_ERROR(ErrCode::PARAM));

    int64_t N_ = self.size(0);
    int64_t Co = weight.size(0);
    int64_t H = self.size(2);
    int64_t W = self.size(3);
    int64_t Ho = (H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1;
    int64_t Wo = (W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1;
    c10::SmallVector<int64_t, SIZE> output_size = {N_, Co, Ho, Wo};
    return output_size;
}

c10::SmallVector<int64_t, SIZE> reflection_pad3d_npu_out_size(const at::Tensor &self, at::IntArrayRef padding)
{
    uint64_t padding_num = padding.size();
    int64_t self_num = self.dim();
    // 6 is padding length
    TORCH_CHECK(padding_num == 6, "padding length should be 6", OPS_ERROR(ErrCode::PARAM));
    // 4 and 5 are dim number of self
    TORCH_CHECK(self_num == 4 || self_num == 5, "self should be 4D or 5D", OPS_ERROR(ErrCode::PARAM));
    // -4, -3, -2, -1, 0, 1, 2, 3, 4, 5 are indexes of self and padding
    int64_t padding_l = padding[0];
    int64_t padding_r = padding[1];
    int64_t padding_t = padding[2];
    int64_t padding_b = padding[3];
    int64_t padding_f = padding[4];
    int64_t padding_back = padding[5];
    int64_t C = self.size(-4);
    int64_t D = self.size(-3);
    int64_t H = self.size(-2);
    int64_t W = self.size(-1);
    int64_t Do = D + padding_f + padding_back;
    int64_t Ho = H + padding_t + padding_b;
    int64_t Wo = W + padding_l + padding_r;
    c10::SmallVector<int64_t, SIZE> output_size = {C, Do, Ho, Wo};
    // 5 means self format is NCDHW
    if (self_num == 5) {
        // 0 is the first index of self
        int64_t N_ = self.size(0);
        output_size = {N_, C, Do, Ho, Wo};
    }
    return output_size;
}

c10::SmallVector<int64_t, SIZE> repeat_interleave_npu_output_size(const at::Tensor &self, int64_t repeats, int64_t dim)
{
    c10::SmallVector<int64_t, SIZE> shape;
    if (dim < 0) {
        dim = dim + self.dim();
    }
    for (int64_t i = 0; i < self.dim(); i++) {
        if (i == dim) {
            shape.emplace_back(self.size(i) * repeats);
        } else {
            shape.emplace_back(self.size(i));
        }
    }
    return shape;
}

c10::SmallVector<int64_t, SIZE> repeat_interleave_npu_output_size(const at::Tensor &self, const at::Tensor &repeats,
                                                                  int64_t dim)
{
    c10::SmallVector<int64_t, SIZE> shape;
    if (dim < 0) {
        dim = dim + self.dim();
    }
    for (int64_t i = 0; i < self.dim(); i++) {
        if (i == dim) {
            if (repeats.numel() == 1) {
                shape.emplace_back(repeats.item().toLong() * self.size(i));
            } else {
                shape.emplace_back(repeats.sum().item().toLong());
            }
        } else {
            shape.emplace_back(self.size(i));
        }
    }
    return shape;
}

c10::SmallVector<int64_t, SIZE> replication_pad1d_npu_out_size(const at::Tensor &self, at::IntArrayRef padding)
{
    uint64_t padding_num = padding.size();
    int64_t self_num = self.dim();
    TORCH_CHECK(padding_num == 2, "padding length should be 2", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(self_num == 2 || self_num == 3, "self should be 2D or 3D", OPS_ERROR(ErrCode::PARAM));
    // 0, 1, -2, -1 are indexes
    int64_t padding_l = padding[0];
    int64_t padding_r = padding[1];
    int64_t C = self.size(-2);
    int64_t W = self.size(-1);
    int64_t Wo = W + padding_l + padding_r;
    c10::SmallVector<int64_t, SIZE> output_size = {C, Wo};
    // 3 is dim
    if (self_num == 3) {
        // -3 is index
        int64_t N_ = self.size(-3);
        output_size = {N_, C, Wo};
    }
    return output_size;
}

c10::SmallVector<int64_t, SIZE> replication_pad2d_npu_output_size(const at::Tensor &self, c10::IntArrayRef padding)
{
    TORCH_CHECK(self.dim() >= 3, "The self is expected to be at least 3D, but got: ", self.dim(), "D",
        OPS_ERROR(ErrCode::PARAM));
    int64_t N_ = self.dim() == 3 ? 1 : self.size(-4);
    int64_t C = self.size(-3);
    int64_t H = self.size(-2);
    int64_t W = self.size(-1);
    int64_t padding_l = 0;
    int64_t padding_r = 0;
    int64_t padding_t = 0;
    int64_t padding_b = 0;
    if (!padding.empty() && padding.size() == 1) {
        padding_l = padding[0];
        padding_r = padding[0];
        padding_t = padding[0];
        padding_b = padding[0];
    } else if (!padding.empty() && 4 == padding.size()) {
        padding_l = padding[0];
        padding_r = padding[1];
        padding_t = padding[2];
        padding_b = padding[3];
    }
    int64_t Ho = H + padding_t + padding_b;
    int64_t Wo = W + padding_l + padding_r;

    c10::SmallVector<int64_t, SIZE> outputSize = {N_, C, Ho, Wo};
    return outputSize;
}

c10::SmallVector<int64_t, SIZE> replication_pad2d_npu_out_size(const at::Tensor &self, at::IntArrayRef padding)
{
    uint64_t padding_num = padding.size();
    int64_t self_num = self.dim();
    TORCH_CHECK(padding_num == 4, "padding length should be 4", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(self_num == 3 || self_num == 4, "self should be 3D or 4D", OPS_ERROR(ErrCode::PARAM));
    // -3, -2, -1, 0, 1, 2, 3 are indexes
    int64_t padding_l = padding[0];
    int64_t padding_r = padding[1];
    int64_t padding_t = padding[2];
    int64_t padding_b = padding[3];
    int64_t C = self.size(-3);
    int64_t H = self.size(-2);
    int64_t W = self.size(-1);
    int64_t Ho = H + padding_t + padding_b;
    int64_t Wo = W + padding_l + padding_r;
    c10::SmallVector<int64_t, SIZE> output_size = {C, Ho, Wo};
    // 4 is dim
    if (self_num == 4) {
        // -4 is index
        int64_t N_ = self.size(-4);
        output_size = {N_, C, Ho, Wo};
    }
    return output_size;
}

c10::SmallVector<int64_t, SIZE> replication_pad3d_npu_out_size(const at::Tensor &self, at::IntArrayRef padding)
{
    uint64_t padding_num = padding.size();
    int64_t self_num = self.dim();
    // 6 is padding length
    TORCH_CHECK(padding_num == 6, "padding length should be 6", OPS_ERROR(ErrCode::PARAM));
    // 4 and 5 are dim number of self
    TORCH_CHECK(self_num == 4 || self_num == 5, "self should be 4D or 5D", OPS_ERROR(ErrCode::PARAM));
    // -4, -3, -2, -1, 0, 1, 2, 3, 4, 5 are indexes of self and padding
    int64_t padding_l = padding[0];
    int64_t padding_r = padding[1];
    int64_t padding_t = padding[2];
    int64_t padding_b = padding[3];
    int64_t padding_f = padding[4];
    int64_t padding_back = padding[5];
    int64_t C = self.size(-4);
    int64_t D = self.size(-3);
    int64_t H = self.size(-2);
    int64_t W = self.size(-1);
    int64_t Do = D + padding_f + padding_back;
    int64_t Ho = H + padding_t + padding_b;
    int64_t Wo = W + padding_l + padding_r;
    c10::SmallVector<int64_t, SIZE> output_size = {C, Do, Ho, Wo};
    // 5 means self format is NCDHW
    if (self_num == 5) {
        // 0 is the first index of self
        int64_t N_ = self.size(0);
        output_size = {N_, C, Do, Ho, Wo};
    }
    return output_size;
}

std::tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>>
nms_v4_npu_output_size(c10::Scalar max_output_size)
{
    c10::SmallVector<int64_t, SIZE> selected_indices = {max_output_size.toInt()};
    c10::SmallVector<int64_t, SIZE> valid_outputs = {};
    return std::tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>>(selected_indices,
                                                                                        valid_outputs);
}

c10::SmallVector<int64_t, SIZE> im2col_backward_npu_output_size(const at::Tensor &grad_output,
                                                                const at::IntArrayRef &input_size,
                                                                const at::IntArrayRef &kernel_size)
{
    TORCH_CHECK((grad_output.dim() == 2 && grad_output.size(0) != 0 && grad_output.size(1) != 0) ||
                    (grad_output.dim() == 3 && grad_output.size(1) != 0 && grad_output.size(2) != 0),
                "Expected 2D or 3D (batch mode) tensor for gradOutput with possibly 0 batch size and non-zero "
                "dimensions for gradOutput, but got: ",
                grad_output.sizes(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(kernel_size[0] * kernel_size[1] != 0, "kernel_size should not be zero", OPS_ERROR(ErrCode::PARAM));
    c10::SmallVector<int64_t, SIZE> outputSize;
    if (grad_output.dim() == 2) {
        outputSize = {grad_output.size(0) / (kernel_size[0] * kernel_size[1]), input_size[0], input_size[1]};
    } else {
        outputSize = {grad_output.size(0), grad_output.size(1) / (kernel_size[0] * kernel_size[1]), input_size[0],
                      input_size[1]};
    }
    return outputSize;
}

c10::SmallVector<int64_t, SIZE> repeat_npu_output_size(const at::Tensor &self, c10::IntArrayRef repeats)
{
    int64_t num_new_dimensions = static_cast<int64_t>(repeats.size()) - self.dim();
    // Fill num_ new_ Dimensions elements with a value of 1
    c10::SmallVector<int64_t, SIZE> padded_size(num_new_dimensions, 1);
    padded_size.insert(padded_size.end(), self.sizes().begin(), self.sizes().end());
    c10::SmallVector<int64_t, SIZE> target_size(repeats.size());
    for (uint64_t idx = 0; idx < repeats.size(); ++idx) {
        target_size[idx] = padded_size[idx] * repeats[idx];
    }
    return target_size;
}

c10::SmallVector<int64_t, SIZE> soft_margin_loss_npu_output_size(const at::Tensor &self, int64_t reduction)
{
    c10::SmallVector<int64_t, SIZE> outputSize;
    if (reduction == at::Reduction::None) {
        outputSize = input_same_output_size(self);
    } else {
        outputSize = {1};
    }
    return outputSize;
}

c10::SmallVector<int64_t, SIZE> slow_conv_dilated2d_npu_output_size(const at::Tensor &input, const at::Tensor &weight,
                                                                    c10::IntArrayRef stride, c10::IntArrayRef padding,
                                                                    c10::IntArrayRef dilation)
{
    TORCH_CHECK(input.dim() > 3, "tensor input's dimension must be greater than 3, "
        "but got Tensor of dimension ", input.dim(), OPS_ERROR(ErrCode::PARAM));

    int64_t N_ = input.size(0);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    int64_t Co = weight.size(0);
    auto kernel_size = weight.sizes().slice(2);

    int64_t Ho = (H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1;
    int64_t Wo = (W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1;

    c10::SmallVector<int64_t, SIZE> outputSize = {N_, Co, Ho, Wo};

    return outputSize;
}

std::tuple<c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef> slow_conv_dilated2d_backward_npu_output_size(
    const at::Tensor &grad_output, const at::Tensor &self, const at::Tensor &weight)
{
    return std::tuple<c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef>(grad_output.sizes(), self.sizes(),
                                                                            weight.sizes());
}

std::tuple<c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef> slow_conv_transpose2d_backward_npu_output_size(
    const at::Tensor &grad_output, const at::Tensor &self, const at::Tensor &weight)
{
    return std::tuple<c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef>(self.sizes(), weight.sizes(),
                                                                            grad_output.sizes());
}

c10::IntArrayRef smooth_l1_loss_npu_output_size(const at::Tensor &self, int64_t reduction)
{
    c10::IntArrayRef outputSize;
    if (reduction == at::Reduction::None) {
        outputSize = input_same_output_size(self);
    }
    return outputSize;
}

tuple_vector softmax_cross_entropy_with_logits_impl_npu_output_size(const at::Tensor &self)
{
    c10::SmallVector<int64_t, SIZE> resultSize = array_to_small_vector(self.size(0));
    c10::SmallVector<int64_t, SIZE> backpropSize = array_to_small_vector(self.sizes());

    return std::tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>>(resultSize, backpropSize);
}

c10::SmallVector<int64_t, SIZE> sum_npu_output_size(const at::Tensor &self, c10::IntArrayRef dim, bool keepdim)
{
    return reduce_ops_npu_output_size(self, dim, keepdim);
}

c10::SmallVector<int64_t, SIZE> swiglu_backward_infershape(const at::Tensor &x, int64_t dim)
{
    if (dim < 0) {
        dim += static_cast<int64_t>(x.sizes().size());
    }
    TORCH_CHECK(dim < x.sizes().size(), "dim out of range", dim, OPS_ERROR(ErrCode::PARAM));
    auto output_sizes = op_infer::array_to_small_vector(x.sizes());
    output_sizes[dim] /= 2;
    return output_sizes;
}

c10::SmallVector<int64_t, SIZE> topk_npu_output_size(const at::Tensor &self, int64_t k, int64_t dim)
{
    int64_t wrap_dim = make_wrap_dim(dim, self.dim());
    auto shape = array_to_small_vector(self.sizes());
    if (shape.size() > 0) {
        shape[wrap_dim] = k;
    }
    return shape;
}

c10::SmallVector<int64_t, SIZE> transpose_npu_output_size(const at::Tensor &self, c10::IntArrayRef perm)
{
    auto sizes = self.sizes();
    c10::SmallVector<int64_t, SIZE> shape;
    for (uint64_t i = 0; i < perm.size(); i++) {
        shape.emplace_back(sizes[perm[i]]);
    }

    return shape;
}

c10::SmallVector<int64_t, 3> upsample_infershape_with_scale(c10::IntArrayRef input_size,
                                                            c10::optional<c10::IntArrayRef> output_size,
                                                            c10::optional<c10::ArrayRef<double>> scale_factors)
{
    const auto spatial_dimensions = static_cast<int64_t>(input_size.size()) - 2;
    if (output_size) {
        TORCH_CHECK(!scale_factors, "Must specify exactly one of output_size and scale_factors",
            OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(static_cast<int64_t>(output_size->size()) == spatial_dimensions, OPS_ERROR(ErrCode::PARAM));
        return {output_size->data(), output_size->data() + output_size->size()};
    }
    if (scale_factors) {
        TORCH_CHECK(!output_size, "Must specify exactly one of output_size and scale_factors",
            OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(static_cast<int64_t>(scale_factors->size()) == spatial_dimensions, OPS_ERROR(ErrCode::PARAM));
        c10::SmallVector<int64_t, 3> ret;
        for (const auto i : c10::irange(spatial_dimensions)) {
            ret.push_back(static_cast<double>(input_size[i + 2]) * scale_factors.value()[i]);
        }
        return ret;
    }
    TORCH_CHECK(false, "Must specify exactly one of output_size and scale_factors", OPS_ERROR(ErrCode::PARAM));
}

c10::SmallVector<int64_t, SIZE> upsample_bicubic2d_npu_output_size(const at::Tensor &self,
                                                                   c10::IntArrayRef output_size)
{
    TORCH_CHECK(self.dim() == 4, "It is expected input_size equals to 4, but got size ", self.dim(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(output_size.size() == 2, "It is expected output_size equals to 2, but got size ", output_size.size(),
        OPS_ERROR(ErrCode::PARAM));

    int64_t N_ = self.size(0);
    int64_t C = self.size(1);
    int64_t H = output_size[0];
    int64_t W = output_size[1];

    c10::SmallVector<int64_t, SIZE> outputSize = {N_, C, H, W};
    return outputSize;
}

c10::IntArrayRef upsample_bicubic2d_backward_npu_output_size(c10::IntArrayRef input_size)
{
    return input_size;
}

c10::SmallVector<int64_t, SIZE> upsample_bilinear2d_npu_output_size(const at::Tensor &self,
                                                                    c10::IntArrayRef output_size)
{
    TORCH_CHECK(self.dim() > 1, "tensor self's dimension must be greater than 1, "
        "but got Tensor of dimension ", self.dim(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(output_size.size() > 1, "output_size length should be greater than 1, "
        "but got the output_size length is ", output_size.size(), OPS_ERROR(ErrCode::PARAM));

    // the input's dim of upsample_bilinear2d
    int64_t N_ = self.size(0);
    int64_t C = self.size(1);
    int64_t H = output_size[0];
    int64_t W = output_size[1];

    c10::SmallVector<int64_t, SIZE> outputSize = {N_, C, H, W};
    return outputSize;
}

c10::SmallVector<int64_t, SIZE> upsample_linear1d_npu_output_size(const at::Tensor &self, c10::IntArrayRef output_size)
{
    TORCH_CHECK(self.dim() > 1, "tensor self's dimension must be greater than 1, "
        "but got Tensor of dimension ", self.dim(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(output_size.size() > 0, "output_size length should be greater than 0, "
        "but got the output_size length is ", output_size.size(), OPS_ERROR(ErrCode::PARAM));

    int64_t N_ = self.size(0);
    int64_t C = self.size(1);
    int64_t W = output_size[0];

    c10::SmallVector<int64_t, SIZE> outputSize = {N_, C, W};
    return outputSize;
}

c10::SmallVector<int64_t, SIZE> upsample_trilinear3d_npu_output_size(const at::Tensor &input,
                                                                     at::IntArrayRef output_size)
{
    TORCH_CHECK(input.dim() > 1, "tensor input's dimension must be greater than 1, "
        "but got Tensor of dimension ", input.dim(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(output_size.size() == 3, "It is expected output_size equals to 3, but got size ", output_size.size(),
        OPS_ERROR(ErrCode::PARAM));

    int64_t output_depth = output_size[0];
    int64_t output_height = output_size[1];
    int64_t output_width = output_size[2];

    int64_t nbatch = input.size(0);
    int64_t channels = input.size(1);

    c10::SmallVector<int64_t, SIZE> outputSize = {nbatch, channels, output_depth, output_height, output_width};
    return outputSize;
}

c10::SmallVector<int64_t, SIZE> upsample_nearest3d_npu_output_size(const at::Tensor &input, at::IntArrayRef output_size)
{
    TORCH_CHECK(input.dim() > 1, "tensor input's dimension must be greater than 1, "
        "but got Tensor of dimension ", input.dim(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(output_size.size() > 2, "It is expected output_size greater than 2, "
        "but got size ", output_size.size(), OPS_ERROR(ErrCode::PARAM));

    int64_t output_depth = output_size[0];
    int64_t output_height = output_size[1];
    int64_t output_width = output_size[2];

    int64_t nbatch = input.size(0);
    int64_t channels = input.size(1);

    c10::SmallVector<int64_t, SIZE> output_sizes = {nbatch, channels, output_depth, output_height, output_width};

    return output_sizes;
}

c10::SmallVector<int64_t, SIZE> var_npu_output_size(const at::Tensor &self, c10::IntArrayRef dim, bool keepdim)
{
    c10::SmallVector<int64_t, SIZE> outputSize = reduce_ops_npu_output_size(self, dim, keepdim);
    return outputSize;
}

c10::SmallVector<int64_t, SIZE> glu_npu_output_size(const at::Tensor &self, int64_t dim)
{
    dim = make_wrap_dim(dim, self.dim());
    auto shape = array_to_small_vector(self.sizes());
    shape[dim] = shape[dim] / 2;

    return shape;
}

c10::SmallVector<int64_t, SIZE> crop_and_resize_npu_output_size(const at::Tensor &self, at::IntArrayRef box_index,
                                                                at::IntArrayRef crop_size)
{
    TORCH_CHECK(self.dim() == 4, "input x dim must be 4", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(crop_size.size() == 2, "crop_size size must be 2", OPS_ERROR(ErrCode::PARAM));
    int64_t N_ = static_cast<int64_t>(box_index.size());
    int64_t H = crop_size[0];
    int64_t W = crop_size[1];
    int64_t C = self.size(1);

    c10::SmallVector<int64_t, SIZE> outputSize = {N_, C, H, W};
    return outputSize;
}

c10::SmallVector<int64_t, SIZE> decode_jpeg_npu_output_size(at::IntArrayRef image_shape, int64_t channels)
{
    TORCH_CHECK(image_shape.size() == 3, "image_shape size must be 3", OPS_ERROR(ErrCode::PARAM));
    int64_t H = image_shape[0];
    int64_t W = image_shape[1];
    int64_t C = image_shape[2];

    c10::SmallVector<int64_t, SIZE> outputSize;
    if (channels == 0) {
        outputSize = {C, H, W};
    } else {
        outputSize = {channels, H, W};
    }

    return outputSize;
}

// This logic is specially made for stride_add, and will be removed in future version.
c10::SmallVector<int64_t, SIZE> infersize_stride_add(c10::IntArrayRef shape1_, c10::IntArrayRef shape2_)
{
    auto shape1 = op_infer::array_to_small_vector(shape1_);
    auto shape2 = op_infer::array_to_small_vector(shape2_);

    c10::SmallVector<int64_t, SIZE> output_shape;
    if (shape1.size() < shape2.size()) {
        c10::SmallVector<int64_t, SIZE> shapeTemp = shape1;
        shape1 = shape2;
        shape2 = shapeTemp;
    }

    uint64_t shape1_size = shape1.size();
    uint64_t shape2_size = shape2.size();
    for (uint64_t i = 0; i < shape1_size - shape2_size; i++) {
        shape2.insert(shape2.begin(), 1);
    }

    for (uint64_t i = 0; i < shape1_size; i++) {
        if (shape1[i] == 0 || shape2[i] == 0) {
            output_shape.emplace_back(static_cast<int64_t>(0));
        } else {
            output_shape.emplace_back((shape1[i] > shape2[i]) ? shape1[i] : shape2[i]);
        }
    }
    return output_shape;
}

c10::SmallVector<int64_t, SIZE> infersize_affine_grid_generator(at::IntArrayRef size)
{
    c10::SmallVector<int64_t, SIZE> output_size;
    if (size.size() == 4) {
        output_size = {size[0], size[2] * size[3], 2};
    } else {
        TORCH_CHECK(size.size() > 4, "It is expected size greater than 4, "
            "but got size ", size.size(), OPS_ERROR(ErrCode::PARAM));
        output_size = {size[0], size[2] * size[3] * size[4], 3};
    }
    return output_size;
}

c10::SmallVector<int64_t, SIZE> infersize_all(const at::Tensor &self, int64_t dim)
{
    c10::SmallVector<int64_t, SIZE> output_size;
    for (int64_t i = 0; i < self.dim(); i++) {
        if (dim != i) {
            output_size.emplace_back(self.size(i));
        }
    }
    return output_size;
}

c10::SmallVector<int64_t, SIZE> infersize_npu_anchor_response_flags(at::IntArrayRef featmap_size,
                                                                    int64_t num_base_anchors)
{
    int64_t output_value = featmap_size[0] * featmap_size[1] * num_base_anchors;
    c10::SmallVector<int64_t, SIZE> output_size = {output_value};
    return output_size;
}

c10::SmallVector<int64_t, SIZE> infersize_arange(const at::Scalar& start, const at::Scalar& end, const at::Scalar& step,
                                                 at::ScalarType out_type)
{
    int64_t size_value = 0;
    if (out_type == at::kLong) {
        if (step.toLong() != 0) {
            size_value = CeilDiv(end.toLong() - start.toLong(), step.toLong());
        }
    } else {
        if (step.toDouble() != 0) {
            double size_arange = std::ceil(static_cast<double>(end.toDouble() - start.toDouble()) / step.toDouble());
            TORCH_CHECK(size_arange >= 0 && size_arange <= static_cast<double>(std::numeric_limits<int64_t>::max()),
                "invalid size, possible overflow?")
            size_value = static_cast<int64_t>(size_arange);
        }
    }
    c10::SmallVector<int64_t, SIZE> output_size = {size_value};
    return output_size;
}

c10::SmallVector<int64_t, SIZE> image_to_col_npu_output_size(const at::Tensor &self, at::IntArrayRef ksizes,
                                                             at::IntArrayRef strides, at::IntArrayRef dilations,
                                                             at::IntArrayRef pads)
{
    if (ksizes.size() == 1) {
        c10::SmallVector<int64_t, SIZE> kernel_sizes = {ksizes[0], ksizes[0]};
        ksizes = at::IntArrayRef(kernel_sizes);
    }
    small_vector default_size = {1};
    small_vector pads_default_size = {0};
    strides = strides.empty() ? at::IntArrayRef(default_size) : strides;
    if (strides.size() == 1) {
        c10::SmallVector<int64_t, SIZE> stride_sizes = {strides[0], strides[0]};
        strides = at::IntArrayRef(stride_sizes);
    }

    dilations = dilations.empty() ? at::IntArrayRef(default_size) : dilations;
    if (dilations.size() == 1) {
        c10::SmallVector<int64_t, SIZE> dilation_sizes = {dilations[0], dilations[0]};
        dilations = at::IntArrayRef(dilation_sizes);
    }

    pads = pads.empty() ? at::IntArrayRef(pads_default_size) : pads;
    if (pads.size() == 1) {
        c10::SmallVector<int64_t, SIZE> pad_sizes = {pads[0], pads[0]};
        pads = at::IntArrayRef(pad_sizes);
    }

    bool need_squeeze = self.dim() == 3 ? true : false;
    size_t index = need_squeeze ? 1 : 2;
    int64_t out_h = div_rtn<int64_t>((self.size(index) + 2 * pads[0] - (dilations[0] * (ksizes[0] - 1) + 1)),
                                     strides[0]) + 1;
    int64_t out_w = div_rtn<int64_t>((self.size(index + 1) + 2 * pads[1] - (dilations[1] * (ksizes[1] - 1) + 1)),
                                     strides[1]) + 1;
    if (out_h < 1 || out_w < 1) {
        AT_ERROR("The shape (",out_h, ",", out_w, ") of the array calculated by other parameters "
                "must be at least one.");
    }
    small_vector output_size = need_squeeze ? small_vector({self.size(0) * ksizes[0] * ksizes[1], out_h * out_w}) :
                            small_vector({self.size(0), self.size(1) * ksizes[0] * ksizes[1], out_h * out_w});
    return output_size;
}

c10::SmallVector<int64_t, SIZE> clamp_npu_output_size(const at::Tensor &self, const c10::optional<at::Tensor> &min,
                                                      const c10::optional<at::Tensor> &max)
{
    TORCH_CHECK(min.has_value() || max.has_value(), "torch.clamp: At least one of 'min' or 'max' must not be None",
        OPS_ERROR(ErrCode::PARAM));
    if (self.numel() == 0) {
        c10::SmallVector<int64_t, SIZE> empty_sizes;
        for (int64_t i = 0; i < self.dim(); ++i) {
            empty_sizes.push_back(self.size(i));
        }
        return empty_sizes;
    }
    if (min.has_value() && max.has_value()) {
        auto brc_shape_min = broadcast_ops_npu_output_size(self.sizes(), min.value().sizes());
        return broadcast_ops_npu_output_size(brc_shape_min, max.value().sizes());
    }
    if (min.has_value()) {
        return broadcast_ops_npu_output_size(self.sizes(), min.value().sizes());
    }
    return broadcast_ops_npu_output_size(self.sizes(), max.value().sizes());
}

c10::SmallVector<int64_t, SIZE> cat_npu_output_size(c10::SmallVector<at::Tensor, N> &tensors, int64_t dimension)
{
    bool all_skipped = true;
    int64_t n_dims = 0;
    at::Tensor *not_skipped_tensor;
    auto num_inputs = tensors.size();
    auto should_skip = [](const at::Tensor *t) { return t->nbytes() == 0 && t->dim() == 1; };

    for (uint64_t i = 0; i < num_inputs; i++) {
        if (should_skip(static_cast<at::Tensor *>(&tensors[i]))) {
            continue;
        }
        // found a non-empty tensor
        all_skipped = false;
        not_skipped_tensor = static_cast<at::Tensor *>(&tensors[i]);
        n_dims = not_skipped_tensor->dim();
        break;
    }

    if (all_skipped) {
        c10::SmallVector<int64_t, SIZE> size = {0};
        return size;
    }

    // Compute size of the result in the cat dimension
    int64_t cat_dim_size = 0;
    for (uint64_t i = 0; i < num_inputs; i++) {
        at::Tensor *tensor = static_cast<at::Tensor *>(&tensors[i]);
        if (should_skip(tensor)) {
            continue;
        }
        cat_dim_size += tensor->size(dimension);
    }

    c10::SmallVector<int64_t, SIZE> size;
    size.resize(n_dims);
    for (int64_t dim = 0; dim < n_dims; dim++) {
        int64_t result_dim_size = not_skipped_tensor->size(dim);
        if (dim == dimension) {
            result_dim_size = cat_dim_size;
        }
        size[dim] = result_dim_size;
    }
    return size;
}

c10::SmallVector<int64_t, SIZE> max_pool2d_out_size(const at::Tensor &self, at::IntArrayRef output_size)
{
    auto shape = array_to_small_vector(self.sizes());
    if ((self.dim() == 3 || self.dim() == 4) && output_size.size() == 2) {
        shape[shape.size() - 2] = output_size[0];
        shape[shape.size() - 1] = output_size[1];
    }
    return shape;
}

c10::SmallVector<int64_t, SIZE> ger_output_size(const at::Tensor &self, const at::Tensor &vec2)
{
    int64_t outputsize_0 = self.size(0);
    int64_t outputsize_1 = vec2.size(0);
    c10::SmallVector<int64_t, SIZE> output_size = {outputsize_0, outputsize_1};
    return output_size;
}

// infer output shape for int repeats case
c10::SmallVector<int64_t, SIZE> repeat_interleave_npu_output_size_opapi(const at::Tensor &self, int64_t repeats,
                                                                        c10::optional<int64_t> dim)
{
    c10::SmallVector<int64_t, SIZE> shape;
    if (dim.has_value()) {
        int64_t real_dim = dim.value_or(0);
        real_dim = (real_dim < 0) ? (real_dim + self.dim()) : real_dim;
        for (int64_t i = 0; i < self.dim(); i++) {
            if (i == real_dim) {
                shape.emplace_back(repeats * self.size(i));
            } else {
                shape.emplace_back(self.size(i));
            }
        }
    } else {
        shape.emplace_back(repeats * self.numel());
    }
    return shape;
}

// infer output shape for tensor repeats case
small_vector repeat_interleave_npu_output_size_opapi(const at::Tensor &self, const at::Tensor &repeats,
                                                     c10::optional<int64_t> dim, c10::optional<int64_t> output_size)
{
    c10::SmallVector<int64_t, SIZE> shape;
    int64_t output_size_value = output_size.value_or(0);
    if (dim.has_value()) {
        int64_t real_dim = dim.value_or(0);
        real_dim = (real_dim < 0) ? (real_dim + self.dim()) : real_dim;
        for (int64_t i = 0; i < self.dim(); i++) {
            if (i == real_dim) {
                if (output_size_value != 0) {
                    shape.emplace_back(output_size_value);
                } else {
                    // if repeats only has one element, size will be sum(repeats)*self.size(dim). Otherwise is sum(repeats)
                    int64_t arg = 1;
                    if (repeats.numel() == 1) {
                        arg = self.size(real_dim);
                    }
                    shape.emplace_back(arg * (repeats.sum().item()).toLong());
                }
            } else {
                shape.emplace_back(self.size(i));
            }
        }
    } else { // without dim, need flatten
        // if repeats only has one element, size will be sum(repeats) * self.numel(). Otherwise is sum(repeats)
        int64_t base = output_size_value;
        if (base == 0) {
            base = repeats.sum().item().toLong();
            if (repeats.numel() == 1) {
                base *= self.numel();
            }
        }
        shape.emplace_back(base);
    }
    return shape;
}

std::vector<c10::SmallVector<int64_t, SIZE>> rms_norm_npu_output_size(const at::Tensor &self,
                                                                      const at::Tensor &gamma)
{
    TORCH_CHECK(self.dim() >= gamma.dim(), "The gamma shape should not be bigger than self shape.",
        OPS_ERROR(ErrCode::PARAM));
    auto x_shape = array_to_small_vector(self.sizes());
    auto x_dim_num = self.dim();
    auto gamma_dim_num = gamma.dim();
    c10::SmallVector<int64_t, SIZE> rstd_shape;
    for (int64_t i = 0; i < x_dim_num; i++) {
        if (i < x_dim_num - gamma_dim_num) {
            rstd_shape.push_back(x_shape[i]);
        } else {
            rstd_shape.push_back(1);
        }
    }
    std::vector<c10::SmallVector<int64_t, SIZE>> output_size;
    output_size.push_back(x_shape);
    output_size.push_back(rstd_shape);
    return output_size;
}

std::vector<c10::SmallVector<int64_t, SIZE>> rms_norm_grad_npu_output_size(const at::Tensor &self,
                                                                           const at::Tensor &gamma)
{
    auto x_shape = array_to_small_vector(self.sizes());
    auto gamma_shape = array_to_small_vector(gamma.sizes());
    std::vector<c10::SmallVector<int64_t, SIZE>> output_size;
    output_size.push_back(x_shape);
    output_size.push_back(gamma_shape);
    return output_size;
}

c10::SmallVector<int64_t, SIZE> max_pool3d_output_size(const at::Tensor &self, at::IntArrayRef output_size)
{
    c10::SmallVector<int64_t, SIZE> shape = {};
    if (self.dim() == 4) {
        shape = {self.size(0), output_size[0], output_size[1], output_size[2]};
    } else if (self.dim() == 5) {
        shape = {self.size(0), self.size(1), output_size[0], output_size[1], output_size[2]};
    }
    return shape;
}

c10::SmallVector<int64_t, SIZE> diag_output_size(const at::Tensor& self, int64_t diagonal)
{
    c10::SmallVector<int64_t, SIZE> shape;
    if (self.dim() < 2) {
        shape.emplace_back(self.size(0) + std::abs(diagonal));
        shape.emplace_back(self.size(0) + std::abs(diagonal));
        return shape;
    }
    int64_t m = self.size(0);
    int64_t n = self.size(1);
    if (diagonal > 0) {
        shape.emplace_back(std::min(m, n - diagonal));
        // Judge whether the parameter is out of range
        TORCH_CHECK(diagonal <= n,
                    "If the value is 2-dimensional tensor, the diagonal shoule be less than shape.Diagonal is ",
                    diagonal, OPS_ERROR(ErrCode::VALUE));
    } else {
        shape.emplace_back(std::min(m + diagonal, n));
        // Judge whether the parameter is out of range
        TORCH_CHECK(-diagonal <= m,
                    "If the value is 2-dimensional tensor, the diagonal shoule be less than shape.Diagonal is ",
                    diagonal, OPS_ERROR(ErrCode::VALUE));
    }
    return shape;
}

c10::SmallVector<int64_t, SIZE> stack_output_size(at::TensorList tensors, int64_t dim)
{
    dim = op_plugin::utils::make_warp_dim(dim, tensors[0].dim() + 1);
    at::SmallVector<int64_t, SIZE> shape;
    for (int64_t i = 0; i < dim; i++) {
        shape.emplace_back(tensors[0].size(i));
    }
    shape.emplace_back(tensors.size());
    for (int i = dim; i < tensors[0].dim(); i++) {
        shape.emplace_back(tensors[0].size(i));
    }
    return shape;
}

at::SmallVector<int64_t, SIZE> upsample_nearest_exact2d_output_size_npu(
    const at::Tensor &input,
    at::IntArrayRef output_size)
{
    TORCH_CHECK(input.dim() >= 2, "Input's dim must be at least 2.", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(output_size.size() >= 2, "Output size must be at least 2.", OPS_ERROR(ErrCode::PARAM));
    int64_t N_ = input.size(0);
    int64_t C = input.size(1);
    int64_t H = output_size[0];
    int64_t W = output_size[1];
    at::SmallVector<int64_t, SIZE> outputSize = {N_, C, H, W};
    return outputSize;
}

at::SmallVector<int64_t, SIZE> npu_cross_entropy_loss_loss_output_size(
    const at::Tensor &input,
    c10::string_view reduction)
{
    at::SmallVector<int64_t, SIZE> outputSize;
    if (reduction == "none") {
        outputSize = {input.size(0)};
    } else {
        outputSize = {1};
    }
    return outputSize;
}

at::SmallVector<int64_t, SIZE> npu_cross_entropy_loss_zloss_output_size(
    const at::Tensor &input,
    c10::string_view reduction,
    bool return_zloss)
{
    at::SmallVector<int64_t, SIZE> outputSize;
    if (return_zloss) {
        outputSize = op_infer::npu_cross_entropy_loss_loss_output_size(input, reduction);
    } else {
        outputSize = {0};
    }
    return outputSize;
}

at::SmallVector<int64_t, SIZE> npu_cross_entropy_loss_lse_for_zloss_output_size(
    const at::Tensor &input,
    float lse_square_scale_for_zloss)
{
    at::SmallVector<int64_t, SIZE> outputSize;
    if (lse_square_scale_for_zloss != 0.0) {
        outputSize = {input.size(0)};
    } else {
        outputSize = {0};
    }
    return outputSize;
}

c10::SmallVector<int64_t, SIZE> kronecker_quant_out_size(const at::Tensor &self)
{
    auto outputSize = op_infer::array_to_small_vector(self.sizes());
    auto self_dim_num = self.dim();
    TORCH_CHECK(outputSize[self_dim_num - 1] % INT4_NUMS_IN_INT32_SPACE == 0,
        "input shape last dim must be divded by 8" + OPS_ERROR(ErrCode::PARAM));
    outputSize[self_dim_num - 1] /= INT4_NUMS_IN_INT32_SPACE;
    return outputSize;
}

c10::SmallVector<int64_t, SIZE> kronecker_quant_scale_size(const at::Tensor &self)
{
    int64_t resultSize = self.size(0);
    at::SmallVector<int64_t, SIZE> outputSize = {resultSize};
    return outputSize;
}

c10::SmallVector<int64_t, SIZE> matmul_output_size(const at::Tensor &tensor1, const at::Tensor &tensor2)
{
    c10::SmallVector<int64_t, SIZE> output_size;
    auto dim_tensor1 = tensor1.dim();
    auto dim_tensor2 = tensor2.dim();

    TORCH_CHECK(dim_tensor1 > 0 && dim_tensor2 > 0, "matmul got error dimentions: ", "(", dim_tensor1, ", ",
                dim_tensor2, ")", OPS_ERROR(ErrCode::PARAM));

    if (dim_tensor1 == 1 && dim_tensor2 == 1) {
        output_size = {};
    } else if (dim_tensor1 == 2 && dim_tensor2 == 1) {
        output_size = {tensor1.size(0)};
    } else if (dim_tensor1 == 1 && dim_tensor2 == 2) {
        output_size = {tensor2.size(1)};
    } else if (dim_tensor1 == 2 && dim_tensor2 == 2) {
        output_size = {tensor1.size(0), tensor2.size(1)};
    } else if (dim_tensor1 >= 3 && (dim_tensor2 == 1 || dim_tensor2 == 2)) {
        // t1:(N, n, m) * t2:(m, p)
        auto size1 = tensor1.sizes();
        auto tmp = c10::SmallVector<int64_t, SIZE>{tensor2.size(0), 1};
        auto size2 = dim_tensor2 == 1 ? tmp : tensor2.sizes();
        output_size.insert(output_size.end(), size1.begin(), size1.end() - 1);
        if (dim_tensor2 > 1) {
            output_size.push_back(size2[dim_tensor2 - 1]);
        }
    } else if ((dim_tensor1 == 1 || dim_tensor1 == 2) && dim_tensor2 >= 3) {
        auto tmp = c10::SmallVector<int64_t, SIZE>{1, tensor1.size(0)};
        auto size1 = dim_tensor1 == 1 ? tmp : tensor1.sizes();
        auto size2 = tensor2.sizes();
        output_size.insert(output_size.end(), size2.begin(), size2.end() - 2);
        if (dim_tensor1 > 1) {
            output_size.push_back(size1[0]);
        }
        output_size.push_back(size2[dim_tensor2 - 1]);
    } else if (dim_tensor1 >= 3 && dim_tensor2 >= 3) {
        // t1:(b1, n, m1) * t2:(x2, m2, p)
        int64_t n = tensor1.size(-2);
        at::IntArrayRef batch_tensor1(tensor1.sizes().data(), dim_tensor1 - 2);
        int64_t p = tensor2.size(-1);
        at::IntArrayRef batch_tensor2(tensor2.sizes().data(), dim_tensor2 - 2);
        std::vector<int64_t> expand_batch_portion = at::infer_size(batch_tensor1, batch_tensor2);
        c10::SmallVector<int64_t, SIZE> output_expand_size(expand_batch_portion);
        output_expand_size.insert(output_expand_size.end(), {n, p});
        output_size = output_expand_size;
    } else {
        TORCH_CHECK(false, "matmul got error sizes: ", "(", dim_tensor1, ", ", dim_tensor2, ")", OPS_ERROR(ErrCode::PARAM));
    }

    return output_size;
}

c10::SmallVector<int64_t, SIZE> npu_transpose_batchmatmul_output_size(const at::Tensor &input, const at::Tensor &weight, const at::Tensor &scale_real,
                                                                      at::IntArrayRef perm_x1_real, at::IntArrayRef perm_x2_real, at::IntArrayRef perm_y_real,
                                                                      int32_t batch_split_factor_value)
{
    c10::SmallVector<int64_t, SIZE> output_size;
    auto input_dim_num = input.dim();
    auto weight_dim_num = weight.dim();
    constexpr int EXPECTED_DIM = 3;

    TORCH_CHECK(input_dim_num == EXPECTED_DIM && weight_dim_num == EXPECTED_DIM,
                "input dim is ", input_dim_num, "but expected is ", EXPECTED_DIM,
                "weight dim is ", weight_dim_num, "but expected is ", EXPECTED_DIM, OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(((input.scalar_type() == at::ScalarType::Half) || (input.scalar_type() == at::ScalarType::BFloat16) ||
                 (input.scalar_type() == at::ScalarType::Float)),
                "input's type supported for float16, float32 and bfloat16." + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(((weight.scalar_type() == at::ScalarType::Half) || (weight.scalar_type() == at::ScalarType::BFloat16) ||
                 (weight.scalar_type() == at::ScalarType::Float)),
                "weight's type supported for float16, float32 and bfloat16." + OPS_ERROR(ErrCode::PARAM));

    auto check_perm_x1 = (perm_x1_real[0] == 0 && perm_x1_real[1] == 1 && perm_x1_real[2] == 2) ||
                         (perm_x1_real[0] == 1 && perm_x1_real[1] == 0 && perm_x1_real[2] == 2);
    TORCH_CHECK(check_perm_x1, "perm_x1 should be [0, 1, 2] or [1, 0, 2]" + OPS_ERROR(ErrCode::PARAM));
    auto check_perm_x2 = perm_x2_real[0] == 0 && perm_x2_real[1] == 1 && perm_x2_real[2] == 2;
    TORCH_CHECK(check_perm_x2, "perm_x2 should be [0, 1, 2]" + OPS_ERROR(ErrCode::PARAM));
    auto check_perm_y = perm_y_real[0] == 1 && perm_y_real[1] == 0 && perm_y_real[2] == 2;
    TORCH_CHECK(check_perm_y, "perm_y should be [1, 0, 2]" + OPS_ERROR(ErrCode::PARAM));
    auto m_dim = input.size(perm_x1_real[1]);
    auto batch_dim = input.size(perm_x1_real[0]);
    auto n_dim = weight.size(perm_x2_real[2]);

    output_size = {m_dim, batch_dim, n_dim};
    if (scale_real.defined()) {
        output_size = {m_dim, 1, batch_dim * n_dim};
    }
    if (batch_split_factor_value > 1) {
        output_size = {batch_split_factor_value, m_dim, batch_dim * n_dim / batch_split_factor_value};
    }
    return output_size;
}

c10::SmallVector<int64_t, SIZE> npu_group_quant_out_size(const at::Tensor& x, c10::optional<at::ScalarType> dst_dtype)
{
    at::ScalarType dst_type = c10::value_or_else(dst_dtype, [] {return at::ScalarType::Char;});
    c10::SmallVector<int64_t, SIZE> output_shape = op_infer::array_to_small_vector(x.sizes());
    if (dst_type == at::ScalarType::QUInt4x2) {
        auto x_dim_num = x.dim();
        TORCH_CHECK(output_shape[x_dim_num - 1] % INT4_NUMS_IN_INT32_SPACE == 0,
                    "input shape last dim must be divded by 8" + OPS_ERROR(ErrCode::PARAM));
        output_shape[x_dim_num - 1] /= INT4_NUMS_IN_INT32_SPACE;
    }

    return output_shape;
}

c10::SmallVector<int64_t, SIZE> npu_gather_sparse_index_out_size(const at::Tensor& input, const at::Tensor& index)
{
    c10::SmallVector<int64_t, SIZE> output_shape;
    if (input.dim() == 0) {
        output_shape = op_infer::array_to_small_vector(input.sizes());
        return output_shape;
    }

    int64_t npu_tensor_dim_limit = 8;
    TORCH_CHECK((input.dim() + index.dim() - 1 <= npu_tensor_dim_limit),
                "input.dim() + index.dim() - 1 must not greater than 8." + OPS_ERROR(ErrCode::PARAM));
    auto size_input = input.sizes();
    auto size_index = index.sizes();
    output_shape.insert(output_shape.end(), size_index.begin(), size_index.end());
    output_shape.insert(output_shape.end(), size_input.begin() + 1, size_input.end());

    return output_shape;
}

c10::SmallVector<int64_t, SIZE> npu_nsa_compress_out_size(const at::Tensor& input, c10::optional<int64_t> actual_seq_len_type, at::OptionalIntArrayRef actual_seq_len, int64_t compress_block_size, int64_t compress_stride)
{
    int64_t compress_kv_num = 0;
    int64_t pre_seqlen = 0;
    c10::SmallVector<int64_t, SIZE> output_shape;
    auto actual_seq_len_type_value = actual_seq_len_type.value_or(0);
    if (actual_seq_len_type_value == 0) {
        auto actual_seq_len_value = actual_seq_len.value();
        for (size_t i = 0; i < actual_seq_len_value.size(); i++) {
            int64_t cur_seq_len = actual_seq_len_value[i] - pre_seqlen;
            if (cur_seq_len >= compress_block_size) {
                TORCH_CHECK(compress_stride > 0, "compress_stride must be greater than 0." + OPS_ERROR(ErrCode::VALUE));
                compress_kv_num += (cur_seq_len - compress_block_size + compress_stride) / compress_stride;
            }
            pre_seqlen += cur_seq_len;
        }
        output_shape.push_back(compress_kv_num);
        output_shape.push_back(input.size(NPU_NSA_COMPRESS_INPUT_DIM_SECOND));
        output_shape.push_back(input.size(NPU_NSA_COMPRESS_INPUT_DIM_THIRD));
    }
    return output_shape;
}

c10::SmallVector<int64_t, SIZE> npu_nsa_select_attention_infer_out_size(const at::Tensor& query, const at::Tensor& value, int64_t head_num, int64_t key_value_head_num, c10::string_view layout)
{
    std::string input_layout = std::string(layout);
    TORCH_CHECK(input_layout == "BSH" || input_layout == "BSND", "layout only support BSH or BSND now.", OPS_ERROR(ErrCode::PARAM));

    at::SmallVector<int64_t, SIZE> output_size;
    if (input_layout == "BSH") {
        TORCH_CHECK(key_value_head_num >0, "key_value_head_num must be greater than 0." + OPS_ERROR(ErrCode::VALUE));
        auto key_head_dim = value.size(DIM_2) / key_value_head_num;
        output_size = {query.size(DIM_0), query.size(DIM_1), head_num * key_head_dim};
    } else {
        output_size = {query.size(DIM_0), query.size(DIM_1), query.size(DIM_2), value.size(DIM_3)};
    }

    return output_size;
}


c10::SmallVector<int64_t, SIZE> npu_moe_token_permute_out_size(const at::Tensor &tokens, const at::Tensor &indices, c10::optional<int64_t> num_out_tokens)
{
    TORCH_CHECK(tokens.dim() == DIM_2,
                "The dims of input tokens should be 2 dimensional, but got ", tokens.dim(), "-dimensional." + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(indices.dim() == DIM_1 || indices.dim() == DIM_2,
                "The dims of input indices should be 2 or 1 dimensional, but got ", indices.dim(), "-dimensional." + OPS_ERROR(ErrCode::PARAM));
    int64_t num_out_tokens_value = num_out_tokens.value_or(0);
    int64_t flatten_size = indices.numel();
    int64_t actual_num_out_tokens = (num_out_tokens_value > 0) ? std::min(num_out_tokens_value, flatten_size) : num_out_tokens_value + flatten_size;
    c10::SmallVector<int64_t, SIZE> output_shape;
    output_shape = {actual_num_out_tokens, tokens.size(1)};
    return output_shape;
}

c10::SmallVector<int64_t, SIZE> npu_moe_token_unpermute_out_size(const at::Tensor& permuted_tokens, const at::Tensor &sorted_indices, const c10::optional<at::Tensor>& probs)
{
    const static int64_t DEFAULT_TOPK = 1;
    if (probs.has_value()) {
            TORCH_CHECK(probs.value().dim() == DIM_2,
                        "The dims of input probs should be 2 dimensional, but got ", probs.value().dim(), "-dimensional." + OPS_ERROR(ErrCode::PARAM));
    }
    TORCH_CHECK(permuted_tokens.dim() == DIM_2,
                "The dims of input permuted_tokens should be 2 dimensional, but got ", permuted_tokens.dim(), "-dimensional." + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(sorted_indices.dim() == DIM_1,
                "The dims of input sorted_indices should be 1 dimensional, but got ", sorted_indices.dim(), "-dimensional." + OPS_ERROR(ErrCode::PARAM));
    
    int64_t topk = probs.has_value() ? probs.value().size(1) : DEFAULT_TOPK;
    c10::SmallVector<int64_t, SIZE> output_shape;
    output_shape = {sorted_indices.size(0) / topk, permuted_tokens.size(-1)};
    return output_shape;
}

} // namespace op_infer

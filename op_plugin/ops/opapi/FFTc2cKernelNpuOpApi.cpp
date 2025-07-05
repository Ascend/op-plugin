// Copyright (c) 2024 Huawei Technologies Co., Ltd
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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/custom_functions/opapi/FFTCommonOpApi.h"
namespace op_api {
#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
using npu_preparation = at_npu::native::OpPreparation;
enum class fft_norm_mode {
    none,       // No normalization
    by_root_n,  // Divide by sqrt(signal_size)
    by_n,       // Divide by signal_size
};
enum class fft_mode {
    c2c,
    r2c,
    c2r,
};

double _fft_normalization_scale(int64_t normalization, at::IntArrayRef sizes, at::IntArrayRef dims)
{
    auto norm = static_cast<fft_norm_mode>(normalization);
    if (norm == fft_norm_mode::none) {
        return 1.0;
    }

    int64_t signal_numel = 1;
    for (auto dim : dims) {
        signal_numel *= sizes[dim];
    }
    const double scale_denom = (norm == fft_norm_mode::by_root_n) ?
        std::sqrt(signal_numel) : static_cast<double>(signal_numel);
    return 1.0 / scale_denom;
}

const at::Tensor &_fft_apply_normalization(const at::Tensor &self, int64_t normalization,
                                           at::IntArrayRef sizes, at::IntArrayRef dims)
{
    auto scale = _fft_normalization_scale(normalization, sizes, dims);
    return (scale == 1.0) ? self : self.mul_(scale);
}

static void HackComplexintoFloat(at::Tensor& self)
{
    auto old_sizes = self.sym_sizes();
    at::SymDimVector new_sizes(old_sizes.size() + 1);
    std::copy(old_sizes.begin(), old_sizes.end(), new_sizes.begin());
    new_sizes.back() = 2;

    auto old_strides = self.sym_strides();
    at::SymDimVector new_strides(old_strides.size() + 1);
    for (uint32_t i = 0; i < old_strides.size(); i++) {
        new_strides[i] = old_strides[i] * 2;
    }
    new_strides.back() = 1;

    auto *impl = self.unsafeGetTensorImpl();
    impl->set_storage_and_dtype(self.storage(), c10::scalarTypeToTypeMeta(c10::toRealValueType(self.scalar_type())));
    impl->set_sizes_and_strides(new_sizes, new_strides, self.sym_storage_offset() * 2);
}


static void HackFloatintoComplex(at::Tensor& self)
{
    auto old_sizes = self.sym_sizes();
    at::SymDimVector new_sizes(old_sizes.size() - 1);
    std::copy(old_sizes.begin(), old_sizes.end() - 1, new_sizes.begin());

    auto old_strides = self.sym_strides();
    at::SymDimVector new_strides(old_strides.size() - 1);
    for (uint32_t i = 0; i < new_strides.size(); i++) {
        new_strides[i] = old_strides[i] / 2;
    }

    auto *impl = self.unsafeGetTensorImpl();
    impl->set_storage_and_dtype(self.storage(), c10::scalarTypeToTypeMeta(c10::toComplexType(self.scalar_type())));
    impl->set_sizes_and_strides(new_sizes, new_strides, self.sym_storage_offset() / 2);
}

static at::DimVector _sort_dims(const at::Tensor& self, at::IntArrayRef dim, int64_t mode_code = 0)
{
    auto mode = static_cast<fft_mode>(mode_code);
    at::DimVector sorted_dims(dim.begin(), dim.end() - (mode_code > 0));
    auto self_strides = self.strides();
    std::sort(sorted_dims.begin(), sorted_dims.end(),
        [&](int64_t a, int64_t b) { return self_strides[a] > self_strides[b]; });
    if (mode == fft_mode::c2r) {
        sorted_dims.push_back(dim.back());
    } else if (mode == fft_mode::r2c) {
        sorted_dims.insert(sorted_dims.begin(), dim.back());
    }
    return sorted_dims;
}

static op_api::PlanMode get_plan_mode(fft_mode mode, int idx, int signal_ndim, bool oneside)
{
    switch (mode) {
        case fft_mode::c2c:
            return op_api::PlanMode::c2c;
        case fft_mode::r2c:
            if (idx != 0) {
                return op_api::PlanMode::c2c;
            }
            if (oneside) {
                return op_api::PlanMode::r2c;
            } else {
                return op_api::PlanMode::r2c_bothside;
            }
        case fft_mode::c2r:
            if (idx != signal_ndim - 1) {
                return op_api::PlanMode::c2c;
            }
            return op_api::PlanMode::c2r;
    }
}

static at::Tensor& _exec_fft(at::Tensor& out_, const at::Tensor& self_, at::IntArrayRef out_sizes,
    at::IntArrayRef dim, int64_t normalization, bool forward, int64_t mode_code = 0)
{
    auto mode = static_cast<fft_mode>(mode_code);
    auto self = self_.view(self_.sizes());
    auto out = out_.view(out_.sizes());
    const auto ndim = self.dim();
    const auto signal_ndim = dim.size();
    const auto batch_dims = ndim - signal_ndim;

    // Permute dimensions so [signal_dims | batch_dims], and in each stride order
    at::DimVector dim_permute(ndim);
    std::iota(dim_permute.begin(), dim_permute.end(), int64_t{0});

    std::vector<bool> is_transformed_dim(ndim);
    for (const auto& d : dim) {
        is_transformed_dim[d] = true;
    }
    auto batch_end = std::partition(dim_permute.begin(), dim_permute.end(),
        [&](int64_t d) {return is_transformed_dim[d]; });
    auto self_strides = self.strides();
    at::DimVector sorted_dims = _sort_dims(self, dim, mode_code);
    sorted_dims = _sort_dims(self, dim, mode_code);
    std::copy(sorted_dims.begin(), sorted_dims.end(), dim_permute.begin());
    std::sort(batch_end, dim_permute.end(),
        [&](int64_t a, int64_t b) { return self_strides[a] > self_strides[b]; });

    if (mode != fft_mode::r2c) {
        dim_permute.insert(dim_permute.begin(), ndim);
        HackComplexintoFloat(self);
    }
    if (mode != fft_mode::c2r) {
        HackComplexintoFloat(out);
    }
    self = self.permute(dim_permute);

    // Calculate the output shape
    at::DimVector final_sizes(ndim);
    auto self_begin = self.sizes().begin();
    if (mode != fft_mode::r2c) {
        self_begin++;
    }
    std::copy(self_begin + signal_ndim, self.sizes().end(), final_sizes.begin());
    std::copy(self_begin, self_begin + signal_ndim, final_sizes.begin() + batch_dims);
    if (mode == fft_mode::c2r) {
        final_sizes[ndim - 1] = -1;
    } else if (mode == fft_mode::r2c) {
        final_sizes[batch_dims] = -1;
    }
    if (mode != fft_mode::c2r) {
        final_sizes.push_back(2);
    }
    // Create compute buffer
    at::Tensor tmp_pingpong[2];
    int64_t numel_buffer = self.numel();
    if (mode != fft_mode::c2c) {
        numel_buffer *= 2;
    }
    tmp_pingpong[0] = npu_preparation::apply_tensor_without_format(
        numel_buffer, self_.options().dtype(c10::toRealValueType(self_.scalar_type())));
    tmp_pingpong[1] = npu_preparation::apply_tensor_without_format(
        numel_buffer, self_.options().dtype(c10::toRealValueType(self_.scalar_type())));
    tmp_pingpong[0].resize_(self.sizes());
    tmp_pingpong[0].copy_(self);

    int64_t signal_total = 1;
    for (int64_t i = 0; i < signal_ndim; i++) {
        signal_total *= *(self_begin + i);
    }

    int64_t batch_total = 1;
    for (int64_t i = 0; i < batch_dims; i++) {
        batch_total *= final_sizes[i];
    }

    at::DimVector collapsed_sizes(3);

    // outer iteration
    uint32_t ping = 0;
    uint32_t pong = 1;
    for (int64_t i = 0; i < signal_ndim; i++) {
        if (*(self_begin + i) == 1) {
            if (mode == fft_mode::r2c && i == 0) {
                collapsed_sizes[0] = 1;
                collapsed_sizes[1] = -1;
                collapsed_sizes[2] = batch_total;
                tmp_pingpong[ping] = tmp_pingpong[ping].reshape(collapsed_sizes);
                collapsed_sizes[1] = tmp_pingpong[ping].size(1);
                out.resize_(collapsed_sizes);
                at::zeros_like_out(out, tmp_pingpong[ping]);
                collapsed_sizes[0] = 2;
                tmp_pingpong[pong].resize_(collapsed_sizes);
                at::cat_out(tmp_pingpong[pong], {tmp_pingpong[ping], out}, 0);
                ping = 1 - ping;
                pong = 1 - pong;
            } else {
                collapsed_sizes[0] = 2;
                collapsed_sizes[1] = -1;
                collapsed_sizes[2] = batch_total;
                tmp_pingpong[ping].reshape(collapsed_sizes);
            }
            continue;
        }
        // get plan
        auto plan_mode = get_plan_mode(mode, i, signal_ndim, out_sizes[dim.back()] != *(self_begin + i));
        int64_t radix = *(self_begin + i);
        if (plan_mode == op_api::PlanMode::c2r) {
            radix = out_sizes[dim.back()];
        }
        auto plan_item = op_api::get_plan(radix, forward, plan_mode, c10::toRealValueType(self_.scalar_type()));
        auto coefficient_matrix_list = plan_item.get_rotate_matrices();
        auto factors_list = plan_item.get_factors();
        uint32_t factors_num = coefficient_matrix_list.size();
        if ((plan_mode == op_api::PlanMode::c2r) && (out_sizes[dim.back()] != *(self_begin + i))) {
            // Complete the image
            collapsed_sizes[0] = 2;
            collapsed_sizes[1] = *(self_begin + i);
            collapsed_sizes[2] = -1;
            auto buffer_0 = tmp_pingpong[ping].reshape(collapsed_sizes);
            auto buffer_1 = at::slice(buffer_0, 1, 1, collapsed_sizes[1]
                                      - ((out_sizes[dim.back()] == 2 * (collapsed_sizes[1] - 1)) ? 1 : 0), 1);
            out.resize_(buffer_1.sizes());
            at::flip_out(out, buffer_1, 1);
            auto imag_buffer = at::select(out, 0, 1);
            at::neg_(imag_buffer);
            collapsed_sizes[1] = out_sizes[dim.back()];
            collapsed_sizes[2] = buffer_0.size(2);
            tmp_pingpong[pong].resize_(collapsed_sizes);
            at::cat_out(tmp_pingpong[pong], {buffer_0, out}, 1);

            ping = 1 - ping;
            pong = 1 - pong;
        }
        // fft iteration
        for (auto& coefficient_matrix : coefficient_matrix_list) {
            collapsed_sizes[0] = coefficient_matrix.size(0);
            collapsed_sizes[1] = coefficient_matrix.size(2);
            collapsed_sizes[2] = -1;
            tmp_pingpong[ping] = tmp_pingpong[ping].reshape(collapsed_sizes);
            collapsed_sizes[1] = coefficient_matrix.size(1);
            collapsed_sizes[2] = tmp_pingpong[ping].size(2);
            tmp_pingpong[pong].resize_(collapsed_sizes);

            uint8_t cube_math_type = 0;
            EXEC_NPU_CMD(aclnnMatmul, coefficient_matrix, tmp_pingpong[ping], tmp_pingpong[pong], cube_math_type);
            ping = 1 - ping;
            pong = 1 - pong;
        }
        // reshape
        at::DimVector reshape_sizes(factors_num + 3);
        if ((mode == fft_mode::c2r) && (i == (signal_ndim - 1))) {
            reshape_sizes[factors_num] = 1;
        } else {
            reshape_sizes[factors_num] = 2;
        }
        reshape_sizes[factors_num + 1] = signal_total / *(self_begin + i);
        reshape_sizes[factors_num + 2] = batch_total;
        std::copy(factors_list.cbegin(), factors_list.cend(), reshape_sizes.begin());
        if (plan_mode == op_api::PlanMode::r2c) {
            signal_total /= factors_list.back();
            signal_total *= (factors_list.back() / 2 + 1);
            reshape_sizes[factors_num - 1] = reshape_sizes[factors_num - 1] / 2 + 1;
        }
        tmp_pingpong[ping] = tmp_pingpong[ping].reshape(reshape_sizes);

        // permute
        at::DimVector dim_permute_(factors_num + 3);
        std::iota(dim_permute_.rbegin() + 1, dim_permute_.rbegin() + factors_num + 1, int64_t{0});
        dim_permute_[0] = factors_num;
        dim_permute_[1] = factors_num + 1;
        dim_permute_[factors_num + 2] = factors_num + 2;
        tmp_pingpong[ping] = tmp_pingpong[ping].permute(dim_permute_);

        // transpose
        if (i != (signal_ndim - 1)) {
            tmp_pingpong[pong].resize_(tmp_pingpong[ping].sizes());
            tmp_pingpong[pong].copy_(tmp_pingpong[ping]);
            ping = 1 - ping;
            pong = 1 - pong;
        }
    }

    if ((mode == fft_mode::r2c) && (signal_ndim > 1)) {
        if (out_sizes[dim.back()] != *(self_begin)) {
            int64_t signal_total_final = out_sizes[dim.back()];
            for (int64_t i = 1; i < signal_ndim - 1; i++) {
                signal_total_final *= *(self_begin + i);
            }
            tmp_pingpong[ping] = at::slice(tmp_pingpong[ping], 1, 0, signal_total_final, 1);
        }
    }

    int out_ndim = tmp_pingpong[ping].dim();
    at::DimVector dim_permute_out(out_ndim);
    std::iota(dim_permute_out.begin(), dim_permute_out.end(), int64_t{0});
    dim_permute_out[0] = out_ndim - 1;
    dim_permute_out[out_ndim - 1] = 0;
    tmp_pingpong[ping] = tmp_pingpong[ping].permute(dim_permute_out);

    out.resize_(tmp_pingpong[ping].sizes());
    out.copy_(tmp_pingpong[ping]);
    out = out.reshape(final_sizes);
    if ((mode == fft_mode::r2c) && (signal_ndim == 1) && (out_sizes[dim.back()] != *(self_begin))) {
        out = at::slice(out, ndim - 1, 0, out_sizes[dim.back()], 1);
    }

    if (mode != fft_mode::r2c) {
        _fft_apply_normalization(out, normalization, out_sizes, dim);
    } else {
        _fft_apply_normalization(out, normalization, self_.sizes(), dim);
    }

    if (mode != fft_mode::c2r) {
        HackFloatintoComplex(out);
    }
    // Inplace reshaping to original batch shape and inverting the dimension permutation
    if (mode != fft_mode::r2c) {
        dim_permute.erase(dim_permute.begin());
    }
    at::DimVector out_strides(ndim);
    auto now_strides_ = out.strides();
    for (const auto i : c10::irange(0, signal_ndim)) {
        out_strides[dim_permute[i]] = now_strides_[i + batch_dims];
    }
    for (const auto i : c10::irange(signal_ndim, ndim)) {
        out_strides[dim_permute[i]] = now_strides_[i - signal_ndim];
    }
    out_.as_strided_(out_sizes, out_strides, out.storage_offset());
    return out_;
}

static at::Tensor& _exec_fft_asdsip(at::Tensor& out_, const at::Tensor& self_, at::IntArrayRef out_sizes,
    at::IntArrayRef dim, int64_t normalization, bool forward, int64_t mode_code = 0)
{
    auto mode = static_cast<fft_mode>(mode_code);
    auto self = self_.view(self_.sizes());
    auto out = out_.view(out_.sizes());
    const auto ndim = self.dim();
    const auto signal_ndim = dim.size();
    const auto batch_dims = ndim - signal_ndim;

    // trans signal_dim to the back
    at::DimVector dim_permute(ndim);
    std::iota(dim_permute.begin(), dim_permute.end(), int64_t{0});

    std::vector<bool> is_transformed_dim(ndim);
    for (const auto& d : dim) {
        is_transformed_dim[d] = true;
    }
    auto batch_end = std::partition(dim_permute.begin(), dim_permute.end(),
        [&](int64_t d) {return !is_transformed_dim[d]; });
    auto self_strides = self.strides();
    at::DimVector sorted_dims(dim.begin(), dim.end() - (mode != fft_mode::c2c));
    std::sort(sorted_dims.begin(), sorted_dims.end(),
        [&](int64_t a, int64_t b) { return self_strides[a] > self_strides[b]; });
    if (mode != fft_mode::c2c) {
        sorted_dims.push_back(dim.back());
    }
    std::copy(sorted_dims.begin(), sorted_dims.end(), batch_end);
    std::sort(dim_permute.begin(), batch_end,
        [&](int64_t a, int64_t b) { return self_strides[a] > self_strides[b]; });

    at::Tensor asdfft_input;
    if (mode != fft_mode::r2c) {
        HackComplexintoFloat(self);
        dim_permute.push_back(ndim);
        asdfft_input = self.permute(dim_permute).contiguous();
        dim_permute.pop_back();
        HackFloatintoComplex(asdfft_input);
    } else {
        asdfft_input = self.permute(dim_permute).contiguous();
    }

    auto out_ori = out.permute(dim_permute);

    // squeeze batch_dims
    at::DimVector asdsip_sizes(signal_ndim + 1);
    asdsip_sizes[0] = -1;
    std::copy(asdfft_input.sizes().begin() + batch_dims, asdfft_input.sizes().end(), asdsip_sizes.begin() + 1);
    asdfft_input = asdfft_input.reshape(asdsip_sizes);

    at::DimVector out_sizes_(out_ori.sizes()); // record shape
    std::copy(out_ori.sizes().begin() + batch_dims, out_ori.sizes().end(), asdsip_sizes.begin() + 1);
    out_ori = out_ori.reshape(asdsip_sizes);
    out.resize_(out_ori.sizes());

    // call asdsip
    FFTParam param;
    param.batchSize = asdfft_input.size(0);
    if (forward) {
        param.direction = asdFftDirection::ASCEND_FFT_FORWARD;
    } else {
        param.direction = asdFftDirection::ASCEND_FFT_BACKWARD;
    }
    switch (mode) {
        case fft_mode::c2c:
            param.fftXSize = asdfft_input.size(1);
            if (signal_ndim == 2) {
                param.fftYSize = asdfft_input.size(2);
            }
            param.fftType = asdFftType::ASCEND_FFT_C2C;
            EXEC_ASDSIP_FFT_NPU_CMD(C2C, asdfft_input, out, param);
            break;
        case fft_mode::r2c:
            param.fftXSize = asdfft_input.size(1);
            if (signal_ndim == 2) {
                param.fftYSize = asdfft_input.size(2);
            }
            param.fftType = asdFftType::ASCEND_FFT_R2C;
            EXEC_ASDSIP_FFT_NPU_CMD(R2C, asdfft_input, out, param);
            break;
        case fft_mode::c2r:
            param.fftXSize = out.size(1);
            if (signal_ndim == 2) {
                param.fftYSize = out.size(2);
            }
            param.fftType = asdFftType::ASCEND_FFT_C2R;
            EXEC_ASDSIP_FFT_NPU_CMD(C2R, asdfft_input, out, param);
            break;
    }

    // normalize
    auto norm_out_sizes = out_sizes;
    if (mode == fft_mode::r2c) {
        norm_out_sizes = self_.sizes();
    }
    if (mode != fft_mode::c2r) {
        HackComplexintoFloat(out);
    }
    _fft_apply_normalization(out, normalization, norm_out_sizes, dim);
    if (mode != fft_mode::c2r) {
        HackFloatintoComplex(out);
    }

    // restore shape
    out = out.reshape(out_sizes_);
    auto now_strides_ = out.strides();
    at::DimVector out_strides(ndim);
    for (const auto i : c10::irange(0, ndim)) {
        out_strides[dim_permute[i]] = now_strides_[i];
    }
    out_.as_strided_(out_sizes, out_strides, out.storage_offset());
    return out_;
}

at::Tensor _fft_c2c(const at::Tensor& self, at::IntArrayRef dim, int64_t normalization, bool forward)
{
    TORCH_CHECK(self.is_complex(), OPS_ERROR(ErrCode::PARAM));
    auto output_size = op_infer::input_same_output_size(self);
    auto out = npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(self.scalar_type()));
    DO_ASDSIP_COMPATIBILITY(C2C, _exec_fft(out, self, self.sizes(), dim, normalization, forward, 0));
    if (dim.size() > 1 || self.scalar_type() == at::ScalarType::ComplexHalf) {
        _exec_fft(out, self, self.sizes(), dim, normalization, forward, 0);
    } else {
        _exec_fft_asdsip(out, self, self.sizes(), dim, normalization, forward, 0);
    }
    return out;
}

at::Tensor& _fft_c2c_out(const at::Tensor& self, at::IntArrayRef dim, int64_t normalization,
    bool forward, at::Tensor& out)
{
    TORCH_CHECK(self.is_complex(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(out.is_complex(), OPS_ERROR(ErrCode::PARAM));
    DO_ASDSIP_COMPATIBILITY(C2C, _exec_fft(out, self, self.sizes(), dim, normalization, forward, 0));
    if (dim.size() > 1 || self.scalar_type() == at::ScalarType::ComplexHalf) {
        _exec_fft(out, self, self.sizes(), dim, normalization, forward, 0);
    } else {
        _exec_fft_asdsip(out, self, self.sizes(), dim, normalization, forward, 0);
    }
    return out;
}

at::Tensor _fft_r2c(const at::Tensor& self, at::IntArrayRef dim, int64_t normalization, bool onesided)
{
    TORCH_CHECK(self.is_floating_point(), OPS_ERROR(ErrCode::PARAM));
    auto input_sizes = self.sizes();
    at::DimVector out_sizes(input_sizes.begin(), input_sizes.end());
    auto last_dim = dim.back();
    auto last_dim_halfsize = (input_sizes[last_dim]) / 2 + 1;
    if (onesided) {
        out_sizes[last_dim] = last_dim_halfsize;
    }
    auto out = npu_preparation::apply_tensor_without_format(
        out_sizes, self.options().dtype(c10::toComplexType(self.scalar_type())));

    DO_ASDSIP_COMPATIBILITY(R2C, _exec_fft(out, self, out_sizes, dim, normalization, true, 1));
    if (dim.size() > 1 || self.scalar_type() == at::ScalarType::Half) {
        _exec_fft(out, self, out_sizes, dim, normalization, true, 1);
    } else {
        _exec_fft_asdsip(out, self, out_sizes, dim, normalization, true, 1);
    }
    return out;
}

at::Tensor &_fft_r2c_out(const at::Tensor &self, at::IntArrayRef dim,
                         int64_t normalization, bool onesided, at::Tensor &out)
{
    TORCH_CHECK(self.is_floating_point(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(out.is_complex(), OPS_ERROR(ErrCode::PARAM));
    auto input_sizes = self.sizes();
    at::DimVector out_sizes(input_sizes.begin(), input_sizes.end());
    auto last_dim = dim.back();
    auto last_dim_halfsize = (input_sizes[last_dim]) / 2 + 1;
    if (onesided) {
        out_sizes[last_dim] = last_dim_halfsize;
    }
    DO_ASDSIP_COMPATIBILITY(R2C, _exec_fft(out, self, out_sizes, dim, normalization, true, 1));
    if (dim.size() > 1 || self.scalar_type() == at::ScalarType::Half) {
        _exec_fft(out, self, out_sizes, dim, normalization, true, 1);
    } else {
        _exec_fft_asdsip(out, self, out_sizes, dim, normalization, true, 1);
    }
    return out;
}

at::Tensor _fft_c2r(const at::Tensor& self, at::IntArrayRef dim, int64_t normalization, int64_t lastdim)
{
    TORCH_CHECK(self.is_complex(), OPS_ERROR(ErrCode::PARAM));
    auto in_sizes = self.sizes();
    at::DimVector out_sizes(in_sizes.begin(), in_sizes.end());
    out_sizes[dim.back()] = lastdim;
    auto out = npu_preparation::apply_tensor_without_format(
        out_sizes, self.options().dtype(c10::toRealValueType(self.scalar_type())));
    DO_ASDSIP_COMPATIBILITY(C2R, _exec_fft(out, self, out_sizes, dim, normalization, self.is_conj(), 2));
    if (dim.size() > 1 || self.scalar_type() == at::ScalarType::ComplexHalf) {
        _exec_fft(out, self, out_sizes, dim, normalization, self.is_conj(), 2);
    } else {
        _exec_fft_asdsip(out, self, out_sizes, dim, normalization, self.is_conj(), 2);
    }
    return out;
}

at::Tensor &_fft_c2r_out(const at::Tensor &self, at::IntArrayRef dim,
                         int64_t normalization, int64_t lastdim, at::Tensor &out)
{
    TORCH_CHECK(self.is_complex(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(out.is_floating_point(), OPS_ERROR(ErrCode::PARAM));
    auto in_sizes = self.sizes();
    at::DimVector out_sizes(in_sizes.begin(), in_sizes.end());
    out_sizes[dim.back()] = lastdim;
    DO_ASDSIP_COMPATIBILITY(C2R, _exec_fft(out, self, out_sizes, dim, normalization, self.is_conj(), 2));
    if (dim.size() > 1 || self.scalar_type() == at::ScalarType::ComplexHalf) {
        _exec_fft(out, self, out_sizes, dim, normalization, self.is_conj(), 2);
    } else {
        _exec_fft_asdsip(out, self, out_sizes, dim, normalization, self.is_conj(), 2);
    }
    return out;
}

#endif

}  // namespace op_api

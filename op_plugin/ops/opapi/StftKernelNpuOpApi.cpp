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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {

static const int X1_NFFT = 400;
static const int X1_HOP = 160;
static const int TYPE_SIZE = 4; // float32
static const int BLOCK_SIZE = 32;
static const int PACKAGE_SIZE = 128;
static const int MAX_CACHE_SIZE = 500 * 1024 * 1024;                   // 500MB
static const uint64_t MAX_GM_SIZE = (uint64_t)35 * 1024 * 1024 * 1024; // 35GB
static const int64_t DIM_ONE = 1;
static const int64_t DIM_TWO = 2;

at::ScalarType get_output_type(bool return_complex, at::ScalarType input_type)
{
    at::ScalarType output_type;
    if (return_complex) {
        if (input_type == at::ScalarType::ComplexFloat || input_type == at::ScalarType::ComplexDouble) {
            output_type = input_type;
        } else if (input_type == at::ScalarType::Float) {
            output_type = at::ScalarType::ComplexFloat;
        } else if (input_type == at::ScalarType::Double) {
            output_type = at::ScalarType::ComplexDouble;
        }
    } else {
        if (input_type == at::ScalarType::Float || input_type == at::ScalarType::Double) {
            output_type = input_type;
        } else if (input_type == at::ScalarType::ComplexFloat) {
            output_type = at::ScalarType::Float;
        } else if (input_type == at::ScalarType::ComplexDouble) {
            output_type = at::ScalarType::Double;
        }
    }
    return output_type;
}

c10::SmallVector<int64_t, SIZE> get_output_size(bool return_complex, int64_t batch, int64_t frames, int64_t n)
{
    c10::SmallVector<int64_t, SIZE> output_size;
    c10::SmallVector<int64_t, SIZE> output_complex_with_batch = {batch, n, frames};
    c10::SmallVector<int64_t, SIZE> output_complex = {n, frames};
    c10::SmallVector<int64_t, SIZE> output_real_with_batch = {batch, n, frames, 2};
    c10::SmallVector<int64_t, SIZE> output_real = {n, frames, 2};

    if (return_complex) {
        output_size = batch > 0 ? output_complex_with_batch : output_complex;
    } else {
        output_size = batch > 0 ? output_real_with_batch : output_real;
    }
    return output_size;
}

bool is_aclstft_supported(at::Tensor const &self,
                          at::Tensor const &window,
                          int64_t nFft,
                          int64_t hopLength,
                          bool normalized,
                          bool onesided,
                          bool returnComplex)
{
    if (self.scalar_type() == at::ScalarType::Double || self.scalar_type() == at::ScalarType::ComplexDouble) {
        return true;
    }
    
    bool res = false;
    int64_t batch = self.dim() == 2 ? self.size(0) : 0;
    int64_t len = self.dim() == 2 ? self.size(1) : self.size(0);
    if (hopLength <= 0) {
        return false;
    }
    int64_t matmulN = (len - nFft) / hopLength + 1;
    int64_t matmulM = onesided ? nFft / 2 + 1 : nFft;
    int alignNum =
        (nFft == X1_NFFT && hopLength == X1_HOP && normalized == false && onesided == true && returnComplex == false) ?
            BLOCK_SIZE / TYPE_SIZE :
            PACKAGE_SIZE / TYPE_SIZE;
    int64_t nFftAlign = (nFft + alignNum - 1) / alignNum * alignNum;
    int64_t cacheSize = (matmulM * nFftAlign) * TYPE_SIZE * 2;
    // gm size ： input + output + splitWindow workspace + matmul workspace
    uint64_t gmSize = batch * len * TYPE_SIZE + batch * matmulM * matmulN * 2 * TYPE_SIZE +
                      batch * matmulN * nFftAlign * TYPE_SIZE + batch * matmulM * matmulN * 2 * TYPE_SIZE;
    if (cacheSize <= MAX_CACHE_SIZE && gmSize <= MAX_GM_SIZE && self.scalar_type() == at::ScalarType::Float) {
        if (!window.defined() || window.scalar_type() == at::ScalarType::Float) {
            res = true;
        }
    }
    return res;
}

at::Tensor _stft(at::Tensor const &self,
                 int64_t n_fft,
                 c10::optional<int64_t> hop_lengthOpt,
                 c10::optional<int64_t> win_lengthOpt,
                 c10::optional<at::Tensor> const &window_opt,
                 bool normalized,
                 c10::optional<bool> onesidedOpt,
                 c10::optional<bool> return_complexOpt,
                 c10::optional<bool> align_to_windowOpt)
{
    c10::MaybeOwned<at::Tensor> window_maybe_owned = at::borrow_from_optional_tensor(window_opt);
    const at::Tensor& window = *window_maybe_owned;

    // default_init hop_length and win_length
    auto hop_length = hop_lengthOpt.value_or(n_fft >> 2);
    auto win_length = win_lengthOpt.value_or(n_fft);
    const bool return_complex = return_complexOpt.value_or(
        self.is_complex() || (window.defined() && window.is_complex()));

    at::Tensor input = self;
    if (self.dim() == 1) {
        input = input.unsqueeze(0);
    }

    int64_t batch = input.size(0);
    int64_t len = input.size(1);

    auto window_ = window;
    if (win_length < n_fft) {
        // pad center
        auto left = (n_fft - win_length) / 2;
        if (window.defined()) {
            window_ = at::zeros({n_fft}, window.options());
            window_.narrow(0, left, win_length).copy_(window);
        } else {
            window_ = at::zeros({n_fft}, self.options());
            window_.narrow(0, left, win_length).fill_(1);
        }
    }

    const bool align_to_window = align_to_windowOpt.value_or(false);
    int64_t n_frames;
    if (align_to_window) {
        // Calculate n_frames based on window length, since we are aligning start of window with t = 0.
        n_frames = 1 + (len - win_length) / hop_length;
        // Window-based padding.
        input = at::pad(input, {(n_fft - win_length) / 2, (n_fft - win_length) / 2}, "constant");
    } else {
        n_frames = 1 + (len - n_fft) / hop_length;
    }
    // time2col
    input = input.as_strided({batch, n_frames, n_fft}, {input.stride(0), hop_length * input.stride(1), input.stride(1)});
    if (window_.defined()) {
        input = input.mul(window_);
    }

    // FFT and transpose to get (batch x fft_size x num_frames)
    const bool complex_fft = input.is_complex();
    const auto onesided = onesidedOpt.value_or(!complex_fft);

    at::Tensor out;
    if (complex_fft) {
        TORCH_CHECK(!onesided, "Cannot have onesided output if window or input is complex");
        out = at::_fft_c2c(input, input.dim() - 1, static_cast<int64_t>(normalized), true);
    } else {
        out = at::_fft_r2c(input, input.dim() - 1, static_cast<int64_t>(normalized), onesided);
    }
    out.transpose_(DIM_ONE, DIM_TWO);

    if (self.dim() == 1) {
        out.squeeze_(0);
    }

    if (return_complex) {
        return out;
    } else {
        return at::view_as_real(out);
    }
}

}

#if VERSION_BETWEEN(V2R1, V2R6)
at::Tensor stft(at::Tensor const &self,
                int64_t n_fft,
                c10::optional<int64_t> hop_length,
                c10::optional<int64_t> win_length,
                c10::optional<at::Tensor> const &window,
                bool normalized,
                c10::optional<bool> onesided,
                c10::optional<bool> return_complex)
{
    TORCH_CHECK(self.dim() == 1 || self.dim() == 2, "input should be a 1D or 2D tensor", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(n_fft > 0 && n_fft <= self.size(-1), "expected: 0 < n_fft < input.size(-1)", OPS_ERROR(ErrCode::PARAM));

    int64_t hop_length_value = hop_length.has_value() ? hop_length.value() : n_fft / 4;
    TORCH_CHECK(hop_length_value > 0, "expected: hop_length > 0", OPS_ERROR(ErrCode::VALUE));

    if (window.has_value() && win_length.has_value()) {
        TORCH_CHECK(window.value().dim() == 1 && window.value().size(0) == win_length.value(),
                    "expected: window size and win_length should be equal", OPS_ERROR(ErrCode::PARAM))
    }

    int win_length_value = win_length.has_value() ? win_length.value() : n_fft;
    TORCH_CHECK(win_length_value > 0 && win_length_value <= n_fft, "expected: 0 < win_length <= n_fft", OPS_ERROR(ErrCode::PARAM));

    const at::Tensor &window_value = c10::value_or_else(window, [] { return at::Tensor(); });
    bool onesided_value = onesided.has_value() ? onesided.value() : !self.is_complex();
    bool return_complex_value = return_complex.has_value() ?
                          return_complex.value() :
                          self.is_complex() || (window.has_value() && window.value().is_complex());

    int64_t batch = self.dim() == 2 ? self.size(0) : 0;
    int64_t len = self.dim() == 2 ? self.size(1) : self.size(0);
    int64_t frames = (len - n_fft) / hop_length_value + 1;
    int64_t n = onesided_value ? n_fft / 2 + 1 : n_fft;
    at::ScalarType output_type = get_output_type(return_complex_value, self.scalar_type());
    TORCH_CHECK(output_type == at::ScalarType::Float || output_type == at::ScalarType::Double ||
                output_type == at::ScalarType::ComplexFloat || output_type == at::ScalarType::ComplexDouble,
                "output type should be float, double, complex<float> or complex<double>", OPS_ERROR(ErrCode::TYPE));

    if (!is_aclstft_supported(self, window_value, n_fft, hop_length_value, normalized, onesided_value, return_complex_value)) {
        return _stft(self, n_fft, hop_length, win_length, window, normalized, onesided, return_complex, c10::nullopt);
    }

    c10::SmallVector<int64_t, SIZE> output_size = get_output_size(return_complex_value, batch, frames, n);
    at::Tensor output = npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(output_type));

    EXEC_NPU_CMD(aclStft, self, window_value, output, n_fft, hop_length_value, win_length_value, normalized, onesided_value, return_complex_value);
    return output;
}
#endif

#if VERSION_BETWEEN(V2R7, VERSION_NEWEST)
at::Tensor stft(at::Tensor const &self,
                int64_t n_fft,
                c10::optional<int64_t> hop_length_opt,
                c10::optional<int64_t> win_length_opt,
                c10::optional<at::Tensor> const &window_opt,
                bool normalized,
                c10::optional<bool> onesided_opt,
                c10::optional<bool> return_complex_opt,
                c10::optional<bool> align_to_window)
{
    TORCH_CHECK(self.dim() == 1 || self.dim() == 2, "input should be a 1D or 2D tensor", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(n_fft > 0 && n_fft <= self.size(-1), "expected: 0 < n_fft < input.size(-1)", OPS_ERROR(ErrCode::PARAM));

    int64_t hop_length = hop_length_opt.has_value() ? hop_length_opt.value() : n_fft / 4;
    TORCH_CHECK(hop_length > 0, "expected: hop_length > 0", OPS_ERROR(ErrCode::VALUE));

    if (window_opt.has_value() && win_length_opt.has_value()) {
        TORCH_CHECK(window_opt.value().dim() == 1 && window_opt.value().size(0) == win_length_opt.value(),
                    "expected: window size and win_length should be equal", OPS_ERROR(ErrCode::PARAM))
    }

    int win_length = win_length_opt.has_value() ? win_length_opt.value() : n_fft;
    TORCH_CHECK(win_length > 0 && win_length <= n_fft, "expected: 0 < win_length <= n_fft", OPS_ERROR(ErrCode::PARAM));

    const at::Tensor &window = c10::value_or_else(window_opt, [] { return at::Tensor(); });
    bool onesided = onesided_opt.has_value() ? onesided_opt.value() : !self.is_complex();
    bool return_complex = return_complex_opt.has_value() ?
                          return_complex_opt.value() :
                          self.is_complex() || (window_opt.has_value() && window_opt.value().is_complex());

    int64_t batch = self.dim() == 2 ? self.size(0) : 0;
    int64_t len = self.dim() == 2 ? self.size(1) : self.size(0);
    int64_t frames = (len - n_fft) / hop_length + 1;
    int64_t n = onesided == true ? n_fft / 2 + 1 : n_fft;
    at::ScalarType output_type = get_output_type(return_complex, self.scalar_type());
    TORCH_CHECK(output_type == at::ScalarType::Float || output_type == at::ScalarType::Double ||
                output_type == at::ScalarType::ComplexFloat || output_type == at::ScalarType::ComplexDouble,
                "output type should be float, double, complex<float> or complex<double>", OPS_ERROR(ErrCode::TYPE));

    if (!is_aclstft_supported(self, window, n_fft, hop_length, normalized, onesided, return_complex)) {
        return _stft(self, n_fft, hop_length_opt, win_length_opt, window_opt, normalized, onesided_opt, return_complex_opt, align_to_window);
    }

    c10::SmallVector<int64_t, SIZE> output_size = get_output_size(return_complex, batch, frames, n);
    at::Tensor output = npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(output_type));

    EXEC_NPU_CMD(aclStft, self, window, output, n_fft, hop_length, win_length, normalized, onesided, return_complex);
    return output;
}
#endif

#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
enum class fft_norm_mode {
    none,       // No normalization
    by_root_n,  // Divide by sqrt(signal_size)
    by_n,       // Divide by signal_size
};

static inline bool _maybe_overlapping_memory(
    c10::SymIntArrayRef sizes,
    c10::SymIntArrayRef strides)
{
    if (!sizes.empty()) {
        std::vector<std::size_t> argsort(sizes.size());
        std::iota(argsort.begin(), argsort.end(), 0);
        std::sort(argsort.begin(), argsort.end(), [&](std::size_t i, std::size_t j) {
            return strides[i] < strides[j];
        });

        c10::SymInt max_index_in_slice = 0;
        for (auto i : argsort) {
            const auto& stride_ = strides[i];
            if (stride_ <= max_index_in_slice) {
                return true;
            }
            max_index_in_slice += stride_ * (sizes[i] - 1);
        }
    }
    return false;
}

static inline c10::SymInt _min_storage_size(
    c10::SymIntArrayRef sizes,
    c10::SymIntArrayRef strides,
    c10::SymInt storage_offset)
{
    c10::SymInt storage_size = storage_offset + 1;
    auto dim = sizes.size();
    for (const auto i : c10::irange(dim)) {
        const auto& size_i = sizes[i];
        if (size_i == 0) {
            return storage_offset;
        }
        storage_size += (size_i - 1) * strides[i];
    }
    return storage_size;
}

at::Tensor as_strided_backward_(
    at::Tensor grad,
    const at::TensorGeometry& input_geometry,
    c10::SymIntArrayRef sym_sizes,
    c10::SymIntArrayRef sym_strides,
    const c10::optional<c10::SymInt>& sym_storage_offset_)
{
    auto sym_storage_offset = sym_storage_offset_.value_or(input_geometry.sym_storage_offset());
    auto odim = grad.dim();
    std::vector<c10::SymInt> out_sizes_;
    std::vector<c10::SymInt> out_strides_;
    out_sizes_.reserve(odim);
    out_strides_.reserve(odim);
    for (int64_t i = odim - 1; i >= 0; i--) {
        const auto& size_i = sym_sizes[i];
        const auto& stride_i = sym_strides[i];
        if (size_i == 0) {
            return at::zeros_symint(input_geometry.sym_sizes(), grad.options());
        } else if (size_i == 1) {
            grad = grad.squeeze(i);
        } else if (stride_i == 0) {
            grad = grad.sum(i, false);
        } else {
            out_sizes_.insert(out_sizes_.begin(), size_i);
            out_strides_.insert(out_strides_.begin(), stride_i);
        }
    }
    auto out_maybe_overlap = _maybe_overlapping_memory(out_sizes_, out_strides_);

    auto idim = input_geometry.dim();
    auto inp_sizes = input_geometry.sym_sizes();
    auto inp_strides = input_geometry.sym_strides();
    std::vector<c10::SymInt> inp_sizes_;
    std::vector<c10::SymInt> inp_strides_;
    inp_sizes_.reserve(idim);
    inp_strides_.reserve(idim);
    for (int64_t i = idim - 1; i >= 0; i--) {
        const auto& size_i = inp_sizes[i];
        const auto& stride_i = inp_strides[i];
        if (size_i == 0) {
            return at::zeros_symint(input_geometry.sym_sizes(), grad.options());
        } else if (size_i != 1) {
            inp_sizes_.insert(inp_sizes_.begin(), size_i);
            inp_strides_.insert(inp_strides_.begin(), stride_i);
        }
    }

    auto inp_maybe_overlap = _maybe_overlapping_memory(inp_sizes_, inp_strides_);

    auto shared_offset =
        input_geometry.sym_storage_offset().min(sym_storage_offset);
    auto inp_effective_offset =
        input_geometry.sym_storage_offset() - shared_offset;
    auto out_effective_offset = sym_storage_offset - shared_offset;
    auto base_size1 =
        _min_storage_size(inp_sizes_, inp_strides_, inp_effective_offset);
    auto base_size2 =
        _min_storage_size(out_sizes_, out_strides_, out_effective_offset);
    auto base_size = base_size1.max(base_size2);
    auto storage = grad.new_zeros_symint(c10::SymIntArrayRef(base_size));

    c10::optional<at::Tensor> flatten_full_indices;
    if (inp_maybe_overlap || out_maybe_overlap) {
        flatten_full_indices =
            at::arange(
                0,
                base_size.guard_int(__FILE__, __LINE__),
                grad.options().dtype(at::kLong));
    }

    if (out_maybe_overlap) {
        auto out_indices = flatten_full_indices->as_strided_symint(
            out_sizes_, out_strides_, out_effective_offset);
        storage.index_add_(0, out_indices.reshape(-1), grad.reshape(-1));
    } else {
        storage.as_strided_symint(out_sizes_, out_strides_, out_effective_offset)
            .copy_(grad);
    }

    if (inp_maybe_overlap) {
        auto count = at::zeros_like(storage, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
        auto inp_indices =
            flatten_full_indices
                ->as_strided_symint(inp_sizes_, inp_strides_, inp_effective_offset)
                .reshape(-1);
        count.index_add_(
            0, inp_indices, at::ones({1}, grad.options()).expand_as(inp_indices));
        storage.div_(count);
    }
    return storage.as_strided_symint(
        inp_sizes, inp_strides, inp_effective_offset);
}

at::Tensor stft_backward(
    at::Tensor const &grad_output,
    at::Tensor const &self,
    int64_t n_fft,
    c10::optional<int64_t> hop_length,
    c10::optional<int64_t> win_length,
    c10::optional<at::Tensor> const &window,
    bool normalized,
    c10::optional<bool> onesided,
    c10::optional<bool> return_complex)
{
    c10::MaybeOwned<at::Tensor> window_maybe_owned = at::borrow_from_optional_tensor(window);
    const at::Tensor& real_window = *window_maybe_owned;
    auto hop_length_value = hop_length.has_value() ? hop_length.value() : n_fft / 4;
    TORCH_CHECK(hop_length_value > 0, "expected: hop_length > 0", OPS_ERROR(ErrCode::VALUE));
    auto win_length_value = win_length.value_or(n_fft);
    const bool return_complex_value = return_complex.value_or(
        self.is_complex() || (real_window.defined() && real_window.is_complex()));
    auto window_ = real_window;
    if (win_length_value < n_fft) {
        auto left = (n_fft - win_length_value) / 2;
        if (real_window.defined()) {
            window_ = at::zeros({n_fft}, real_window.options());
            window_.narrow(0, left, win_length_value).copy_(real_window);
        } else {
            window_ = at::zeros({n_fft}, self.options());
            window_.narrow(0, left, win_length_value).fill_(1);
        }
    }

    at::Tensor grad_input = grad_output;
    if (!return_complex_value) {
        grad_input = at::view_as_complex(grad_output.contiguous());
    }
    if (self.dim() == 1) {
        grad_input = grad_input.unsqueeze(0);
    }
    grad_input = grad_input.transpose(1, 2).contiguous();
    const bool complex_fft = self.is_complex() || window_.is_complex();
    const auto onesided_value = onesided.value_or(!complex_fft);
    const fft_norm_mode norm = normalized ? fft_norm_mode::by_root_n : fft_norm_mode::none;
    if (complex_fft) {
        grad_input = at::_fft_c2c(grad_input, grad_input.dim() - 1, static_cast<int64_t>(norm), false);
    } else {
        if (!onesided_value) {
            grad_input = at::_fft_c2c(grad_input, grad_input.dim() - 1, static_cast<int64_t>(norm), false);
        } else {
            auto half_sizes = grad_input.sym_sizes();
            std::vector<c10::SymInt> new_grad_shape(half_sizes.begin(), half_sizes.end());
            new_grad_shape[grad_input.dim() - 1] = n_fft;

            const auto zero_length = n_fft - grad_input.sym_size(grad_input.dim() - 1);
            auto complex_full_grad = zero_length > 0 ? grad_input.new_zeros_symint(new_grad_shape) : grad_input;
            if (zero_length > 0) {
                complex_full_grad.slice_symint(grad_input.dim() - 1, 0, half_sizes[grad_input.dim() - 1]).copy_(grad_input);
            }
            grad_input = at::_fft_c2c(complex_full_grad, grad_input.dim() - 1, static_cast<int64_t>(norm), false);
        }
        grad_input = at::view_as_real(grad_input).select(grad_input.dim(), 0).contiguous();
    }
    if (window_.defined()) {
        grad_input = grad_input * window_.conj();
        if (!self.is_complex() && grad_input.is_complex()) {
            grad_input = at::view_as_real(grad_input).select(grad_input.dim(), 0).contiguous();
        }
    }
    auto self_ = self;
    if (self.dim() == 1) {
        self_ = self_.unsqueeze(0);
    }
    if (self.is_complex()) {
        self_ = at::view_as_real(self_);
        grad_input = at::view_as_real(grad_input);
    }
    int64_t batch = self_.size(0);
    int64_t len = self_.size(1);
    int64_t n_frames = 1 + (len - n_fft) / hop_length_value;
    std::vector<c10::SymInt> sym_sizes = {batch, n_frames, n_fft};
    std::vector<c10::SymInt> sym_strides = {self_.stride(0), hop_length_value * self_.stride(1), self_.stride(1)};
    if (self.is_complex()) {
        sym_sizes.push_back(2);
        sym_strides.push_back(1);
    }
    grad_input = as_strided_backward_(grad_input, at::TensorGeometry(self_), sym_sizes, sym_strides, c10::nullopt);
    if (self.is_complex()) {
        grad_input = at::view_as_complex(grad_input);
    }
    if (self.dim() == 1) {
        grad_input = grad_input.unsqueeze(0);
    }
    return grad_input;
}
#endif

}

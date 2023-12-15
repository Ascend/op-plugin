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
}

at::Tensor stft(at::Tensor const &self,
                int64_t n_fft,
                c10::optional<int64_t> hop_length_opt,
                c10::optional<int64_t> win_length_opt,
                c10::optional<at::Tensor> const &window_opt,
                bool normalized,
                c10::optional<bool> onesided_opt,
                c10::optional<bool> return_complex_opt)
{
    TORCH_CHECK(self.dim() == 1 || self.dim() == 2, "input should be a 1D or 2D tensor");
    TORCH_CHECK(n_fft > 0 && n_fft <= self.size(-1), "expected: 0 < n_fft < input.size(-1)");

    int64_t hop_length = hop_length_opt.has_value() ? hop_length_opt.value() : n_fft / 4;
    TORCH_CHECK(hop_length > 0, "expected: hop_length > 0");

    if (window_opt.has_value() && win_length_opt.has_value()) {
        TORCH_CHECK(window_opt.value().dim() == 1 && window_opt.value().size(0) == win_length_opt.value(),
                    "expected: window size and win_length should be equal")
    }

    int win_length = win_length_opt.has_value() ? win_length_opt.value() : n_fft;
    TORCH_CHECK(win_length > 0 && win_length <= n_fft, "expected: 0 < win_length <= n_fft");

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
                "output type should be float, double, complex<float> or complex<double>");
    c10::SmallVector<int64_t, SIZE> output_size = get_output_size(return_complex, batch, frames, n);
    at::Tensor output = npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(output_type));

    EXEC_NPU_CMD(aclStft, self, window, output, n_fft, hop_length, win_length, normalized, onesided, return_complex);
    return output;
}
}

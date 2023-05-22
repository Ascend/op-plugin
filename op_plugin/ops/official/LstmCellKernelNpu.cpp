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

#include <torch/csrc/autograd/custom_function.h>

#include "op_plugin/ops/OpInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace op_plugin {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;
using torch::autograd::AutogradContext;
using torch::autograd::Function;

namespace {
std::vector<at::Tensor> lstm_cell_npu_impl(
    const at::Tensor& input,
    const at::Tensor& w_ih,
    const at::Tensor& w_hh,
    const at::Tensor& h,
    const at::Tensor& c,
    const at::Tensor& bias) {
  at::Tensor input_reshape = input.reshape({1, input.size(0), input.size(1)});
  at::Tensor h_reshape = h.reshape({1, h.size(0), h.size(1)});
  at::Tensor c_reshape = c.reshape({1, c.size(0), c.size(1)});
  int64_t num_step = input_reshape.size(0);
  int64_t batch_size = input_reshape.size(1);
  int64_t hidden_size = w_hh.size(1) / 4;

  at::SmallVector<int64_t, SIZE> output_size = {num_step, batch_size, hidden_size};
  at::Tensor y_output = npu_preparation::ApplyTensor(input_reshape, output_size);
  at::Tensor h_output = npu_preparation::ApplyTensor(input_reshape, output_size);
  at::Tensor c_output = npu_preparation::ApplyTensor(input_reshape, output_size);
  at::Tensor i_output = npu_preparation::ApplyTensorWithFormat(input_reshape, output_size, ACL_FORMAT_FRACTAL_NZ);
  at::Tensor j_output = npu_preparation::ApplyTensorWithFormat(input_reshape, output_size, ACL_FORMAT_FRACTAL_NZ);
  at::Tensor f_output = npu_preparation::ApplyTensorWithFormat(input_reshape, output_size, ACL_FORMAT_FRACTAL_NZ);
  at::Tensor o_output = npu_preparation::ApplyTensorWithFormat(input_reshape, output_size, ACL_FORMAT_FRACTAL_NZ);
  at::Tensor tanhc = npu_preparation::ApplyTensorWithFormat(input_reshape, output_size, ACL_FORMAT_FRACTAL_NZ);

  at_npu::native::OpCommand cmd;
  cmd.Name("DynamicRNNV2")
      .Input(input_reshape)
      .Input(w_ih)
      .Input(w_hh);
  if (bias.defined()) {
    cmd.Input(bias);
  } else {
    cmd.Input();
  }
  cmd.Input()
      .Input(h_reshape)
      .Input(c_reshape)
      .Output(y_output)
      .Output(h_output)
      .Output(c_output)
      .Output(i_output)
      .Output(j_output)
      .Output(f_output)
      .Output(o_output)
      .Output(tanhc)
      .Attr("cell_type", (string)"LSTM")
      .Attr("direction", (string)"UNIDIRECTIONAL")
      .Attr("cell_depth", (int64_t)1)
      .Attr("use_peephole", (bool)false)
      .Attr("keep_prob", (float)1.0)
      .Attr("cell_clip", (float)-1.0)
      .Attr("num_proj", (int64_t)0)
      .Attr("time_major", (bool)true)
      .Attr("activation", (string)"tanh")
      .Attr("forget_bias", (float)0.0)
      .Attr("gate_order", (string)"ifco")
      .Run();
  at::Tensor h_out = h_output[0];
  at::Tensor c_out = c_output[0];
  std::vector<at::Tensor> results = {y_output, h_out, c_out, i_output, j_output, f_output, o_output, tanhc};
  return results;
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&> lstm_cell_backward_npu_impl_nocheck(
    at::Tensor& grad_input,
    at::Tensor& grad_wih,
    at::Tensor& grad_whh,
    at::Tensor& grad_bias,
    at::Tensor& grad_ht,
    at::Tensor& grad_ct,
    const at::Tensor& grad_y,
    const at::Tensor& grad_h,
    const at::Tensor& grad_c,
    const at::Tensor& input,
    const at::Tensor& w_ih,
    const at::Tensor& w_hh,
    const at::Tensor& h,
    const at::Tensor& c,
    const at::Tensor& y_output,
    const at::Tensor& h_output,
    const at::Tensor& c_output,
    const at::Tensor& i,
    const at::Tensor& j,
    const at::Tensor& f,
    const at::Tensor& o,
    const at::Tensor& tanhc) {
  at::Tensor seq_length = at::zeros({}, input.options());
  at::Tensor mask = at::zeros({}, input.options().dtype(at::kByte));
  at::Tensor wci = at::zeros({}, input.options());
  at::Tensor wcf = at::zeros({}, input.options());
  at::Tensor wco = at::zeros({}, input.options());
  at_npu::native::OpCommand cmd;
  cmd.Name("DynamicRNNV2Grad")
      .Input(input)
      .Input(w_ih)
      .Input(w_hh)
      .Input(y_output)
      .Input(h)
      .Input(c)
      .Input(h_output)
      .Input(c_output)
      .Input(grad_y)
      .Input(grad_h)
      .Input(grad_c)
      .Input(i)
      .Input(j)
      .Input(f)
      .Input(o)
      .Input(tanhc)
      .Input(seq_length)
      .Input(wci)
      .Input(wcf)
      .Input(wco)
      .Input(mask)
      .Output(grad_wih)
      .Output(grad_whh)
      .Output(grad_bias)
      .Output(grad_input)
      .Output(grad_ht)
      .Output(grad_ct)
      .Attr("cell_type", (string)"LSTM")
      .Attr("direction", (string)"UNIDIRECTIONAL")
      .Attr("cell_depth", (int64_t)1)
      .Attr("use_peephole", (bool)false)
      .Attr("keep_prob", (float)1.0)
      .Attr("cell_clip", (float)-1.0)
      .Attr("num_proj", (int64_t)0)
      .Attr("time_major", (bool)true)
      .Attr("activation", (string)"tanh")
      .Attr("recurrent_activation", (string)"sigmoid")
      .Attr("gate_order", (string)"ifjo")
      .Attr("stateful", (bool)false)
      .Attr("merge_mode", (string)"concat")
      .Run();

  return std::tie(grad_input, grad_wih, grad_whh, grad_bias, grad_ht, grad_ct);
}
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_lstm_cell_backward(
    const c10::optional<at::Tensor>& grad_y_opt,
    const c10::optional<at::Tensor>& grad_h_opt,
    const c10::optional<at::Tensor>& grad_c_opt,
    const at::Tensor& input,
    const at::Tensor& w_ih,
    const at::Tensor& w_hh,
    const at::Tensor& h,
    const at::Tensor& c,
    const at::Tensor& y_output,
    const at::Tensor& h_output,
    const at::Tensor& c_output,
    const at::Tensor& i,
    const at::Tensor& j,
    const at::Tensor& f,
    const at::Tensor& o,
    const at::Tensor& tanhc) {
  const at::Tensor& grad_y_opt_val = c10::value_or_else(grad_y_opt, [] {return at::Tensor();});
  const at::Tensor& grad_h_opt_val = c10::value_or_else(grad_h_opt, [] {return at::Tensor();});
  const at::Tensor& grad_c_opt_val = c10::value_or_else(grad_c_opt, [] {return at::Tensor();});
  auto grad_y = grad_y_opt_val.defined() ? grad_y_opt_val : at::zeros(h.sizes(), h.options());
  auto grad_h = grad_h_opt_val.defined() ? grad_h_opt_val : at::zeros(h.sizes(), h_output.options());
  auto grad_c = grad_c_opt_val.defined() ? grad_c_opt_val : at::zeros(c.sizes(), c_output.options());
  int64_t hidden_size = y_output.size(2);
  at::SmallVector<int64_t, SIZE> output_size = {4 * hidden_size};
  at::Tensor grad_input = npu_preparation::ApplyTensor(input);
  at::Tensor grad_wih = npu_preparation::ApplyTensor(w_ih);
  at::Tensor grad_whh = npu_preparation::ApplyTensor(w_hh);
  at::Tensor grad_bias = npu_preparation::ApplyTensor(i, output_size);
  at::Tensor grad_ht = npu_preparation::ApplyTensor(h);
  at::Tensor grad_ct = npu_preparation::ApplyTensor(c);
  lstm_cell_backward_npu_impl_nocheck(grad_input, grad_wih, grad_whh, grad_bias, grad_ht, grad_ct,
      grad_y, grad_h, grad_c, input, w_ih, w_hh, h, c, y_output, h_output, c_output, i, j, f, o, tanhc);
  return std::tie(grad_input, grad_wih, grad_whh, grad_bias, grad_ht, grad_ct);
}

class NPULstmCellFunction : public torch::autograd::Function<NPULstmCellFunction> {
public:
  static std::vector<at::Tensor> forward(AutogradContext *ctx,
      const at::Tensor& input,
      const at::Tensor& w_ih,
      const at::Tensor& w_hh,
      const at::Tensor& h,
      const at::Tensor& c,
      const c10::optional<at::Tensor>& b_ih_opt,
      const c10::optional<at::Tensor>& b_hh_opt) {
    at::AutoNonVariableTypeMode g;
    const at::Tensor& b_ih = c10::value_or_else(b_ih_opt, [] {return at::Tensor();});
    const at::Tensor& b_hh = c10::value_or_else(b_hh_opt, [] {return at::Tensor();});
    at::Tensor bias;
    if (b_ih.defined()) {
      bias = at::add(b_ih, b_hh).to(input.dtype());
    }
    auto results = lstm_cell_npu_impl(input, w_ih, w_hh, h, c, bias);
    ctx->save_for_backward({input, w_ih, w_hh, h, c,
        results[0], results[1], results[2], results[3], results[4], results[5], results[6], results[7]});
    return results;
  }

  static std::vector<at::Tensor> backward(AutogradContext* ctx,
      std::vector<at::Tensor> grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto w_ih = saved[1];
    auto w_hh = saved[2];
    auto h = saved[3];
    auto c = saved[4];
    auto y_output = saved[5];
    auto h_output = saved[6];
    auto c_output = saved[7];
    auto i = saved[8];
    auto j = saved[9];
    auto f = saved[10];
    auto o = saved[11];
    auto tanhc = saved[12];

    auto results = op_plugin::npu_lstm_cell_backward(
        grad_outputs[0], grad_outputs[1], grad_outputs[2], input, w_ih,
        w_hh, h, c, y_output, h_output, c_output, i, j, f, o, tanhc);

    std::vector<at::Tensor> output_list = {
        std::get<0>(results),
        std::get<1>(results),
        std::get<2>(results),
        std::get<4>(results),
        std::get<5>(results),
        std::get<3>(results),
        std::get<3>(results)};
    return output_list;
  }
};

std::vector<at::Tensor> npu_lstm_cell(
    const at::Tensor& input,
    const at::Tensor& w_ih,
    const at::Tensor& w_hh,
    const at::Tensor& h,
    const at::Tensor& c,
    const c10::optional<at::Tensor>& b_ih_opt,
    const c10::optional<at::Tensor>& b_hh_opt) {
  return NPULstmCellFunction::apply(input, w_ih, w_hh, h, c, b_ih_opt, b_hh_opt);
}

std::tuple<at::Tensor, at::Tensor> lstm_cell(
    const at::Tensor& input,
    at::TensorList hx,
    const at::Tensor& w_ih,
    const at::Tensor& w_hh,
    const c10::optional<at::Tensor>& b_ih_opt,
    const c10::optional<at::Tensor>& b_hh_opt) {
  at::Tensor weight_ih = w_ih.t().to(input.dtype());
  at::Tensor weight_hh = w_hh.t().to(input.dtype());
  at::Tensor h = hx[0];
  at::Tensor c = hx[1];
  auto result = op_plugin::npu_lstm_cell(input, weight_ih, weight_hh, h, c, b_ih_opt, b_hh_opt);
  std::tuple<at::Tensor, at::Tensor> output(result[1], result[2]);
  return output;
}
}  // namespace op_plugin

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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using torch::autograd::AutogradContext;
using npu_preparation = at_npu::native::OpPreparation;
using calcu_op_util = at_npu::native::CalcuOpUtil;

namespace{
std::vector<at::Tensor> npu_lstm_npu_nocheck(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& seq_mask,
    const at::Tensor& h,
    const at::Tensor& c,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first,
    bool flag_seq,
    bool flag_direction) {
  int64_t num_step = input.size(0);
  int64_t batch_size = input.size(1);
  int64_t hidden_size = bias.size(0) / 4;

  c10::SmallVector<int64_t, SIZE> output_size = {num_step, batch_size, hidden_size};

  at::Tensor y_output = npu_preparation::apply_tensor(input, output_size);
  at::Tensor h_output = npu_preparation::apply_tensor(input, output_size);
  at::Tensor c_output = npu_preparation::apply_tensor(input, output_size);
  at::Tensor i_output = npu_preparation::apply_tensor_with_format(input, output_size, ACL_FORMAT_FRACTAL_NZ);
  at::Tensor j_output = npu_preparation::apply_tensor_with_format(input, output_size, ACL_FORMAT_FRACTAL_NZ);
  at::Tensor f_output = npu_preparation::apply_tensor_with_format(input, output_size, ACL_FORMAT_FRACTAL_NZ);
  at::Tensor o_output = npu_preparation::apply_tensor_with_format(input, output_size, ACL_FORMAT_FRACTAL_NZ);
  at::Tensor tanhc = npu_preparation::apply_tensor_with_format(input, output_size, ACL_FORMAT_FRACTAL_NZ); 
 
  string direction = flag_direction? "REDIRECTIONAL" : "UNIDIRECTIONAL";
  string gate_order = "ifjo";
  at_npu::native::OpCommand cmd;
  cmd.Name("DynamicRNN")
      .Input(input, "x")
      .Input(weight, "w")
      .Input(bias, "b");

  // if input is PackSequence, seq_mask is not None, Otherwise, it is None.
  if (!flag_seq) {
    cmd.Input();
  } else{
    cmd.Input(seq_mask, "seq_length");
  }
  cmd.Input(h, "init_h")
      .Input(c, "init_c")
      .Output(y_output)
      .Output(h_output)
      .Output(c_output)
      .Output(i_output)
      .Output(j_output)
      .Output(f_output)
      .Output(o_output)
      .Output(tanhc)
      .Attr("cell_type", (string)"LSTM")
      .Attr("direction", direction)
      .Attr("cell_depth", (int64_t)1)
      .Attr("use_peephole", (bool)false)
      .Attr("keep_prob", (float)1.0)
      .Attr("cell_clip", (float)-1.0)
      .Attr("num_proj", (int64_t)0)
      .Attr("time_major", (bool)true)
      .Attr("activation", (string)"tanh")
      .Attr("forget_bias", (float)0.0)
      .Attr("is_training", train)
      .Attr("gate_order", gate_order)
      .Run();
  std::vector<at::Tensor> results = {y_output, h_output, c_output, i_output, j_output, f_output, o_output, tanhc};
  return results;
}

std::tuple<at::Tensor, at::Tensor> get_wb_single_layer_direc(
    const at::Tensor& input,
    at::TensorList params,
    bool has_biases) {
  // get weight
  at::Tensor ih_weight = params[0];
  at::Tensor hh_weight = params[1];	
  at::Tensor weight = at::cat({ih_weight, hh_weight}, 1).t().to(input.dtype());
  
  // get bias
  at::Tensor bias = at::zeros(weight.size(1), weight.options());
  if (has_biases) {
    bias = at::add(params[2], params[3]).to(input.dtype());
  }
  return std::tie(weight, bias);
}

std::tuple<at::Tensor, at::Tensor> get_wb_double_layer_or_bidirec(
    const at::Tensor& input,
    at::TensorList params,
    bool has_biases) {
  at::Tensor weight;
  at::Tensor bias; 
  if (has_biases) {
    weight = at::cat({params[4], params[5]}, 1).t().to(input.dtype());
    bias = at::add(params[6], params[7]).to(input.dtype());
  } else {
    weight = at::cat({params[2], params[3]}, 1).t().to(input.dtype());
    bias = at::zeros(weight.size(1), weight.options());
  }
  return std::tie(weight, bias);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_single_layer_direc_npu(
    const at::Tensor& input,
    at::TensorList hx,
    at::TensorList params,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first,
    bool direction) {
  int64_t num_step = input.size(0);
  
  // get weight
  at::Tensor ih_weight = params[0];
  at::Tensor hh_weight = params[1];
	
  at::Tensor weight = at::cat({ih_weight, hh_weight}, 1).t().to(input.dtype());
  
  // get bias
  at::Tensor bias = at::zeros(weight.size(1), weight.options());
  if (has_biases) {
    bias = at::add(params[2], params[3]).to(input.dtype());
  }
  
  // get init_h, init_c
  at::Tensor h = hx[0];
  at::Tensor c = hx[1];
  
  at::Tensor seq_mask = at::empty({0}, input.options());
  auto results = acl_op::npu_lstm(input, weight, bias, seq_mask, h, c, has_biases, num_layers, dropout,
      train, bidirectional, batch_first, false, direction);

  // get the last dimension of the T-axis	
  at::Tensor th_output = at::unsqueeze(results[1][num_step-1], 0);
  at::Tensor tc_output = at::unsqueeze(results[2][num_step-1], 0);

  return std::tie(results[0], th_output, tc_output);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_single_layer_bidirec_npu(
    const at::Tensor& input,
    at::TensorList hx,
    at::TensorList params,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first) {
  int64_t num_step = input.size(0);
  // get h and c of forward direction
  at::Tensor h = hx[0].slice(0, 0, 1);
  at::Tensor c = hx[1].slice(0, 0, 1);
  // caculate forward direction, direction of attr is UNIDIRECTIONAL(npu_lstm need add the attr of direction)
  auto results_forward = lstm_single_layer_direc_npu(input, {h, c}, params, has_biases,
      num_layers, dropout, train, bidirectional, batch_first, false);

  // get w/ b/ h/ c of backward direction
  at::Tensor weight_back;
  at::Tensor bias_back;
  at::Tensor h_back = hx[0].slice(0, 1, 2);
  at::Tensor c_back = hx[1].slice(0, 1, 2);
  std::tie(weight_back, bias_back) = get_wb_double_layer_or_bidirec(input, params, has_biases);

  at::Tensor seq_mask = at::empty({0}, input.options());
  auto rev_inputs = at::flip(input, {0});

  // caculate backward direction, direction of attr is REDIRECTIONAL,
  // but the inverse operator does not support the specified direction,
  // it is necessary to flip the input and output at the adaptation layer.
  auto results_backward = acl_op::npu_lstm(rev_inputs, weight_back, bias_back, seq_mask, h_back, c_back,
      has_biases, num_layers, dropout, train, bidirectional, batch_first, false, false);

  // get the first dimension of the T-axis when caculate reverse direction
  at::Tensor revY = at::flip(results_backward[0],{0});
  at::Tensor th = at::flip(results_backward[1],{0});
  at::Tensor tc = at::flip(results_backward[2],{0});
  at::Tensor th_output = at::unsqueeze(th[0], 0);
  at::Tensor tc_output = at::unsqueeze(tc[0], 0);

  at::Tensor y = at::cat({std::get<0>(results_forward), revY}, 2);
  at::Tensor h_out = at::cat({std::get<1>(results_forward), th_output}, 0);
  at::Tensor c_out = at::cat({std::get<2>(results_forward), tc_output}, 0);

  return std::tie(y, h_out, c_out);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_double_layer_direc_npu(
    const at::Tensor& input,
    at::TensorList hx,
    at::TensorList params,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first) {
  int64_t num_step = input.size(0);
  // get h and c of first layer
  at::Tensor h = hx[0].slice(0, 0, 1);
  at::Tensor c = hx[1].slice(0, 0, 1);

  // caculate first layer
  auto results = lstm_single_layer_direc_npu(input, {h, c}, params, has_biases,
      num_layers, dropout, train, bidirectional, batch_first, false);

  // get w/ b/ h/ c of twice layer
  at::Tensor weight_2_layer;
  at::Tensor bias_2_layer;
  at::Tensor h_2_layer = hx[0].slice(0, 1, 2);
  at::Tensor c_2_layer = hx[1].slice(0, 1, 2);
  std::tie(weight_2_layer, bias_2_layer) = get_wb_double_layer_or_bidirec(input, params, has_biases);

  // output of first layer as input of second layer
  at::Tensor input_2_layer = std::get<0>(results);
  at::Tensor seq_mask = at::empty({0}, input.options());

  // caculate output of second layer
  auto results_2_layer = acl_op::npu_lstm(input_2_layer, weight_2_layer, bias_2_layer, seq_mask, h_2_layer, c_2_layer,
      has_biases, num_layers, dropout, train, bidirectional, batch_first, false, false);
  at::Tensor th_output_2_layer = at::unsqueeze(results_2_layer[1][num_step-1], 0);
  at::Tensor tc_output_2_layer = at::unsqueeze(results_2_layer[2][num_step-1], 0);
  at::Tensor th = at::cat({std::get<1>(results), th_output_2_layer}, 0);
  at::Tensor tc = at::cat({std::get<2>(results), tc_output_2_layer}, 0);

  return std::tie(results_2_layer[0], th, tc);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_double_layer_bidirec_npu(
    const at::Tensor& input,
    at::TensorList hx,
    at::TensorList params,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first) {
  int64_t num_step = input.size(0);

  // get h and c of first layer 
  at::Tensor hL0 = hx[0].slice(0, 0, 2);
  at::Tensor cL0 = hx[1].slice(0, 0, 2);

  // get h and c of second layer
  at::Tensor hL1 = hx[0].slice(0, 2, 4);
  at::Tensor cL1 = hx[1].slice(0, 2, 4);

  // first Single-layer bidirectional LSTM
  auto results_layer1 = lstm_single_layer_bidirec_npu(input, {hL0, cL0}, params, has_biases,
      num_layers, dropout, train, bidirectional, batch_first);

  // second Single-layer bidirectional LSTM, output of Single-layer bidirectional LSTM as input of second layer
  at::Tensor inputLayer2 = std::get<0>(results_layer1);
  at::Tensor y;
  at::Tensor h;
  at::Tensor c;
  if (has_biases) {
    std::tie(y, h, c) = lstm_single_layer_bidirec_npu(inputLayer2, {hL1, cL1}, params.slice(8, 8),
        has_biases, num_layers, dropout, train, bidirectional, batch_first);
  } else {
    std::tie(y, h, c) = lstm_single_layer_bidirec_npu(inputLayer2, {hL1, cL1}, params.slice(4, 4),
        has_biases, num_layers, dropout, train, bidirectional, batch_first);
  }

  at::Tensor th = at::cat({std::get<1>(results_layer1), h}, 0);
  at::Tensor tc = at::cat({std::get<2>(results_layer1), c}, 0);
  return std::tie(y, th, tc);
}

at::Tensor get_mask(const at::Tensor& input, const at::Tensor& batch_sizes, const at::Tensor& h, int64_t max_len) {
  // caculate lengths, but input expected to be sorted
  std::vector<int64_t> lens;
  for (int64_t i = 0; i < input.size(1); ++i) {
    auto batch_sizes_temp = at::sub(batch_sizes , i);
    auto batch_sizes_bool = at::gt(batch_sizes_temp, 0);
    auto batch_sizes_int = batch_sizes_bool.to(at::ScalarType::Int);
    auto cout_len = at::sum(batch_sizes_int, at::ScalarType::Int);
    int64_t len = cout_len.item().toInt();
    lens.emplace_back(len);
  }
  at::Tensor length = calcu_op_util::CopyTensorHostToDevice(
      at::from_blob(lens.data(), {lens.size()}, at::kLong));

  c10::SmallVector<at::Tensor, N> mask_list;
  // Slice by T axis
  for (int64_t i = 0; i < max_len; ++i) {
    at::Tensor maskTemp1 = at::gt(length, i);
    at::Tensor maskTemp2 = maskTemp1.reshape({1, input.size(1), 1});
    // mask need to be expanded to (1,batch_size,hidden_size)
    at::Tensor mask_expand = maskTemp2.expand({1, input.size(1), h.size(2)});
    mask_list.emplace_back(mask_expand);
  }
  at::Tensor mask = at::cat(mask_list, 0).to(at::ScalarType::Half);
  return mask;
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&> lstm_backward_out_npu_nocheck(
    at::Tensor& dw,
    at::Tensor& db,
    at::Tensor& dx,
    at::Tensor& dht,
    at::Tensor& dct,
    const at::Tensor& x,
    const at::Tensor& w,
    const at::Tensor& b,
    const at::Tensor& init_h,
    const at::Tensor& init_c,
    const at::Tensor& dy,
    const at::Tensor& dh,
    const at::Tensor& dc,
    const at::Tensor& y,
    const at::Tensor& h,
    const at::Tensor& c,
    const at::Tensor& i,
    const at::Tensor& j,
    const at::Tensor& f,
    const at::Tensor& o,
    const at::Tensor& tanhc,
    const c10::optional<at::Tensor>& batch_sizes_ = c10::nullopt) {
  const at::Tensor& batch_sizes = c10::value_or_else(batch_sizes_, [] {return at::Tensor();});
  at::Tensor seqmask_h = at::unsqueeze(init_h, 0);
  at::Tensor seq_length = batch_sizes.defined() ?
      get_mask(x, batch_sizes, seqmask_h, batch_sizes.size(0)) : at::zeros({}, x.options());
  at::Tensor mask = at::zeros({}, x.options().dtype(at::kByte));
  at::Tensor wci = at::zeros({}, x.options());
  at::Tensor wcf = at::zeros({}, x.options());
  at::Tensor wco = at::zeros({}, x.options());
  string gate_order = "ifjo";

  at_npu::native::OpCommand cmd;
  cmd.Name("DynamicRNNGrad")
      .Input(x)
      .Input(w)
      .Input(b)
      .Input(y)
      .Input(init_h)
      .Input(init_c)
      .Input(h)
      .Input(c)
      .Input(dy)
      .Input(dh)
      .Input(dc)
      .Input(i)
      .Input(j)
      .Input(f)
      .Input(o)
      .Input(tanhc)
      .Input(seq_length)
      .Input(mask)
      .Input(wci)
      .Input(wcf)
      .Input(wco)
      .Output(dw)
      .Output(db)
      .Output(dx)
      .Output(dht)
      .Output(dct)
      .Attr("cell_type", "LSTM")
      .Attr("direction", "UNIDIRECTIONAL")
      .Attr("cell_depth", (int64_t)0)
      .Attr("use_peephole", (bool)false)
      .Attr("keep_prob", (float)-1.0)
      .Attr("cell_clip", (float)-1.0)
      .Attr("num_proj", (int64_t)0)
      .Attr("time_major", (bool)true)
      .Attr("forget_bias", (float)0.0)
      .Attr("gate_order", gate_order)
      .Run();

  return std::tie(dx, dw, db, dht, dct);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_lstm_data_backward(
    const c10::optional<at::Tensor>& grady_opt,
    const c10::optional<at::Tensor>& gradh_opt,
    const c10::optional<at::Tensor>& gradc_opt,
    const at::Tensor& input,
    const at::Tensor& batch_sizes,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& init_h,
    const at::Tensor& init_c,
    const at::Tensor& y,
    const at::Tensor& h,
    const at::Tensor& c,
    const at::Tensor& i,
    const at::Tensor& j,
    const at::Tensor& f,
    const at::Tensor& o,
    const at::Tensor& tanhc) {
  const at::Tensor& grady = c10::value_or_else(grady_opt, [] {return at::Tensor();});
  const at::Tensor& gradh = c10::value_or_else(gradh_opt, [] {return at::Tensor();});
  const at::Tensor& gradc = c10::value_or_else(gradc_opt, [] {return at::Tensor();});

  at::Tensor inh = at::squeeze(init_h, 0);
  at::Tensor inc = at::squeeze(init_c, 0);

  at::Tensor grad_input = npu_preparation::apply_tensor(input);
  at::Tensor grad_weight = npu_preparation::apply_tensor(weight);
  at::Tensor grad_bias = npu_preparation::apply_tensor(bias);
  at::Tensor grad_ht = npu_preparation::apply_tensor(inh);
  at::Tensor grad_ct = npu_preparation::apply_tensor(inc);

  auto grad_y = grady.defined() ? grady : at::zeros(y.sizes(), y.options());
  auto grad_h = gradh.defined() ? gradh[input.size(0)-1] : at::zeros(inh.sizes(), h.options());
  auto grad_c = gradc.defined() ? gradc[input.size(0)-1] : at::zeros(inc.sizes(), c.options());

  lstm_backward_out_npu_nocheck(grad_weight, grad_bias, grad_input, grad_ht, grad_ct, input, weight,
                        bias, inh, inc, grad_y, grad_h, grad_c, y, h, c, i, j, f, o, tanhc, batch_sizes);
  grad_ht = at::unsqueeze(grad_ht, 0);
  grad_ct = at::unsqueeze(grad_ct, 0);

  return std::tie(grad_input, grad_weight, grad_bias, grad_ht, grad_ct);
}

class NPULstmDataFunction : public torch::autograd::Function<NPULstmDataFunction> {
public:
  static std::vector<at::Tensor> forward(AutogradContext *ctx,
      const at::Tensor& input,
      const at::Tensor& batch_sizes,
      const at::Tensor& weight,
      const at::Tensor& bias,
      const at::Tensor& seq_mask,
      const at::Tensor& h,
      const at::Tensor& c,
      bool has_biases,
      int64_t num_layers,
      double dropout,
      bool train,
      bool bidirectional,
      bool batch_first,
      bool flag_seq,
      bool flag_direction) {
    at::AutoNonVariableTypeMode g;
    auto result = npu_lstm_npu_nocheck(input, weight, bias, seq_mask, h, c, has_biases, num_layers,
        dropout, train, bidirectional, batch_first, flag_seq, flag_direction);
    ctx->save_for_backward({input, batch_sizes, weight, bias, h, c,
        result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7]});
    return result;
  }

  static std::vector<at::Tensor> backward(AutogradContext *ctx,
    std::vector<at::Tensor> grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto batch_sizes = saved[1];
    auto weight = saved[2];
    auto bias = saved[3];
    auto h = saved[4];
    auto c = saved[5];
    auto result0 = saved[6];
    auto result1 = saved[7];
    auto result2 = saved[8];
    auto result3 = saved[9];
    auto result4 = saved[10];
    auto result5 = saved[11];
    auto result6 = saved[12];
    auto result7 = saved[13];

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> result = npu_lstm_data_backward(
        grad_outputs[0], grad_outputs[1], grad_outputs[2], input, batch_sizes, weight, bias, h, c,
        result0, result1, result2, result3, result4, result5, result6, result7);
    std::vector<at::Tensor> output = {
        std::get<0>(result),
        at::Tensor(),
        std::get<1>(result),
        std::get<2>(result),
        at::Tensor(),
        std::get<3>(result),
        std::get<4>(result),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor()};
    return output;
  }
};

std::vector<at::Tensor> npu_lstm_data(
    const at::Tensor& input,
    const at::Tensor& batch_sizes,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& seq_mask,
    const at::Tensor& h,
    const at::Tensor& c,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first,
    bool flag_seq,
    bool flag_direction){
  return NPULstmDataFunction::apply(input, batch_sizes, weight, bias, seq_mask, h, c, has_biases,
      num_layers, dropout, train, bidirectional, batch_first, flag_seq, flag_direction);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_onelayer_direc_packseq(
    const at::Tensor& data, const at::Tensor& batch_sizes, at::TensorList hx,
    at::TensorList params, bool has_biases, int64_t num_layers, double dropout_p,
    bool train, bool bidirectional) {

  int64_t t_size = batch_sizes.numel();
  TORCH_CHECK(t_size > 0, "lstm_onelayer_direc_packseq: t_size is zero!");

  at::Tensor input = data.reshape({t_size, data.size(0) / t_size, data.size(1)});

  bool batch_first = false;
  at::Tensor h = hx[0];
  at::Tensor c = hx[1];

  int64_t num_step = input.size(0);

  at::Tensor ih_weight = params[0];
  at::Tensor hh_weight = params[1];	
  at::Tensor weight = at::cat({ih_weight, hh_weight}, 1).t().to(input.dtype());

  at::Tensor bias = at::zeros(weight.size(1), weight.options());
  if (has_biases) {
    bias = at::add(params[2], params[3]).to(input.dtype());
  }

  int64_t max_len = input.size(0);

  at::Tensor mask = get_mask(input, batch_sizes, h, max_len);
  auto results = npu_lstm_data(input, batch_sizes, weight, bias, mask, h, c, has_biases, num_layers,
      dropout_p, train, bidirectional, false, true, false);

  at::Tensor th_output = at::unsqueeze(results[1][num_step-1], 0);
  at::Tensor tc_output = at::unsqueeze(results[2][num_step-1], 0);

  return std::tuple<at::Tensor, at::Tensor, at::Tensor>(results[0], th_output, tc_output);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_onelayer_bidirec_packseq(
    const at::Tensor& data,
    const at::Tensor& batch_sizes,
    at::TensorList hx,
    at::TensorList params,
    bool has_biases,
    int64_t num_layers,
    double dropout_p,
    bool train,
    bool bidirectional) {
  int64_t t_size = batch_sizes.numel();
  TORCH_CHECK(t_size > 0, "lstm_onelayer_bidirec_packseq: t_size is zero!");

  at::Tensor input = data.reshape({t_size, data.size(0) / t_size, data.size(1)});
  bool batch_first = false;

  at::Tensor h = hx[0].slice(0, 0, 1);
  at::Tensor c = hx[1].slice(0, 0, 1);

  auto results_forward = lstm_onelayer_direc_packseq(data, batch_sizes, {h, c}, params, has_biases,
      num_layers, dropout_p, train, bidirectional);

  // get w/ b/ h/ c of backward direction
  at::Tensor h_back = hx[0].slice(0, 1, 2);
  at::Tensor c_back = hx[1].slice(0, 1, 2);
  
  at::Tensor weight_back;
  at::Tensor bias_back;
  std::tie(weight_back, bias_back) = get_wb_double_layer_or_bidirec(input, params, has_biases);

  int64_t max_len = input.size(0);

  at::Tensor mask = get_mask(input, batch_sizes, h, max_len);
  // caculate forward direction, direction of attr is REDIRECTIONAL
  auto results_backward = npu_lstm_data(input, batch_sizes, weight_back, bias_back, mask, h_back, c_back,
      has_biases, num_layers, dropout_p, train, bidirectional, batch_first, true, true);

  // get the first dimension of the T-axis when caculate reverse direction	
  at::Tensor th_output = at::unsqueeze(results_backward[1][0], 0);
  at::Tensor tc_output = at::unsqueeze(results_backward[2][0], 0);

  at::Tensor y = at::cat({std::get<0>(results_forward), results_backward[0]}, 2);
  at::Tensor h_out = at::cat({std::get<1>(results_forward), th_output}, 0);
  at::Tensor c_out = at::cat({std::get<2>(results_forward), tc_output}, 0);

  return std::tie(y, h_out, c_out);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_double_layer_direc_packseq(
    const at::Tensor& data,
    const at::Tensor& batch_sizes,
    at::TensorList hx,
    at::TensorList params,
    bool has_biases,
    int64_t num_layers,
    double dropout_p,
    bool train,
    bool bidirectional) {
  int64_t t_size = batch_sizes.numel();
  TORCH_CHECK(t_size > 0, "lstm_double_layer_direc_packseq: t_size is zero!");

  at::Tensor input = data.reshape({t_size, data.size(0) / t_size, data.size(1)});

  bool batch_first = false;

  at::Tensor h = hx[0].slice(0, 0, 1);
  at::Tensor c = hx[1].slice(0, 0, 1);

  int64_t num_step = input.size(0);

  auto results = lstm_onelayer_direc_packseq(data, batch_sizes, {h, c}, params, has_biases,
      num_layers, dropout_p, train, bidirectional);

  at::Tensor weight_2_layer;
  at::Tensor bias_2_layer;
  at::Tensor h_2_layer = hx[0].slice(0, 1, 2);
  at::Tensor c_2_layer = hx[1].slice(0, 1, 2);
  std::tie(weight_2_layer, bias_2_layer) = get_wb_double_layer_or_bidirec(input, params, has_biases);

  int64_t max_len = input.size(0);
  at::Tensor mask = get_mask(input, batch_sizes, h, max_len);
  at::Tensor input_2_layer = std::get<0>(results);

  auto results_2_layer = npu_lstm_data(input_2_layer, batch_sizes, weight_2_layer, bias_2_layer, mask, h_2_layer, c_2_layer,
      has_biases, num_layers, dropout_p, train, bidirectional, batch_first, true, false);
  at::Tensor th_output_2_layer = at::unsqueeze(results_2_layer[1][num_step - 1], 0);
  at::Tensor tc_output_2_layer = at::unsqueeze(results_2_layer[2][num_step - 1], 0);
  at::Tensor th = at::cat({std::get<1>(results), th_output_2_layer}, 0);
  at::Tensor tc = at::cat({std::get<2>(results), tc_output_2_layer}, 0);

  return std::tie(results_2_layer[0], th, tc);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_double_layer_bidirec_packseq(
    const at::Tensor& data,
    const at::Tensor& batch_sizes,
    at::TensorList hx,
    at::TensorList params,
    bool has_biases,
    int64_t num_layers,
    double dropout_p,
    bool train,
    bool bidirectional) {
  int64_t t_size = batch_sizes.numel();
  TORCH_CHECK(t_size > 0, "batch_sizes can not be empty.");

  at::Tensor input = data.reshape({t_size, data.size(0) / t_size, data.size(1)});
  bool batch_first = false;

  at::Tensor hL0 = hx[0].slice(0, 0, 2);
  at::Tensor cL0 = hx[1].slice(0, 0, 2);

  at::Tensor hL1 = hx[0].slice(0, 2, 4);
  at::Tensor cL1 = hx[1].slice(0, 2, 4);

  auto results_layer1 = lstm_onelayer_bidirec_packseq(data, batch_sizes, {hL0, cL0}, params, has_biases,
      num_layers, dropout_p, train, bidirectional);

  // second Single-layer bidirectional LSTM, output of Single-layer bidirectional LSTM as input of second layer
  at::Tensor inputLayer2 = std::get<0>(results_layer1);
  at::Tensor dataLayer2 = inputLayer2.contiguous().view({-1, inputLayer2.size(2)});
  at::Tensor y;
  at::Tensor h;
  at::Tensor c;
  if (has_biases) {
    std::tie(y, h, c) = lstm_onelayer_bidirec_packseq(dataLayer2, batch_sizes, {hL1, cL1}, params.slice(8, 8),
        has_biases, num_layers, dropout_p, train, bidirectional);
  } else {
    std::tie(y, h, c) = lstm_onelayer_bidirec_packseq(dataLayer2, batch_sizes, {hL1, cL1}, params.slice(4, 4),
        has_biases, num_layers, dropout_p, train, bidirectional);
  }

  at::Tensor th = at::cat({std::get<1>(results_layer1), h}, 0);
  at::Tensor tc = at::cat({std::get<2>(results_layer1), c}, 0);
  return std::tie(y, th, tc);
}
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm(
    const at::Tensor& input,
    at::TensorList hx,
    at::TensorList params,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first) {
  // The operator of DynamicRnn only supports the T axis as the first axis.
  auto input_trans = batch_first ? input.transpose(0, 1) : input;
  at::Tensor y;
  at::Tensor h;
  at::Tensor c;

  if (num_layers == 1) {
    if (!bidirectional) {
      std::tie(y, h, c) = lstm_single_layer_direc_npu(input_trans, hx, params, has_biases, num_layers,
          dropout, train, bidirectional, batch_first, false);
    } else {
      std::tie(y, h, c) = lstm_single_layer_bidirec_npu(input_trans, hx, params, has_biases, num_layers,
          dropout, train, bidirectional, batch_first);
    }
  }

  if (num_layers == 2) {
    if (!bidirectional) {
      std::tie(y, h, c) = lstm_double_layer_direc_npu(input_trans, hx, params, has_biases, num_layers,
          dropout, train, bidirectional, batch_first);
    } else {
      std::tie(y, h, c) = lstm_double_layer_bidirec_npu(input_trans, hx, params, has_biases, num_layers,
          dropout, train, bidirectional, batch_first);
    }
  }

  // the Bacth axis of output should be first axis when batch_first is True!
  auto output = batch_first ? y.transpose(0, 1) : y;
  return std::tie(output, h, c);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm(
    const at::Tensor& data,
    const at::Tensor& batch_sizes,
    at::TensorList hx,
    at::TensorList params,
    bool has_biases,
    int64_t num_layers,
    double dropout_p,
    bool train,
    bool bidirectional) {
  at::Tensor batch_sizes_cpu = batch_sizes.to("cpu");
  at::Tensor y;
  at::Tensor h;
  at::Tensor c;

  if (num_layers == 1) {
    if (!bidirectional) {
      std::tie(y, h, c) = lstm_onelayer_direc_packseq(data, batch_sizes_cpu, hx, params, has_biases,
          num_layers, dropout_p, train, bidirectional);
    } else {
      std::tie(y, h, c) = lstm_onelayer_bidirec_packseq(data, batch_sizes_cpu, hx, params, has_biases,
          num_layers, dropout_p, train, bidirectional);
    }
  }

  if (num_layers == 2) {
    if (!bidirectional) {
      std::tie(y, h, c) = lstm_double_layer_direc_packseq(data, batch_sizes_cpu, hx, params, has_biases,
          num_layers, dropout_p, train, bidirectional);
    } else {
      std::tie(y, h, c) = lstm_double_layer_bidirec_packseq(data, batch_sizes_cpu, hx, params, has_biases,
          num_layers, dropout_p, train, bidirectional);
    }
  }
  return std::tie(y, h, c);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_lstm_backward(
    const c10::optional<at::Tensor>& grady_opt,
    const c10::optional<at::Tensor>& gradh_opt,
    const c10::optional<at::Tensor>& gradc_opt,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& init_h,
    const at::Tensor& init_c,
    const at::Tensor& y,
    const at::Tensor& h,
    const at::Tensor& c,
    const at::Tensor& i,
    const at::Tensor& j,
    const at::Tensor& f,
    const at::Tensor& o,
    const at::Tensor& tanhc) {
  const at::Tensor& grady = c10::value_or_else(grady_opt, [] {return at::Tensor();});
  const at::Tensor& gradh = c10::value_or_else(gradh_opt, [] {return at::Tensor();});
  const at::Tensor& gradc = c10::value_or_else(gradc_opt, [] {return at::Tensor();});

  at::Tensor inh = at::squeeze(init_h, 0);
  at::Tensor inc = at::squeeze(init_c, 0);

  at::Tensor grad_input = npu_preparation::apply_tensor(input); 
  at::Tensor grad_weight = npu_preparation::apply_tensor(weight);
  at::Tensor grad_bias = npu_preparation::apply_tensor(bias);
  at::Tensor grad_ht = npu_preparation::apply_tensor(inh);
  at::Tensor grad_ct = npu_preparation::apply_tensor(inc);
  
  auto grad_y = grady.defined() ? grady : at::zeros(y.sizes(), y.options());
  auto grad_h = gradh.defined() ? gradh[input.size(0)-1] : at::zeros(inh.sizes(), h.options());
  auto grad_c = gradc.defined() ? gradc[input.size(0)-1] : at::zeros(inc.sizes(), c.options());

  lstm_backward_out_npu_nocheck(grad_weight, grad_bias, grad_input, grad_ht, grad_ct, input, weight,
      bias, inh, inc, grad_y, grad_h, grad_c, y, h, c, i, j, f, o, tanhc);
  grad_ht = at::unsqueeze(grad_ht, 0);
  grad_ct = at::unsqueeze(grad_ct, 0);

  return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>{
      grad_input, grad_weight, grad_bias, grad_ht, grad_ct};
}

class NPULstmFunction : public torch::autograd::Function<NPULstmFunction> {
public:
  static std::vector<at::Tensor> forward(AutogradContext *ctx,
      const at::Tensor& input,
      const at::Tensor& weight,
      const at::Tensor& bias,
      const at::Tensor& seq_mask,
      const at::Tensor& h,
      const at::Tensor& c,
      bool has_biases,
      int64_t num_layers,
      double dropout,
      bool train,
      bool bidirectional,
      bool batch_first,
      bool flag_seq,
      bool flag_direction) {
    at::AutoNonVariableTypeMode g;
    auto result = npu_lstm_npu_nocheck(input, weight, bias, seq_mask, h, c, has_biases,
        num_layers, dropout, train, bidirectional, batch_first, flag_seq, flag_direction);
    auto result0 = result[0];
    auto result1 = result[1];
    auto result2 = result[2];
    auto result3 = result[3];
    auto result4 = result[4];
    auto result5 = result[5];
    auto result6 = result[6];
    auto result7 = result[7];
    ctx->save_for_backward(
        {input, weight, bias, h, c, result0, result1, result2, result3, result4, result5, result6, result7});
    return result;
  }

  static std::vector<at::Tensor> backward(AutogradContext* ctx,
      std::vector<at::Tensor> grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto weight = saved[1];
    auto bias = saved[2];
    auto h = saved[3];
    auto c = saved[4];
    auto result0 = saved[5];
    auto result1 = saved[6];
    auto result2 = saved[7];
    auto result3 = saved[8];
    auto result4 = saved[9];
    auto result5 = saved[10];
    auto result6 = saved[11];
    auto result7 = saved[12];

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> result = acl_op::npu_lstm_backward(
        grad_outputs[0], grad_outputs[1], grad_outputs[2], input, weight, bias, h, c,
        result0, result1, result2, result3, result4, result5, result6, result7);
    std::vector<at::Tensor> output = {std::get<0>(result), std::get<1>(result), std::get<2>(result),
        at::Tensor(), std::get<3>(result), std::get<4>(result), at::Tensor(), at::Tensor(), at::Tensor(),
        at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    return output;
  }
};

std::vector<at::Tensor> npu_lstm(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& seq_mask,
    const at::Tensor& h,
    const at::Tensor& c,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first,
    bool flag_seq,
    bool flag_direction) {
  return NPULstmFunction::apply(input, weight, bias, seq_mask, h, c, has_biases,
      num_layers, dropout, train, bidirectional, batch_first, flag_seq, flag_direction);
}
} // namespace acl_op

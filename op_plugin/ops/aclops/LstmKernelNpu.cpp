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

#include "torch_npu/csrc/aten/CustomFunctions.h"

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"
#include "op_plugin/utils/custom_functions/aclops/inner_compute.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using tensor_list =
    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;
using tensor_list3 = std::tuple<at::Tensor, at::Tensor, at::Tensor>;
using tensor_list5 = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

namespace {
tensor_list npu_lstm_npu_nocheck(const at::Tensor &input, const at::Tensor &weight, const at::Tensor &bias,
                                 const at::Tensor &seq_mask, const at::Tensor &h, const at::Tensor &c, bool has_biases,
                                 int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first,
                                 bool flag_seq, bool flag_direction)
{
    int64_t num_step = input.size(0);
    int64_t batch_size = input.size(1);
    int64_t hidden_size = bias.size(0) / 4;

    c10::SmallVector<int64_t, SIZE> output_size = {num_step, batch_size, hidden_size};

    at::Tensor y_output = npu_preparation::apply_tensor(input, output_size);
    at::Tensor h_output = npu_preparation::apply_tensor(input, output_size);
    at::Tensor c_output = npu_preparation::apply_tensor(input, output_size);
    int64_t output_format = train ? ACL_FORMAT_FRACTAL_NZ : ACL_FORMAT_ND;
    at::Tensor i_output = npu_preparation::apply_tensor_with_format(input, output_size, output_format);
    at::Tensor j_output = npu_preparation::apply_tensor_with_format(input, output_size, output_format);
    at::Tensor f_output = npu_preparation::apply_tensor_with_format(input, output_size, output_format);
    at::Tensor o_output = npu_preparation::apply_tensor_with_format(input, output_size, output_format);
    at::Tensor tanhc = npu_preparation::apply_tensor_with_format(input, output_size, output_format);

    string direction = flag_direction ? "REDIRECTIONAL" : "UNIDIRECTIONAL";
    string gate_order = "ifjo";
    at_npu::native::OpCommand cmd;
    cmd.Name("DynamicRNN").Input(input, "x").Input(weight, "w").Input(bias, "b");

    // if input is PackSequence, seq_mask is not None, Otherwise, it is None.
    if (!flag_seq) {
        cmd.Input();
    } else {
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
        .Attr("cell_type", static_cast<std::string>("LSTM"))
        .Attr("direction", direction)
        .Attr("cell_depth", static_cast<int64_t>(1))
        .Attr("use_peephole", static_cast<bool>(false))
        .Attr("keep_prob", static_cast<float>(1.0))
        .Attr("cell_clip", static_cast<float>(-1.0))
        .Attr("num_proj", static_cast<int64_t>(0))
        .Attr("time_major", static_cast<bool>(true))
        .Attr("activation", static_cast<std::string>("tanh"))
        .Attr("forget_bias", static_cast<float>(0.0))
        .Attr("is_training", train)
        .Attr("gate_order", gate_order)
        .Run();
    return std::make_tuple(y_output, h_output, c_output, i_output, j_output, f_output, o_output, tanhc);
}

std::tuple<at::Tensor, at::Tensor> get_wb_single_layer_direc(const at::Tensor &input, at::TensorList params,
                                                             bool has_biases)
{
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

std::tuple<at::Tensor, at::Tensor> get_wb_double_layer_or_bidirec(const at::Tensor &input, at::TensorList params,
                                                                  bool has_biases)
{
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

std::tuple<at::Tensor, at::Tensor> get_wb_multi_layer_or_bidirec(const at::Tensor &input, at::TensorList params,
                                                                 int64_t layers, bool hasBiases)
{
    TORCH_CHECK(layers > 0, "layers should be greater than 0."
        + OPS_ERROR(ErrCode::VALUE));
    at::Tensor weight;
    at::Tensor bias;
    if (hasBiases) {
        weight = at::cat({params[layers * 4 - 4], params[layers * 4 - 3]}, 1).t().to(input.dtype());
        bias = at::add(params[layers * 4 - 2], params[layers * 4 - 1]).to(input.dtype());
    } else {
        weight = at::cat({params[layers * 2 - 2], params[layers * 2 - 1]}, 1).t().to(input.dtype());
        bias = at::zeros(weight.size(1), weight.options());
    }
    return std::tie(weight, bias);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_single_layer_direc_npu(const at::Tensor &input, at::TensorList hx,
                                                                           at::TensorList params, bool has_biases,
                                                                           int64_t num_layers, double dropout,
                                                                           bool train, bool bidirectional,
                                                                           bool batch_first, bool direction)
{
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
    auto results = at_npu::native::custom_ops::npu_lstm(input, weight, bias, seq_mask, h, c, has_biases, num_layers,
                                                        dropout, train, bidirectional, batch_first, false, direction);

    // get the last dimension of the T-axis
    at::Tensor th_output = at::unsqueeze(std::get<1>(results)[num_step - 1], 0);
    at::Tensor tc_output = at::unsqueeze(std::get<2>(results)[num_step - 1], 0);

    return std::tie(std::get<0>(results), th_output, tc_output);
}

tensor_list3 lstm_single_layer_bidirec_npu(const at::Tensor &input, at::TensorList hx, at::TensorList params,
                                           bool has_biases, int64_t num_layers, double dropout, bool train,
                                           bool bidirectional, bool batch_first)
{
    int64_t num_step = input.size(0);
    // get h and c of forward direction
    at::Tensor h = hx[0].slice(0, 0, 1);
    at::Tensor c = hx[1].slice(0, 0, 1);
    // caculate forward direction, direction of attr is UNIDIRECTIONAL(npu_lstm need add the attr of direction)
    auto results_forward = lstm_single_layer_direc_npu(input, {h, c}, params, has_biases, num_layers, dropout, train,
                                                       bidirectional, batch_first, false);

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
    auto results_backward =
        at_npu::native::custom_ops::npu_lstm(rev_inputs, weight_back, bias_back, seq_mask, h_back, c_back, has_biases,
                                             num_layers, dropout, train, bidirectional, batch_first, false, false);

    // get the first dimension of the T-axis when caculate reverse direction
    at::Tensor revY = at::flip(std::get<0>(results_backward), {0});
    at::Tensor th = at::flip(std::get<1>(results_backward), {0});
    at::Tensor tc = at::flip(std::get<2>(results_backward), {0});
    at::Tensor th_output = at::unsqueeze(th[0], 0);
    at::Tensor tc_output = at::unsqueeze(tc[0], 0);

    at::Tensor y = at::cat({std::get<0>(results_forward), revY}, 2);
    at::Tensor h_out = at::cat({std::get<1>(results_forward), th_output}, 0);
    at::Tensor c_out = at::cat({std::get<2>(results_forward), tc_output}, 0);

    return std::tie(y, h_out, c_out);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_multi_layer_direc_npu(const at::Tensor &input, at::TensorList hx,
                                                                          at::TensorList params, bool has_biases,
                                                                          int64_t layers, int64_t num_layers,
                                                                          double dropout, bool train,
                                                                          bool bidirectional, bool batch_first)
{
    TORCH_CHECK(layers > 0, "layers should be greater than 0."
        + OPS_ERROR(ErrCode::VALUE));
    int64_t num_step = input.size(0);
    at::Tensor y;
    at::Tensor h;
    at::Tensor c;
    // caculate first layer
    if (layers == 1) {
        return lstm_single_layer_direc_npu(input, hx, params, has_biases, num_layers, dropout, train, bidirectional,
                                           batch_first, false);
    } else {
        // get h and c of first layer
        at::Tensor h1 = hx[0].slice(0, 0, layers - 1);
        at::Tensor c1 = hx[1].slice(0, 0, layers - 1);
        std::tie(y, h, c) = lstm_multi_layer_direc_npu(input, {h1, c1}, params, has_biases, layers - 1, num_layers,
                                                       dropout, train, bidirectional, batch_first);
    }
    // get w/ b/ h/ c of twice layer
    at::Tensor weight_multi_layer;
    at::Tensor bias_multi_layer;
    at::Tensor h_multi_layer = hx[0].slice(0, layers - 1, layers);
    at::Tensor c_multi_layer = hx[1].slice(0, layers - 1, layers);
    std::tie(weight_multi_layer, bias_multi_layer) = get_wb_multi_layer_or_bidirec(input, params, layers, has_biases);

    at::Tensor seq_mask = at::empty({0}, input.options());
    // caculate output of second layer
    auto results_multi_layer = at_npu::native::custom_ops::npu_lstm(
        y, weight_multi_layer, bias_multi_layer, seq_mask, h_multi_layer, c_multi_layer, has_biases, num_layers,
        dropout, train, bidirectional, batch_first, false, false);
    at::Tensor th_output_multi_layer = at::unsqueeze(std::get<1>(results_multi_layer)[num_step - 1], 0);
    at::Tensor tc_output_multi_layer = at::unsqueeze(std::get<2>(results_multi_layer)[num_step - 1], 0);
    at::Tensor th = at::cat({h, th_output_multi_layer}, 0);
    at::Tensor tc = at::cat({c, tc_output_multi_layer}, 0);

    return std::tie(std::get<0>(results_multi_layer), th, tc);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_multi_layer_bidirec_npu(const at::Tensor &input, at::TensorList hx,
                                                                            at::TensorList params, bool has_biases,
                                                                            int64_t layers, int64_t num_layers,
                                                                            double dropout, bool train,
                                                                            bool bidirectional, bool batch_first)
{
    TORCH_CHECK(layers > 0, "layers should be greater than 0."
        + OPS_ERROR(ErrCode::VALUE));
    int64_t num_step = input.size(0);
    // get h and c of first layer
    at::Tensor hL0 = hx[0].slice(0, 0, 2);
    at::Tensor cL0 = hx[1].slice(0, 0, 2);
    // get h and c of second layer
    at::Tensor hL1 = hx[0].slice(0, 2, layers * 2);
    at::Tensor cL1 = hx[1].slice(0, 2, layers * 2);

    // first Single-layer bidirectional LSTM
    auto results_layer1 = lstm_single_layer_bidirec_npu(input, {hL0, cL0}, params, has_biases, num_layers, dropout,
                                                        train, bidirectional, batch_first);
    if (layers == 1) {
        return results_layer1;
    }

    // second Single-layer bidirectional LSTM, output of Single-layer bidirectional LSTM as input of second layer
    at::Tensor inputLayer2 = std::get<0>(results_layer1);
    at::Tensor y;
    at::Tensor h;
    at::Tensor c;
    if (layers < 3) {
        if (has_biases) {
            std::tie(y, h, c) =
                lstm_single_layer_bidirec_npu(inputLayer2, {hL1, cL1}, params.slice(8, (layers - 1) * 8), has_biases,
                                              num_layers, dropout, train, bidirectional, batch_first);
        } else {
            std::tie(y, h, c) =
                lstm_single_layer_bidirec_npu(inputLayer2, {hL1, cL1}, params.slice(4, (layers - 1) * 4), has_biases,
                                              num_layers, dropout, train, bidirectional, batch_first);
        }
    } else {
        if (has_biases) {
            std::tie(y, h, c) =
                lstm_multi_layer_bidirec_npu(inputLayer2, {hL1, cL1}, params.slice(8, (layers - 1) * 8), has_biases,
                                             layers - 1, num_layers, dropout, train, bidirectional, batch_first);
        } else {
            std::tie(y, h, c) =
                lstm_multi_layer_bidirec_npu(inputLayer2, {hL1, cL1}, params.slice(4, (layers - 1) * 4), has_biases,
                                             layers - 1, num_layers, dropout, train, bidirectional, batch_first);
        }
    }
    at::Tensor th = at::cat({std::get<1>(results_layer1), h}, 0);
    at::Tensor tc = at::cat({std::get<2>(results_layer1), c}, 0);
    return std::tie(y, th, tc);
}

at::Tensor get_mask(const at::Tensor &input, const at::Tensor &batch_sizes, const at::Tensor &h, int64_t max_len)
{
    // caculate lengths, but input expected to be sorted
    std::vector<int64_t> lens;
    for (int64_t i = 0; i < input.size(1); ++i) {
        auto batch_sizes_temp = at::sub(batch_sizes, i);
        auto batch_sizes_bool = at::gt(batch_sizes_temp, 0);
        auto batch_sizes_int = batch_sizes_bool.to(at::ScalarType::Int);
        auto cout_len = at::sum(batch_sizes_int, at::ScalarType::Int);
        int64_t len = cout_len.item().toInt();
        lens.emplace_back(len);
    }
    at::Tensor length =
        npu_preparation::copy_tensor_host_to_device(at::from_blob(lens.data(), {lens.size()}, at::kLong));

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

std::tuple<at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &> lstm_backward_out_npu_nocheck(
    at::Tensor &dw, at::Tensor &db, at::Tensor &dx, at::Tensor &dht, at::Tensor &dct, const at::Tensor &x,
    const at::Tensor &w, const at::Tensor &b, const at::Tensor &init_h, const at::Tensor &init_c, const at::Tensor &dy,
    const at::Tensor &dh, const at::Tensor &dc, const at::Tensor &y, const at::Tensor &h, const at::Tensor &c,
    const at::Tensor &i, const at::Tensor &j, const at::Tensor &f, const at::Tensor &o, const at::Tensor &tanhc,
    bool flag_direction = false, const c10::optional<at::Tensor> &batch_sizes_ = c10::nullopt)
{
    const at::Tensor &batch_sizes = c10::value_or_else(batch_sizes_, [] { return at::Tensor(); });
    at::Tensor seqmask_h = at::unsqueeze(init_h, 0);
    at::Tensor seq_length =
        batch_sizes.defined() ? get_mask(x, batch_sizes, seqmask_h, batch_sizes.size(0)) : at::zeros({}, x.options());
    at::Tensor mask = at::zeros({}, x.options().dtype(at::kByte));
    at::Tensor wci = at::zeros({}, x.options());
    at::Tensor wcf = at::zeros({}, x.options());
    at::Tensor wco = at::zeros({}, x.options());
    string gate_order = "ifjo";
    string direction = flag_direction ? "REDIRECTIONAL" : "UNIDIRECTIONAL";

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
        .Attr("direction", direction)
        .Attr("cell_depth", static_cast<int64_t>(0))
        .Attr("use_peephole", static_cast<bool>(false))
        .Attr("keep_prob", static_cast<float>(-1.0))
        .Attr("cell_clip", static_cast<float>(-1.0))
        .Attr("num_proj", static_cast<int64_t>(0))
        .Attr("time_major", static_cast<bool>(true))
        .Attr("forget_bias", static_cast<float>(0.0))
        .Attr("gate_order", gate_order)
        .Run();

    return std::tie(dx, dw, db, dht, dct);
}

tensor_list3 lstm_onelayer_direc_packseq(const at::Tensor &data, const at::Tensor &batch_sizes, at::TensorList hx,
                                         at::TensorList params, bool has_biases, int64_t num_layers, double dropout_p,
                                         bool train, bool bidirectional)
{
    int64_t t_size = batch_sizes.numel();
    TORCH_CHECK(t_size > 0, "lstm_onelayer_direc_packseq: t_size is zero!"
        + OPS_ERROR(ErrCode::VALUE));

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
    auto results =
        at_npu::native::custom_ops::npu_lstm_data(input, batch_sizes, weight, bias, mask, h, c, has_biases, num_layers,
                                                  dropout_p, train, bidirectional, false, true, false);

    at::Tensor th_output = at::unsqueeze(std::get<1>(results)[num_step - 1], 0);
    at::Tensor tc_output = at::unsqueeze(std::get<2>(results)[num_step - 1], 0);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(std::get<0>(results), th_output, tc_output);
}

tensor_list3 lstm_onelayer_bidirec_packseq(const at::Tensor &data, const at::Tensor &batch_sizes, at::TensorList hx,
                                           at::TensorList params, bool has_biases, int64_t num_layers, double dropout_p,
                                           bool train, bool bidirectional)
{
    int64_t t_size = batch_sizes.numel();
    TORCH_CHECK(t_size > 0, "lstm_onelayer_bidirec_packseq: t_size is zero!"
        + OPS_ERROR(ErrCode::VALUE));

    at::Tensor input = data.reshape({t_size, data.size(0) / t_size, data.size(1)});
    bool batch_first = false;

    at::Tensor h = hx[0].slice(0, 0, 1);
    at::Tensor c = hx[1].slice(0, 0, 1);

    auto results_forward = lstm_onelayer_direc_packseq(data, batch_sizes, {h, c}, params, has_biases, num_layers,
                                                       dropout_p, train, bidirectional);

    // get w/ b/ h/ c of backward direction
    at::Tensor h_back = hx[0].slice(0, 1, 2);
    at::Tensor c_back = hx[1].slice(0, 1, 2);

    at::Tensor weight_back;
    at::Tensor bias_back;
    std::tie(weight_back, bias_back) = get_wb_double_layer_or_bidirec(input, params, has_biases);

    int64_t max_len = input.size(0);

    at::Tensor mask = get_mask(input, batch_sizes, h, max_len);
    // caculate forward direction, direction of attr is REDIRECTIONAL
    auto results_backward = at_npu::native::custom_ops::npu_lstm_data(input, batch_sizes, weight_back, bias_back, mask,
                                                                      h_back, c_back, has_biases, num_layers, dropout_p,
                                                                      train, bidirectional, batch_first, true, true);

    // get the first dimension of the T-axis when caculate reverse direction
    at::Tensor th_output = at::unsqueeze(std::get<1>(results_backward)[0], 0);
    at::Tensor tc_output = at::unsqueeze(std::get<2>(results_backward)[0], 0);

    at::Tensor y = at::cat({std::get<0>(results_forward), std::get<0>(results_backward)}, 2);
    at::Tensor h_out = at::cat({std::get<1>(results_forward), th_output}, 0);
    at::Tensor c_out = at::cat({std::get<2>(results_forward), tc_output}, 0);

    return std::tie(y, h_out, c_out);
}

tensor_list3 lstm_double_layer_direc_packseq(const at::Tensor &data, const at::Tensor &batch_sizes, at::TensorList hx,
                                             at::TensorList params, bool has_biases, int64_t num_layers,
                                             double dropout_p, bool train, bool bidirectional)
{
    int64_t t_size = batch_sizes.numel();
    TORCH_CHECK(t_size > 0, "lstm_double_layer_direc_packseq: t_size is zero!"
        + OPS_ERROR(ErrCode::VALUE));

    at::Tensor input = data.reshape({t_size, data.size(0) / t_size, data.size(1)});

    bool batch_first = false;

    at::Tensor h = hx[0].slice(0, 0, 1);
    at::Tensor c = hx[1].slice(0, 0, 1);

    int64_t num_step = input.size(0);

    auto results = lstm_onelayer_direc_packseq(data, batch_sizes, {h, c}, params, has_biases, num_layers, dropout_p,
                                               train, bidirectional);

    at::Tensor weight_2_layer;
    at::Tensor bias_2_layer;
    at::Tensor h_2_layer = hx[0].slice(0, 1, 2);
    at::Tensor c_2_layer = hx[1].slice(0, 1, 2);
    std::tie(weight_2_layer, bias_2_layer) = get_wb_double_layer_or_bidirec(input, params, has_biases);

    int64_t max_len = input.size(0);
    at::Tensor mask = get_mask(input, batch_sizes, h, max_len);
    at::Tensor input_2_layer = std::get<0>(results);

    auto results_2_layer = at_npu::native::custom_ops::npu_lstm_data(
        input_2_layer, batch_sizes, weight_2_layer, bias_2_layer, mask, h_2_layer, c_2_layer, has_biases, num_layers,
        dropout_p, train, bidirectional, batch_first, true, false);
    at::Tensor th_output_2_layer = at::unsqueeze(std::get<1>(results_2_layer)[num_step - 1], 0);
    at::Tensor tc_output_2_layer = at::unsqueeze(std::get<2>(results_2_layer)[num_step - 1], 0);
    at::Tensor th = at::cat({std::get<1>(results), th_output_2_layer}, 0);
    at::Tensor tc = at::cat({std::get<2>(results), tc_output_2_layer}, 0);

    return std::tie(std::get<0>(results_2_layer), th, tc);
}

tensor_list3 lstm_double_layer_bidirec_packseq(const at::Tensor &data, const at::Tensor &batch_sizes, at::TensorList hx,
                                               at::TensorList params, bool has_biases, int64_t num_layers,
                                               double dropout_p, bool train, bool bidirectional)
{
    int64_t t_size = batch_sizes.numel();
    TORCH_CHECK(t_size > 0, "batch_sizes can not be empty."
        + OPS_ERROR(ErrCode::VALUE));

    at::Tensor input = data.reshape({t_size, data.size(0) / t_size, data.size(1)});
    bool batch_first = false;

    at::Tensor hL0 = hx[0].slice(0, 0, 2);
    at::Tensor cL0 = hx[1].slice(0, 0, 2);

    at::Tensor hL1 = hx[0].slice(0, 2, 4);
    at::Tensor cL1 = hx[1].slice(0, 2, 4);

    auto results_layer1 = lstm_onelayer_bidirec_packseq(data, batch_sizes, {hL0, cL0}, params, has_biases, num_layers,
                                                        dropout_p, train, bidirectional);

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

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm(const at::Tensor &input, at::TensorList hx, at::TensorList params,
                                                    bool has_biases, int64_t num_layers, double dropout, bool train,
                                                    bool bidirectional, bool batch_first)
{
    // The operator of DynamicRnn only supports the T axis as the first axis.
    auto input_trans = batch_first ? input.transpose(0, 1) : input;
    at::Tensor y;
    at::Tensor h;
    at::Tensor c;

    if (!bidirectional) {
        std::tie(y, h, c) = lstm_multi_layer_direc_npu(input_trans, hx, params, has_biases, num_layers, num_layers,
                                                       dropout, train, bidirectional, batch_first);
    } else {
        std::tie(y, h, c) = lstm_multi_layer_bidirec_npu(input_trans, hx, params, has_biases, num_layers, num_layers,
                                                         dropout, train, bidirectional, batch_first);
    }

    // the Bacth axis of output should be first axis when batch_first is True!
    auto output = batch_first ? y.transpose(0, 1) : y;
    return std::tie(output, h, c);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm(const at::Tensor &data, const at::Tensor &batch_sizes,
                                                    at::TensorList hx, at::TensorList params, bool has_biases,
                                                    int64_t num_layers, double dropout, bool train,
                                                    bool bidirectional)
{
    at::Tensor batch_sizes_cpu = batch_sizes.to("cpu");
    at::Tensor y;
    at::Tensor h;
    at::Tensor c;

    if (num_layers == 1) {
        if (!bidirectional) {
            std::tie(y, h, c) = lstm_onelayer_direc_packseq(data, batch_sizes_cpu, hx, params, has_biases, num_layers,
                                                            dropout, train, bidirectional);
        } else {
            std::tie(y, h, c) = lstm_onelayer_bidirec_packseq(data, batch_sizes_cpu, hx, params, has_biases, num_layers,
                                                              dropout, train, bidirectional);
        }
    }

    if (num_layers == 2) {
        if (!bidirectional) {
            std::tie(y, h, c) = lstm_double_layer_direc_packseq(data, batch_sizes_cpu, hx, params, has_biases,
                                                                num_layers, dropout, train, bidirectional);
        } else {
            std::tie(y, h, c) = lstm_double_layer_bidirec_packseq(data, batch_sizes_cpu, hx, params, has_biases,
                                                                  num_layers, dropout, train, bidirectional);
        }
    }
    return std::tie(y, h, c);
}

tensor_list5 npu_lstm_backward(const c10::optional<at::Tensor> &grady, const c10::optional<at::Tensor> &gradh,
                               const c10::optional<at::Tensor> &gradc, const at::Tensor &input,
                               const at::Tensor &weight, const at::Tensor &bias, const at::Tensor &hx,
                               const at::Tensor &cx, const at::Tensor &y_output, const at::Tensor &h_output,
                               const at::Tensor &c_output, const at::Tensor &i, const at::Tensor &j,
                               const at::Tensor &f, const at::Tensor &o, const at::Tensor &tanhc)
{
    const at::Tensor &grady_opt = c10::value_or_else(grady, [] { return at::Tensor(); });
    const at::Tensor &gradh_opt = c10::value_or_else(gradh, [] { return at::Tensor(); });
    const at::Tensor &gradc_opt = c10::value_or_else(gradc, [] { return at::Tensor(); });

    at::Tensor inh = at::squeeze(hx, 0);
    at::Tensor inc = at::squeeze(cx, 0);

    at::Tensor grad_input = npu_preparation::apply_tensor(input);
    at::Tensor grad_weight = npu_preparation::apply_tensor(weight);
    at::Tensor grad_bias = npu_preparation::apply_tensor(bias);
    at::Tensor grad_ht = npu_preparation::apply_tensor(inh);
    at::Tensor grad_ct = npu_preparation::apply_tensor(inc);

    auto grad_y = grady_opt.defined() ? grady_opt : at::zeros(y_output.sizes(), y_output.options());
    auto grad_h = gradh_opt.defined() ? gradh_opt[input.size(0) - 1] : at::zeros(inh.sizes(), h_output.options());
    auto grad_c = gradc_opt.defined() ? gradc_opt[input.size(0) - 1] : at::zeros(inc.sizes(), c_output.options());

    lstm_backward_out_npu_nocheck(grad_weight, grad_bias, grad_input, grad_ht, grad_ct, input, weight, bias, inh, inc,
                                  grad_y, grad_h, grad_c, y_output, h_output, c_output, i, j, f, o, tanhc);
    grad_ht = at::unsqueeze(grad_ht, 0);
    grad_ct = at::unsqueeze(grad_ct, 0);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>{grad_input, grad_weight, grad_bias,
                                                                                  grad_ht, grad_ct};
}

tensor_list npu_lstm(const at::Tensor &input, const at::Tensor &weight, const at::Tensor &bias,
                     const at::Tensor &seq_mask, const at::Tensor &h, const at::Tensor &c, bool has_biases,
                     int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first,
                     bool flag_seq, bool direction)
{
    return npu_lstm_npu_nocheck(input, weight, bias, seq_mask, h, c, has_biases, num_layers, dropout, train,
                                bidirectional, batch_first, flag_seq, direction);
}

tensor_list npu_lstm_data(const at::Tensor &input, const at::Tensor &batch_sizes, const at::Tensor &weight,
                          const at::Tensor &bias, const at::Tensor &seq_mask, const at::Tensor &h, const at::Tensor &c,
                          bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional,
                          bool batch_first, bool flag_seq, bool direction)
{
    return npu_lstm_npu_nocheck(input, weight, bias, seq_mask, h, c, has_biases, num_layers, dropout, train,
                                bidirectional, batch_first, flag_seq, direction);
}

tensor_list5 npu_lstm_data_backward(const c10::optional<at::Tensor> &grady_opt,
                                    const c10::optional<at::Tensor> &gradh_opt,
                                    const c10::optional<at::Tensor> &gradc_opt, const at::Tensor &input,
                                    const at::Tensor &batch_sizes, const at::Tensor &weight, const at::Tensor &bias,
                                    const at::Tensor &init_h, const at::Tensor &init_c, const at::Tensor &y,
                                    const at::Tensor &h, const at::Tensor &c, const at::Tensor &i, const at::Tensor &j,
                                    const at::Tensor &f, const at::Tensor &o, const at::Tensor &tanhc,
                                    bool flag_direction)
{
    const at::Tensor &grady = c10::value_or_else(grady_opt, [] { return at::Tensor(); });
    const at::Tensor &gradh = c10::value_or_else(gradh_opt, [] { return at::Tensor(); });
    const at::Tensor &gradc = c10::value_or_else(gradc_opt, [] { return at::Tensor(); });

    at::Tensor inh = at::squeeze(init_h, 0);
    at::Tensor inc = at::squeeze(init_c, 0);

    at::Tensor grad_input = npu_preparation::apply_tensor(input);
    at::Tensor grad_weight = npu_preparation::apply_tensor(weight);
    at::Tensor grad_bias = npu_preparation::apply_tensor(bias);
    at::Tensor grad_ht = npu_preparation::apply_tensor(inh);
    at::Tensor grad_ct = npu_preparation::apply_tensor(inc);

    auto grad_y = grady.defined() ? grady : at::zeros(y.sizes(), y.options());
    auto grad_h = gradh.defined() ? gradh[input.size(0) - 1] : at::zeros(inh.sizes(), h.options());
    auto grad_c = gradc.defined() ? gradc[input.size(0) - 1] : at::zeros(inc.sizes(), c.options());

    lstm_backward_out_npu_nocheck(grad_weight, grad_bias, grad_input, grad_ht, grad_ct, input, weight, bias, inh, inc,
                                  grad_y, grad_h, grad_c, y, h, c, i, j, f, o, tanhc, flag_direction, batch_sizes);
    grad_ht = at::unsqueeze(grad_ht, 0);
    grad_ct = at::unsqueeze(grad_ct, 0);

    return std::tie(grad_input, grad_weight, grad_bias, grad_ht, grad_ct);
}
} // namespace acl_op

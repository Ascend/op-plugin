// Copyright (c) 2025 Huawei Technologies Co., Ltd
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

#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/custom_functions/opapi/update_op_api_common.h"

namespace op_api {
#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
const static int FLASH_THRESHOLD = 512;
const static int64_t PFA_SPARSE_HIGH_PRECISION_NO_MASK = 10;
const static int64_t PFA_SPARSE_HIGH_PRECISION_BAND = 14;
const static int64_t DIM_0 = 0;
const static int64_t DIM_1 = 1;
const static int64_t DIM_2 = 2;
const static int64_t DIM_3 = 3;
const static int64_t DIM_4 = 4;
const static int64_t DIM_NUMS_3 = 3;
const static int64_t DIM_NUMS_4 = 4;
const static int64_t PA_BBH_DIMS = 3;
const static int64_t PA_BNBD_DIMS = 4;
const static int64_t PA_NZ_DIMS = 5;
using namespace at_npu::native;
using npu_preparation = at_npu::native::OpPreparation;

static std::pair<std::string, std::string> get_query_and_attention_out_layout(
    const at::Tensor query,
    std::string input_layout_str)
{
    struct parserLayout {
        std::string qLayout;
        std::string outLayout;
        int32_t qDim;
    };

    const std::map<std::string, parserLayout> LAYOUT_MAP = {
        {"BSH",              {"BSH", "BSH", DIM_NUMS_3}},
        {"BSND",             {"BSND", "BSND", DIM_NUMS_4}},
        {"BNSD",             {"BNSD", "BNSD", DIM_NUMS_4}},
        {"TND",              {"TND", "TND", DIM_NUMS_3}},
        {"NTD",              {"NTD", "NTD", DIM_NUMS_3}},
        {"BNSD_BSND",        {"BNSD", "BSND", DIM_NUMS_4}},
        {"BSH_BNSD",         {"BSH", "BNSD", DIM_NUMS_3}},
        {"BSND_BNSD",        {"BSND", "BNSD", DIM_NUMS_4}},
        {"NTD_TND",          {"NTD", "TND", DIM_NUMS_3}},
        {"BSH_NBSD",         {"BSH", "NBSD", DIM_NUMS_3}},
        {"BSND_NBSD",        {"BSND", "NBSD", DIM_NUMS_4}},
        {"BNSD_NBSD",        {"BNSD", "NBSD", DIM_NUMS_4}},
        {"TND_NTD",          {"TND", "NTD", DIM_NUMS_3}},
        {"NSD",              {"NSD", "NSD", DIM_NUMS_3}}
    };

    std::string query_layout = "BSH";
    std::string attention_out_layout = "BSH";
    int32_t query_dim;
    auto it = LAYOUT_MAP.find(input_layout_str);
    if (it != LAYOUT_MAP.end()) {
        query_layout = it->second.qLayout;
        attention_out_layout = it->second.outLayout;
        query_dim = it->second.qDim;
        
        TORCH_CHECK(query.dim() == query_dim,
            "query's dim should be consistent with that of Layout", OPS_ERROR(ErrCode::VALUE));
    } else {
        TORCH_CHECK(
            false,
            "layout only support BSH, BSND, TND, NTD, BNSD_BSND, BSH_BNSD, BSND_BNSD, NTD_TND, ",
            "BSH_NBSD, BSND_NBSD, BNSD_NBSD, TND_NTD, but got ",
            query_layout,
            OPS_ERROR(ErrCode::VALUE)
        );
    }
    return {query_layout, attention_out_layout};
}

static std::tuple<int64_t, int64_t, int64_t, int64_t> get_query_b_n_s_d(
    const at::Tensor &query,
    std::string query_layout,
    int64_t num_heads)
{
    int64_t b = 0;
    int64_t n1 = 0;
    int64_t s1 = 0;
    int64_t d1 = 0;
    if (query_layout == "BSH") {
        b = query.size(DIM_0);
        s1 = query.size(DIM_1);
        n1 = num_heads;
        d1 = query.size(DIM_2) / num_heads;
    } else if (query_layout == "BSND") {
        b = query.size(DIM_0);
        s1 = query.size(DIM_1);
        n1 = query.size(DIM_2);
        d1 = query.size(DIM_3);
    } else if (query_layout == "BNSD") {
        b = query.size(DIM_0);
        s1 = query.size(DIM_2);
        n1 = query.size(DIM_1);
        d1 = query.size(DIM_3);
    } else if (query_layout == "NSD") {
        b = 1;
        s1 = query.size(DIM_1);
        n1 = query.size(DIM_0);
        d1 = query.size(DIM_2);
    } else {
        TORCH_CHECK(
            false,
            "It is not supported in get_query_b_n_s_d function, layout ",
            query_layout,
            OPS_ERROR(ErrCode::VALUE)
        );
    }
    return {b, n1, s1, d1};
}

static std::tuple<int64_t, int64_t, int64_t> get_query_t_n_d(
    const at::Tensor &query,
    std::string query_layout)
{
    int64_t t = 0;
    int64_t n1 = 0;
    int64_t d1 = 0;
    if (query_layout == "TND") {
        t = query.size(DIM_0);
        n1 = query.size(DIM_1);
        d1 = query.size(DIM_2);
    } else if (query_layout == "NTD") {
        t = query.size(DIM_1);
        n1 = query.size(DIM_0);
        d1 = query.size(DIM_2);
    } else {
        TORCH_CHECK(
            false,
            "It is not supported in get_query_t_n_d function, layout ",
            query_layout,
            OPS_ERROR(ErrCode::VALUE)
        );
    }
    return {t, n1, d1};
}

static int64_t get_value_d(
    const c10::optional<at::Tensor> &block_table,
    const at::Tensor &query,
    const at::Tensor &value,
    std::string query_layout,
    int64_t kv_num_heads)
{
    int64_t valueD = 0;
    if (block_table.has_value()) { // PA场景
        if (value.dim() == PA_BBH_DIMS) {
            valueD = value.size(DIM_2) / kv_num_heads;
        } else if (value.dim() == PA_BNBD_DIMS) {
            valueD = value.size(DIM_3);
        } else if (value.dim() == PA_NZ_DIMS) {
            valueD = value.size(DIM_2) * value.size(DIM_4);
    } else {
        TORCH_CHECK(
            false,
            "when Page Attention enabled, value's dim should be 3/4/5, but got ",
            value.dim(),
            OPS_ERROR(ErrCode::VALUE)
        );
    }
    } else { // 非PA场景
        TORCH_CHECK(
            value.dim() == query.dim(),
            "when Page Attention not enabled, value'dim should equal to query's dim!",
            OPS_ERROR(ErrCode::VALUE)
        );
        if (query_layout == "BSH") {
            valueD = value.size(DIM_2) / kv_num_heads;
        } else if (query_layout == "BSND" || query_layout == "BNSD") {
            valueD = value.size(DIM_3);
        } else if (query_layout == "TND" || query_layout == "NTD" || query_layout == "NSD") {
            valueD = value.size(DIM_2);
        }
    }
    return valueD;
}

static int get_change_d_scale_v2(
    const at::Tensor &value,
    c10::optional<int64_t> value_dtype)
{
    const static int changeDScale = 1;
    const static int changeDForInt32 = 8;
    const static int changeDForFP4 = 2;
    // int4 伪装成 int32
    if (value.scalar_type() == at::kInt) {
        return changeDForInt32;
    }
    // fp4 伪装成 uint8
    aclDataType value_acl_type = c10_npu::GetAclDataType(value_dtype.value_or(static_cast<int64_t>(value.scalar_type())));
    if (value_acl_type == aclDataType::ACL_FLOAT4_E1M2 || value_acl_type == aclDataType::ACL_FLOAT4_E2M1) {
        return changeDForFP4;
    }
    return changeDScale;
}

static at::Tensor infer_attention_out_shape(
    std::string attention_out_layout,
    const at::Tensor &query,
    std::string query_layout,
    int64_t num_heads,
    int64_t valueD)
{
    int64_t b = 0;
    int64_t n1 = 0;
    int64_t s1 = 0;
    int64_t d1 = 0;
    int64_t t = 0;
    at::Tensor attention_out = npu_preparation::apply_tensor_without_format(query);
    if (attention_out_layout == "BSH") {
        auto [b, n1, s1, d1] = get_query_b_n_s_d(query, query_layout, num_heads);
        int outH = num_heads * valueD;
        outH = (outH == 0 || query.size(DIM_2) == 0) ? query.size(DIM_2) : outH;
        attention_out = OpPreparation::apply_tensor_without_format(
            {b, s1, outH},
            query.options().dtype(query.dtype())
        );
    } else if (attention_out_layout == "BSND") {
        auto [b, n1, s1, d1] = get_query_b_n_s_d(query, query_layout, num_heads);
        int outD = valueD;
        outD = (outD == 0 || d1 == 0) ? d1 : outD;
        attention_out = OpPreparation::apply_tensor_without_format(
            {b, s1, n1, outD},
            query.options().dtype(query.dtype())
        );
    } else if (attention_out_layout == "BNSD") {
        auto [b, n1, s1, d1] = get_query_b_n_s_d(query, query_layout, num_heads);
        int outD = valueD;
        outD = (outD == 0 || d1 == 0) ? d1 : outD;
        attention_out = OpPreparation::apply_tensor_without_format(
            {b, n1, s1, outD},
            query.options().dtype(query.dtype())
        );
    } else if (attention_out_layout == "NBSD") {
        auto [b, n1, s1, d1] = get_query_b_n_s_d(query, query_layout, num_heads);
        int outD = valueD;
        outD = (outD == 0 || d1 == 0) ? d1 : outD;
        attention_out = OpPreparation::apply_tensor_without_format(
            {n1, b, s1, outD},
            query.options().dtype(query.dtype())
        );
    } else if (attention_out_layout == "TND") {
        auto [t, n1, d1] = get_query_t_n_d(query, query_layout);
        int outD = valueD;
        outD = (outD == 0 || d1 == 0) ? d1 : outD;
        attention_out = OpPreparation::apply_tensor_without_format(
            {t, n1, outD},
            query.options().dtype(query.dtype())
        );
    } else if (attention_out_layout == "NTD") {
        auto [t, n1, d1] = get_query_t_n_d(query, query_layout);
        int outD = valueD;
        outD = (outD == 0 || d1 == 0) ? d1 : outD;
        attention_out = OpPreparation::apply_tensor_without_format(
            {n1, t, outD},
            query.options().dtype(query.dtype())
        );
    } else if (attention_out_layout == "NSD") {
        auto [b, n1, s1, d1] = get_query_b_n_s_d(query, query_layout, num_heads);
        int outD = valueD;
        outD = (outD == 0 || d1 == 0) ? d1 : outD;
        attention_out = OpPreparation::apply_tensor_without_format(
            {n1, s1, outD},
            query.options().dtype(query.dtype())
        );
    }
    return attention_out;
}

static at::Tensor infer_lse_out_shape(
    std::string input_layout_str,
    const at::Tensor &query,
    std::string query_layout,
    int64_t num_heads)
{
    int64_t b = 0;
    int64_t n1 = 0;
    int64_t s1 = 0;
    int64_t d1 = 0;
    int64_t t = 0;
    at::Tensor lse_out;
    if (input_layout_str == "TND" || input_layout_str == "NTD" ||
        input_layout_str == "TND_NTD" || input_layout_str == "NTD_TND") {
        auto [t, n1, d1] = get_query_t_n_d(query, query_layout);
        lse_out = npu_preparation::apply_tensor_without_format({t, n1, 1}, c10::dtype(c10::ScalarType::Float));
    } else {
        auto [b, n1, s1, d1] = get_query_b_n_s_d(query, query_layout, num_heads);
        lse_out = npu_preparation::apply_tensor_without_format({b, n1, s1, 1}, c10::dtype(c10::ScalarType::Float));
    }
    return lse_out;
}

std::tuple<at::Tensor, at::Tensor> construct_fia_output_tensor_v2(
    const at::Tensor &query,
    const at::Tensor &value,
    c10::optional<int64_t> query_dtype,
    c10::optional<int64_t> value_dtype,
    std::string input_layout_str,
    const c10::optional<at::Tensor> &quant_scale_out,
    const c10::optional<at::Tensor> &block_table,
    int64_t num_query_heads,
    int64_t num_key_value_heads, // 增加kvhead用于计算BBH情况下的D
    bool return_softmax_lse,
    const c10::optional<at::Tensor> &query_rope,
    c10::optional<int64_t> out_dtype)
{
    TORCH_CHECK(
        num_query_heads > 0,
        "num_heads should be greater than 0, but the actual value is",
        num_query_heads,
        OPS_ERROR(ErrCode::VALUE)
    );
    num_key_value_heads = (num_key_value_heads == 0) ? num_query_heads : num_key_value_heads;

    // 获取query_layout, attention_out_layout
    auto [query_layout, attention_out_layout] = get_query_and_attention_out_layout(query, input_layout_str);

    // 计算valueD
    int64_t valueD = get_value_d(block_table, query, value, query_layout, num_key_value_heads);

    // 计算changeDScale
    int changeDScale = get_change_d_scale_v2(value, value_dtype);
    valueD = valueD * changeDScale;

    // 推导attenout shape
    at::Tensor tmp_output = infer_attention_out_shape(attention_out_layout, query, query_layout, num_query_heads, valueD);

    // hifp8输入
    bool is_hifloat8_input = query.dtype() == at::kByte && query_dtype.has_value() && c10_npu::GetAclDataType(query_dtype.value()) == aclDataType::ACL_HIFLOAT8;

    at::Tensor output;
    if (quant_scale_out.has_value()) {
        at::ScalarType output_type = at::ScalarType::Char;
        if (out_dtype.has_value()) {
            output_type = c10_npu::GetATenDType(out_dtype.value());
        }
        output = npu_preparation::apply_tensor_without_format(tmp_output.sizes(), output_type);
    } else if (query.dtype() == at::kChar || query.dtype() == at::ScalarType::Float8_e4m3fn || is_hifloat8_input) {
        if (out_dtype.has_value()) {
            at::ScalarType output_type = c10_npu::GetATenDType(out_dtype.value());
            output = npu_preparation::apply_tensor_without_format(tmp_output.sizes(), output_type);
        } else if (query_rope.has_value()) {
            const at::Tensor &query_rope_tensor = c10::value_or_else(query_rope, [] { return at::Tensor(); });
            output = npu_preparation::apply_tensor_without_format(tmp_output.sizes(), c10::dtype(query_rope_tensor.dtype()));
        } else {
            output = npu_preparation::apply_tensor_without_format(tmp_output.sizes(), c10::dtype(c10::ScalarType::Half));
        }
    } else {
        output = npu_preparation::apply_tensor_without_format(tmp_output);
    }

    // 推导lseout shape
    at::Tensor softmax_lse;
    if (return_softmax_lse) {
        softmax_lse = infer_lse_out_shape(input_layout_str, query, query_layout, num_query_heads);
    } else {
        softmax_lse = npu_preparation::apply_tensor_without_format({0}, c10::dtype(c10::ScalarType::Float));
    }

    return std::tuple<at::Tensor, at::Tensor>(output, softmax_lse);
}

std::tuple<at::Tensor, at::Tensor> npu_fused_infer_attention_score_v2_symint(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope,
    const c10::optional<at::Tensor> &pse_shift,
    const c10::optional<at::Tensor> &atten_mask,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_qlen,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_kvlen,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &dequant_scale_query,
    const c10::optional<at::Tensor> &dequant_scale_key,
    const c10::optional<at::Tensor> &dequant_offset_key,
    const c10::optional<at::Tensor> &dequant_scale_value,
    const c10::optional<at::Tensor> &dequant_offset_value,
    const c10::optional<at::Tensor> &dequant_scale_key_rope,
    const c10::optional<at::Tensor> &quant_scale_out,
    const c10::optional<at::Tensor> &quant_offset_out,
    const c10::optional<at::Tensor> &learnable_sink,
    int64_t num_query_heads, int64_t num_key_value_heads, double softmax_scale,
    int64_t pre_tokens, int64_t next_tokens, c10::string_view input_layout,
    int64_t sparse_mode, int64_t block_size,
    int64_t query_quant_mode, int64_t key_quant_mode, int64_t value_quant_mode,
    int64_t inner_precise, bool return_softmax_lse,
    c10::optional<int64_t> query_dtype, c10::optional<int64_t> key_dtype, c10::optional<int64_t> value_dtype,
    c10::optional<int64_t> query_rope_dtype, c10::optional<int64_t> key_rope_dtype,
    c10::optional<int64_t> key_shared_prefix_dtype, c10::optional<int64_t> value_shared_prefix_dtype,
    c10::optional<int64_t> dequant_scale_query_dtype, c10::optional<int64_t> dequant_scale_key_dtype,
    c10::optional<int64_t> dequant_scale_value_dtype, c10::optional<int64_t> dequant_scale_key_rope_dtype,
    c10::optional<int64_t> out_dtype)
{
    // convert str
    std::string input_layout_str = std::string(input_layout);

    // construct the output tensor
    std::tuple<at::Tensor, at::Tensor> fia_output = op_api::construct_fia_output_tensor_v2(query, value, query_dtype, value_dtype, input_layout_str, quant_scale_out, block_table,
                                                                                           num_query_heads, num_key_value_heads, return_softmax_lse, query_rope, out_dtype);
    at::Tensor output = std::get<0>(fia_output);
    at::Tensor softmax_lse = std::get<1>(fia_output);

    char *input_layout_ptr = const_cast<char *>(input_layout_str.c_str());

    at::Tensor default_actual_shared_prefix_len {nullptr};
    at::Tensor default_q_start_idx {nullptr};
    at::Tensor default_kv_start_idx {nullptr};
    at::Tensor dequant_scale1;
    at::Tensor quant_scale1;
    at::Tensor dequant_scale2;
    at::Tensor antiquant_scale;
    at::Tensor antiquant_offset;
    at::Tensor query_padding_size;
    at::Tensor kv_padding_size;
    at::Tensor key_shared_prefix;
    at::Tensor value_shared_prefix;
    int64_t antiquant_mode = 0;
    int64_t default_pse_type_value = 0;

    at::TensorList valueTensors = value;
    at::TensorList keyTensors = key;

    TensorWrapper query_wrapper = make_wrapper(query, query_dtype);
    TensorListWrapper keyTensors_wrapper = make_wrapper(keyTensors, key_dtype);
    TensorListWrapper valueTensors_wrapper = make_wrapper(valueTensors, value_dtype);
    TensorWrapper outTensor_wrapper = make_wrapper(output, out_dtype);

    at::Tensor null_tensor;
    auto query_rope_tmp = query_rope.has_value() ? query_rope.value() : null_tensor;
    TensorWrapper query_rope_wrapper = make_wrapper(query_rope_tmp, query_rope_dtype);
    auto key_rope_tmp = key_rope.has_value() ? key_rope.value() : null_tensor;
    TensorWrapper key_rope_wrapper = make_wrapper(key_rope_tmp, key_rope_dtype);
    auto dequant_scale_query_tmp = dequant_scale_query.has_value() ? dequant_scale_query.value() : null_tensor;
    TensorWrapper dequant_scale_query_wrapper = make_wrapper(dequant_scale_query_tmp, dequant_scale_query_dtype);
    auto dequant_scale_key_tmp = dequant_scale_key.has_value() ? dequant_scale_key.value() : null_tensor;
    TensorWrapper dequant_scale_key_wrapper = make_wrapper(dequant_scale_key_tmp, dequant_scale_key_dtype);
    auto dequant_scale_value_tmp = dequant_scale_value.has_value() ? dequant_scale_value.value() : null_tensor;
    TensorWrapper dequant_scale_value_wrapper = make_wrapper(dequant_scale_value_tmp, dequant_scale_value_dtype);

    if (c10_npu::GetSocVersion() != c10_npu::SocVersion::Ascend950) {
        EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnFusedInferAttentionScoreV4, query_wrapper, keyTensors_wrapper, valueTensors_wrapper, pse_shift, atten_mask, actual_seq_qlen, actual_seq_kvlen, dequant_scale1, quant_scale1, dequant_scale2,
            quant_scale_out, quant_offset_out, antiquant_scale, antiquant_offset, block_table, query_padding_size, kv_padding_size, dequant_scale_key_wrapper, dequant_offset_key, dequant_scale_value_wrapper,
            dequant_offset_value, key_shared_prefix, value_shared_prefix, default_actual_shared_prefix_len, query_rope_wrapper, key_rope_wrapper, dequant_scale_key_rope, dequant_scale_query_wrapper, learnable_sink, num_query_heads, softmax_scale, pre_tokens, next_tokens, input_layout_ptr,
            num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, return_softmax_lse, key_quant_mode, value_quant_mode, query_quant_mode, outTensor_wrapper, softmax_lse);
    } else {
        // Interface aclnnFusedInferAttentionScore versions V1 to V4 are no longer supported on Ascend950
        EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnFusedInferAttentionScoreV5, query_wrapper, keyTensors_wrapper, valueTensors_wrapper, pse_shift, atten_mask, actual_seq_qlen, actual_seq_kvlen, dequant_scale1, quant_scale1, dequant_scale2,
            quant_scale_out, quant_offset_out, antiquant_scale, antiquant_offset, block_table, query_padding_size, kv_padding_size, dequant_scale_key_wrapper, dequant_offset_key, dequant_scale_value_wrapper,
            dequant_offset_value, key_shared_prefix, value_shared_prefix, default_actual_shared_prefix_len, query_rope_wrapper, key_rope_wrapper, dequant_scale_key_rope, dequant_scale_query_wrapper, learnable_sink, default_q_start_idx, default_kv_start_idx, num_query_heads, softmax_scale, pre_tokens, next_tokens, input_layout_ptr,
            num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, return_softmax_lse, key_quant_mode, value_quant_mode, query_quant_mode, default_pse_type_value, outTensor_wrapper, softmax_lse);
    }

    return std::tuple<at::Tensor, at::Tensor>(output, softmax_lse);
}

std::tuple<at::Tensor &, at::Tensor &> npu_fused_infer_attention_score_v2_out_symint(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope,
    const c10::optional<at::Tensor> &pse_shift,
    const c10::optional<at::Tensor> &atten_mask,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_qlen,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_kvlen,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &dequant_scale_query,
    const c10::optional<at::Tensor> &dequant_scale_key,
    const c10::optional<at::Tensor> &dequant_offset_key,
    const c10::optional<at::Tensor> &dequant_scale_value,
    const c10::optional<at::Tensor> &dequant_offset_value,
    const c10::optional<at::Tensor> &dequant_scale_key_rope,
    const c10::optional<at::Tensor> &quant_scale_out,
    const c10::optional<at::Tensor> &quant_offset_out,
    const c10::optional<at::Tensor> &learnable_sink,
    int64_t num_query_heads, int64_t num_key_value_heads, double softmax_scale,
    int64_t pre_tokens, int64_t next_tokens, c10::string_view input_layout,
    int64_t sparse_mode, int64_t block_size,
    int64_t query_quant_mode, int64_t key_quant_mode, int64_t value_quant_mode,
    int64_t inner_precise, bool return_softmax_lse,
    c10::optional<int64_t> query_dtype, c10::optional<int64_t> key_dtype, c10::optional<int64_t> value_dtype,
    c10::optional<int64_t> query_rope_dtype, c10::optional<int64_t> key_rope_dtype,
    c10::optional<int64_t> key_shared_prefix_dtype, c10::optional<int64_t> value_shared_prefix_dtype,
    c10::optional<int64_t> dequant_scale_query_dtype, c10::optional<int64_t> dequant_scale_key_dtype,
    c10::optional<int64_t> dequant_scale_value_dtype, c10::optional<int64_t> dequant_scale_key_rope_dtype,
    c10::optional<int64_t> out_dtype,
    const c10::optional<at::Tensor> &workspace,
    at::Tensor &attention_out,
    at::Tensor &softmax_lse)
{
    // convert str
    std::string input_layout_str = std::string(input_layout);
    char *input_layout_ptr = const_cast<char *>(input_layout_str.c_str());

    at::Tensor default_actual_shared_prefix_len {nullptr};
    at::Tensor default_q_start_idx {nullptr};
    at::Tensor default_kv_start_idx {nullptr};
    at::Tensor dequant_scale1;
    at::Tensor quant_scale1;
    at::Tensor dequant_scale2;
    at::Tensor antiquant_scale;
    at::Tensor antiquant_offset;
    at::Tensor query_padding_size;
    at::Tensor kv_padding_size;
    at::Tensor key_shared_prefix;
    at::Tensor value_shared_prefix;
    int64_t antiquant_mode = 0;
    int64_t default_pse_type_value = 0;

    at::TensorList valueTensors = value;
    at::TensorList keyTensors = key;

    TensorWrapper query_wrapper = make_wrapper(query, query_dtype);
    TensorListWrapper keyTensors_wrapper = make_wrapper(keyTensors, key_dtype);
    TensorListWrapper valueTensors_wrapper = make_wrapper(valueTensors, value_dtype);
    TensorWrapper outTensor_wrapper = make_wrapper(attention_out, out_dtype);

    at::Tensor null_tensor;
    auto query_rope_tmp = query_rope.has_value() ? query_rope.value() : null_tensor;
    TensorWrapper query_rope_wrapper = make_wrapper(query_rope_tmp, query_rope_dtype);
    auto key_rope_tmp = key_rope.has_value() ? key_rope.value() : null_tensor;
    TensorWrapper key_rope_wrapper = make_wrapper(key_rope_tmp, key_rope_dtype);
    auto dequant_scale_query_tmp = dequant_scale_query.has_value() ? dequant_scale_query.value() : null_tensor;
    TensorWrapper dequant_scale_query_wrapper = make_wrapper(dequant_scale_query_tmp, dequant_scale_query_dtype);
    auto dequant_scale_key_tmp = dequant_scale_key.has_value() ? dequant_scale_key.value() : null_tensor;
    TensorWrapper dequant_scale_key_wrapper = make_wrapper(dequant_scale_key_tmp, dequant_scale_key_dtype);
    auto dequant_scale_value_tmp = dequant_scale_value.has_value() ? dequant_scale_value.value() : null_tensor;
    TensorWrapper dequant_scale_value_wrapper = make_wrapper(dequant_scale_value_tmp, dequant_scale_value_dtype);

    if (c10_npu::GetSocVersion() != c10_npu::SocVersion::Ascend950) {
        if (workspace.has_value()) {
            void* workspace_addr = const_cast<void *>(workspace.value().storage().data());
            uint64_t workspace_size = static_cast<uint64_t>(workspace.value().numel() * workspace.value().element_size());
            EXEC_UPDATE_NPU_NO_FORMAT_CHECK_CMD(aclnnFusedInferAttentionScoreV4, workspace_addr, workspace_size, query_wrapper, keyTensors_wrapper, valueTensors_wrapper, pse_shift, atten_mask, actual_seq_qlen, actual_seq_kvlen, dequant_scale1, quant_scale1, dequant_scale2,
                quant_scale_out, quant_offset_out, antiquant_scale, antiquant_offset, block_table, query_padding_size, kv_padding_size, dequant_scale_key_wrapper, dequant_offset_key, dequant_scale_value_wrapper,
                dequant_offset_value, key_shared_prefix, value_shared_prefix, default_actual_shared_prefix_len, query_rope_wrapper, key_rope_wrapper, dequant_scale_key_rope, dequant_scale_query_wrapper, learnable_sink, num_query_heads, softmax_scale, pre_tokens, next_tokens, input_layout_ptr,
                num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, return_softmax_lse, key_quant_mode, value_quant_mode, query_quant_mode, outTensor_wrapper, softmax_lse);
        } else {
            EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnFusedInferAttentionScoreV4, query_wrapper, keyTensors_wrapper, valueTensors_wrapper, pse_shift, atten_mask, actual_seq_qlen, actual_seq_kvlen, dequant_scale1, quant_scale1, dequant_scale2,
                quant_scale_out, quant_offset_out, antiquant_scale, antiquant_offset, block_table, query_padding_size, kv_padding_size, dequant_scale_key_wrapper, dequant_offset_key, dequant_scale_value_wrapper,
                dequant_offset_value, key_shared_prefix, value_shared_prefix, default_actual_shared_prefix_len, query_rope_wrapper, key_rope_wrapper, dequant_scale_key_rope, dequant_scale_query_wrapper, learnable_sink, num_query_heads, softmax_scale, pre_tokens, next_tokens, input_layout_ptr,
                num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, return_softmax_lse, key_quant_mode, value_quant_mode, query_quant_mode, outTensor_wrapper, softmax_lse);
        }
    } else {
        // Interface aclnnFusedInferAttentionScore versions V1 to V4 are no longer supported on Ascend950
        if (workspace.has_value()) {
            void* workspace_addr = const_cast<void *>(workspace.value().storage().data());
            uint64_t workspace_size = static_cast<uint64_t>(workspace.value().numel() * workspace.value().element_size());
            EXEC_UPDATE_NPU_NO_FORMAT_CHECK_CMD(aclnnFusedInferAttentionScoreV5, workspace_addr, workspace_size, query_wrapper, keyTensors_wrapper, valueTensors_wrapper, pse_shift, atten_mask, actual_seq_qlen, actual_seq_kvlen, dequant_scale1, quant_scale1, dequant_scale2,
                quant_scale_out, quant_offset_out, antiquant_scale, antiquant_offset, block_table, query_padding_size, kv_padding_size, dequant_scale_key_wrapper, dequant_offset_key, dequant_scale_value_wrapper,
                dequant_offset_value, key_shared_prefix, value_shared_prefix, default_actual_shared_prefix_len, query_rope_wrapper, key_rope_wrapper, dequant_scale_key_rope, dequant_scale_query_wrapper, learnable_sink, default_q_start_idx, default_kv_start_idx, num_query_heads, softmax_scale, pre_tokens, next_tokens, input_layout_ptr,
                num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, return_softmax_lse, key_quant_mode, value_quant_mode, query_quant_mode, default_pse_type_value, outTensor_wrapper, softmax_lse);
        } else {
            EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnFusedInferAttentionScoreV5, query_wrapper, keyTensors_wrapper, valueTensors_wrapper, pse_shift, atten_mask, actual_seq_qlen, actual_seq_kvlen, dequant_scale1, quant_scale1, dequant_scale2,
                quant_scale_out, quant_offset_out, antiquant_scale, antiquant_offset, block_table, query_padding_size, kv_padding_size, dequant_scale_key_wrapper, dequant_offset_key, dequant_scale_value_wrapper,
                dequant_offset_value, key_shared_prefix, value_shared_prefix, default_actual_shared_prefix_len, query_rope_wrapper, key_rope_wrapper, dequant_scale_key_rope, dequant_scale_query_wrapper, learnable_sink, default_q_start_idx, default_kv_start_idx, num_query_heads, softmax_scale, pre_tokens, next_tokens, input_layout_ptr,
                num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, return_softmax_lse, key_quant_mode, value_quant_mode, query_quant_mode, default_pse_type_value, outTensor_wrapper, softmax_lse);
        }
    }

    return std::tuple<at::Tensor&, at::Tensor&>(attention_out, softmax_lse);
}

at::Tensor _npu_fused_infer_attention_score_v2_get_max_workspace_symint(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope,
    const c10::optional<at::Tensor> &pse_shift,
    const c10::optional<at::Tensor> &atten_mask,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_qlen,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_kvlen,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &dequant_scale_query,
    const c10::optional<at::Tensor> &dequant_scale_key,
    const c10::optional<at::Tensor> &dequant_offset_key,
    const c10::optional<at::Tensor> &dequant_scale_value,
    const c10::optional<at::Tensor> &dequant_offset_value,
    const c10::optional<at::Tensor> &dequant_scale_key_rope,
    const c10::optional<at::Tensor> &quant_scale_out,
    const c10::optional<at::Tensor> &quant_offset_out,
    const c10::optional<at::Tensor> &learnable_sink,
    int64_t num_query_heads, int64_t num_key_value_heads, double softmax_scale,
    int64_t pre_tokens, int64_t next_tokens, c10::string_view input_layout,
    int64_t sparse_mode, int64_t block_size,
    int64_t query_quant_mode, int64_t key_quant_mode, int64_t value_quant_mode,
    int64_t inner_precise, bool return_softmax_lse,
    c10::optional<int64_t> query_dtype, c10::optional<int64_t> key_dtype, c10::optional<int64_t> value_dtype,
    c10::optional<int64_t> query_rope_dtype, c10::optional<int64_t> key_rope_dtype,
    c10::optional<int64_t> key_shared_prefix_dtype, c10::optional<int64_t> value_shared_prefix_dtype,
    c10::optional<int64_t> dequant_scale_query_dtype, c10::optional<int64_t> dequant_scale_key_dtype,
    c10::optional<int64_t> dequant_scale_value_dtype, c10::optional<int64_t> dequant_scale_key_rope_dtype,
    c10::optional<int64_t> out_dtype)
{
    std::string input_layout_str = std::string(input_layout);

    // construct the output tensor
    std::tuple<at::Tensor, at::Tensor> fia_output = op_api::construct_fia_output_tensor_v2(query, value, query_dtype, value_dtype, input_layout_str, quant_scale_out, block_table,
                                                                                           num_query_heads, num_key_value_heads, return_softmax_lse, query_rope, out_dtype);
    at::Tensor output = std::get<0>(fia_output);
    at::Tensor softmax_lse = std::get<1>(fia_output);

    // convert str
    char *input_layout_ptr = const_cast<char *>(input_layout_str.c_str());

    at::Tensor default_actual_shared_prefix_len {nullptr};
    at::Tensor default_q_start_idx {nullptr};
    at::Tensor default_kv_start_idx {nullptr};
    at::Tensor dequant_scale1;
    at::Tensor quant_scale1;
    at::Tensor dequant_scale2;
    at::Tensor antiquant_scale;
    at::Tensor antiquant_offset;
    at::Tensor query_padding_size;
    at::Tensor kv_padding_size;
    at::Tensor key_shared_prefix;
    at::Tensor value_shared_prefix;
    int64_t antiquant_mode = 0;
    int64_t default_pse_type_value = 0;

    at::TensorList valueTensors = value;
    at::TensorList keyTensors = key;

    TensorWrapper query_wrapper = make_wrapper(query, query_dtype);
    TensorListWrapper keyTensors_wrapper = make_wrapper(keyTensors, key_dtype);
    TensorListWrapper valueTensors_wrapper = make_wrapper(valueTensors, value_dtype);
    TensorWrapper outTensor_wrapper = make_wrapper(output, out_dtype);

    at::Tensor null_tensor;
    auto query_rope_tmp = query_rope.has_value() ? query_rope.value() : null_tensor;
    TensorWrapper query_rope_wrapper = make_wrapper(query_rope_tmp, query_rope_dtype);
    auto key_rope_tmp = key_rope.has_value() ? key_rope.value() : null_tensor;
    TensorWrapper key_rope_wrapper = make_wrapper(key_rope_tmp, key_rope_dtype);
    auto dequant_scale_query_tmp = dequant_scale_query.has_value() ? dequant_scale_query.value() : null_tensor;
    TensorWrapper dequant_scale_query_wrapper = make_wrapper(dequant_scale_query_tmp, dequant_scale_query_dtype);
    auto dequant_scale_key_tmp = dequant_scale_key.has_value() ? dequant_scale_key.value() : null_tensor;
    TensorWrapper dequant_scale_key_wrapper = make_wrapper(dequant_scale_key_tmp, dequant_scale_key_dtype);
    auto dequant_scale_value_tmp = dequant_scale_value.has_value() ? dequant_scale_value.value() : null_tensor;
    TensorWrapper dequant_scale_value_wrapper = make_wrapper(dequant_scale_value_tmp, dequant_scale_value_dtype);
    uint64_t workspace_size = 0;

    if (c10_npu::GetSocVersion() != c10_npu::SocVersion::Ascend950) {
        workspace_size = EXEC_GET_MAX_WORKSPACE_CMD(aclnnFusedInferAttentionScoreV4, query_wrapper, keyTensors_wrapper, valueTensors_wrapper, pse_shift, atten_mask, actual_seq_qlen, actual_seq_kvlen, dequant_scale1, quant_scale1, dequant_scale2,
            quant_scale_out, quant_offset_out, antiquant_scale, antiquant_offset, block_table, query_padding_size, kv_padding_size, dequant_scale_key_wrapper, dequant_offset_key, dequant_scale_value_wrapper,
            dequant_offset_value, key_shared_prefix, value_shared_prefix, default_actual_shared_prefix_len, query_rope_wrapper, key_rope_wrapper, dequant_scale_key_rope, dequant_scale_query_wrapper, learnable_sink, num_query_heads, softmax_scale, pre_tokens, next_tokens, input_layout_ptr,
            num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, return_softmax_lse, key_quant_mode, value_quant_mode, query_quant_mode, outTensor_wrapper, softmax_lse);
    } else {
        // Interface aclnnFusedInferAttentionScore versions V1 to V4 are no longer supported on Ascend950
        workspace_size = EXEC_GET_MAX_WORKSPACE_CMD(aclnnFusedInferAttentionScoreV5, query_wrapper, keyTensors_wrapper, valueTensors_wrapper, pse_shift, atten_mask, actual_seq_qlen, actual_seq_kvlen, dequant_scale1, quant_scale1, dequant_scale2,
            quant_scale_out, quant_offset_out, antiquant_scale, antiquant_offset, block_table, query_padding_size, kv_padding_size, dequant_scale_key_wrapper, dequant_offset_key, dequant_scale_value_wrapper,
            dequant_offset_value, key_shared_prefix, value_shared_prefix, default_actual_shared_prefix_len, query_rope_wrapper, key_rope_wrapper, dequant_scale_key_rope, dequant_scale_query_wrapper, learnable_sink, default_q_start_idx, default_kv_start_idx, num_query_heads, softmax_scale, pre_tokens, next_tokens, input_layout_ptr,
            num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, return_softmax_lse, key_quant_mode, value_quant_mode, query_quant_mode, default_pse_type_value, outTensor_wrapper, softmax_lse);
    }
    
    at::Tensor workspace_tensor = npu_preparation::apply_tensor_without_format({workspace_size}, query.options().dtype(query.dtype()));
    return workspace_tensor;
}

std::tuple<at::Tensor, at::Tensor> _npu_fused_infer_attention_score_v2_infer_output(
    const at::Tensor &query,
    const at::Tensor &value,
    c10::optional<int64_t> query_dtype,
    c10::optional<int64_t> value_dtype,
    c10::string_view input_layout,
    const c10::optional<at::Tensor> &quant_scale_out,
    const c10::optional<at::Tensor> &block_table,
    int64_t num_query_heads,
    int64_t num_key_value_heads,
    bool return_softmax_lse,
    const c10::optional<at::Tensor> &query_rope,
    c10::optional<int64_t> out_dtype)
{
    std::string input_layout_str = std::string(input_layout);
    // construct the output tensor
    std::tuple<at::Tensor, at::Tensor> fia_output = op_api::construct_fia_output_tensor_v2(query, value, query_dtype, value_dtype, input_layout_str,
                                                                                           quant_scale_out, block_table, num_query_heads, num_key_value_heads,
                                                                                           return_softmax_lse, query_rope, out_dtype);
    return fia_output;
}

#endif
} // namespace op_api

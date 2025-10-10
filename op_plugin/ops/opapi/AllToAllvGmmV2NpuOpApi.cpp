// Copyright (c) 2025 Huawei Technologies Co., Ltd


#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_alltoallv_gmm_v2(
    const at::Tensor &gmm_x, const at::Tensor &gmm_weight, c10::string_view hcom, int64_t ep_world_size,
    at::IntArrayRef send_counts, at::IntArrayRef recv_counts, const c10::optional<at::Tensor> &send_counts_tensor,
    const c10::optional<at::Tensor> &recv_counts_tensor, const c10::optional<at::Tensor> &mm_x,
    const c10::optional<at::Tensor> &mm_weight, bool trans_gmm_weight, bool trans_mm_weight, bool permute_out_flag,
    const c10::optional<at::Tensor> &gmm_x_scale, const c10::optional<at::Tensor> &gmm_weight_scale,
    c10::string_view comm_mode, int64_t y_dtype, int64_t max_output_token_num)
{
    at::Tensor mm_y{nullptr};
    at::Tensor permute_out{nullptr};
    at::Tensor global_token_per_expert{nullptr};
    int64_t a = 0;
    int64_t default_scale = 2;
    int64_t dtype_float16 = 0;
    int64_t dtype_bfloat16 = 1;
    max_output_token_num = max_output_token_num == -1 ? default_scale * gmm_x.size(0) : max_output_token_num;
    if (comm_mode == "aiv") {
        a = max_output_token_num;
    } else {
        for (auto &i : recv_counts) {
            a += i;
        }
    }
    TORCH_CHECK((y_dtype == 0 || y_dtype == 1), "Invalid y_dtype value:", y_dtype,
                ". Expected 0(fp16) or 1 (bfp16).");
    if (mm_x.has_value() && mm_weight.has_value()) {
        const at::Tensor &mm_x_value = mm_x.value();
        const at::Tensor &mm_weight_value = mm_weight.value();
        int64_t bs = mm_x_value.size(0);
        int64_t n2 = trans_mm_weight ? mm_weight_value.size(0) : mm_weight_value.size(1);
        auto mmy_options = mm_x_value.options();
        if (y_dtype == dtype_float16 && gmm_x.scalar_type() == at::kChar) {
            mmy_options = mmy_options.dtype(at::kHalf);
        } else if (y_dtype == dtype_bfloat16 && gmm_x.scalar_type() == at::kChar) {
            mmy_options = mmy_options.dtype(at::kBFloat16);
        }
        mm_y = at_npu::native::OpPreparation::apply_tensor_without_format({bs, n2}, mmy_options);
    }
    if (permute_out_flag) {
        int64_t h1 = gmm_x.size(1);
        auto gmmx_options = gmm_x.options();
        if (y_dtype == dtype_float16 && gmm_x.scalar_type() == at::kChar) {
            gmmx_options = gmmx_options.dtype(at::kHalf);
        } else if (y_dtype == dtype_bfloat16 && gmm_x.scalar_type() == at::kChar) {
            gmmx_options = gmmx_options.dtype(at::kBFloat16);
        }
        permute_out = at_npu::native::OpPreparation::apply_tensor_without_format({a, h1}, gmmx_options);
    }
    int64_t n1 = trans_gmm_weight ? gmm_weight.size(1) : gmm_weight.size(2);
    auto gmmy_options = gmm_x.options();
    if (y_dtype == dtype_float16 && gmm_x.scalar_type() == at::kChar) {
        gmmy_options = gmmy_options.dtype(at::kHalf);
    } else if (y_dtype == dtype_bfloat16 && gmm_x.scalar_type() == at::kChar) {
        gmmy_options = gmmy_options.dtype(at::kBFloat16);
    }
    auto gmm_y = at_npu::native::OpPreparation::apply_tensor_without_format({a, n1}, gmmy_options);
    const at::Tensor &mm_x_real = mm_x.value_or(at::Tensor());
    const at::Tensor &mm_weight_real = mm_weight.value_or(at::Tensor());
    const at::Tensor &send_count_tensor_real = send_counts_tensor.value_or(at::Tensor());
    const at::Tensor &recv_count_tensor_real = recv_counts_tensor.value_or(at::Tensor());
    auto gmm_weight_scale_real = gmm_weight_scale.value_or(at::Tensor());
    auto gmm_x_scale_real = gmm_x_scale.value_or(at::Tensor());
    static std::vector<int64_t> inplace_vec{0, 0};
    at::IntArrayRef send_counts_real = send_counts.empty() ? at::IntArrayRef(inplace_vec) : send_counts;
    at::IntArrayRef recv_counts_real = recv_counts.empty() ? at::IntArrayRef(inplace_vec) : recv_counts;
    auto global_token_per_expert_dtype = at::kInt;
    if (comm_mode == "aiv") {
        global_token_per_expert = at_npu::native::OpPreparation::apply_tensor_without_format(
            {ep_world_size, gmm_weight.size(0) * ep_world_size}, global_token_per_expert_dtype);
    }
    char *hcom_ptr = const_cast<char *>(hcom.data());
    char *comm_mode_ptr = const_cast<char *>(comm_mode.data());
    EXEC_NPU_CMD(aclnnAlltoAllvGroupedMatMulV2, gmm_x, gmm_weight, send_count_tensor_real, recv_count_tensor_real,
                 mm_x_real, mm_weight_real, gmm_x_scale_real, gmm_weight_scale_real, hcom_ptr, ep_world_size,
                 send_counts_real, recv_counts_real, trans_gmm_weight, trans_mm_weight, permute_out_flag, comm_mode_ptr,
                 gmm_y, mm_y, permute_out, global_token_per_expert);
    return std::tie(gmm_y, mm_y, permute_out, global_token_per_expert);
    }
} // namespace op_api
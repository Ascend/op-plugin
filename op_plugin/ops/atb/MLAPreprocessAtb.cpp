#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/custom_functions/atb/AtbCommon.h"


namespace atb {
namespace {
std::unordered_map<c10::string_view, uint16_t> cache_mode_map = {
    {"krope_ctkv", 1},
    {"int8_nzcache", 2},
    {"nzcache", 3}
};

std::unordered_map<c10::string_view, uint16_t> quant_mode_map = {
    {"per_tensor_quant_asymm", 0},
    {"per_token_quant_symm", 1},
};
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&> npu_mla_preprocess_out(const at::Tensor &input, const at::Tensor &gamma0, const at::Tensor &beta0,
    const at::Tensor &wdqkv, const at::Tensor &descale0, const at::Tensor &gamma1, const at::Tensor &beta1,
    const at::Tensor &wuq, const at::Tensor &descale1, const at::Tensor &gamma2, const at::Tensor &cos, const at::Tensor &sin, const at::Tensor &wuk,
    const at::Tensor &kv_cache, const at::Tensor &kv_cache_rope, const at::Tensor &slotmapping,
    const c10::optional<at::Tensor>  &quant_scale0, const c10::optional<at::Tensor> &quant_offset0, const c10::optional<at::Tensor> &bias0,
    const c10::optional<at::Tensor> &quant_scale1, const c10::optional<at::Tensor>  &quant_offset1, const c10::optional<at::Tensor>  &bias1,
    const c10::optional<at::Tensor> &ctkv_scale, const c10::optional<at::Tensor> &q_nope_scale,
    c10::optional<c10::string_view> cache_mode_opt, c10::optional<c10::string_view> quant_mode_opt,
    at::Tensor &q_out0,
    at::Tensor &kv_cache_out0,
    at::Tensor &q_out1,
    at::Tensor &kv_cache_out1)
{
    const c10::OptionalDeviceGuard device_guard(device_of(input));
    uint32_t wdq_dim = 0;
    uint32_t q_rope_dim = 0;
    uint32_t k_rope_dim = 0;
    float epsilon = 1e-5;
    uint32_t q_rotary_coeff = 2;
    uint32_t k_rotary_coeff = 2;
    bool transpose_wdq = true;
    bool transpose_wuq = true;
    bool transpose_wuk = true;
    auto cache_mode = atb::utils::get_op_mode(cache_mode_map, cache_mode_opt, "krope_ctkv", "cache_mode");
    auto quant_mode = atb::utils::get_op_mode(quant_mode_map, quant_mode_opt, "per_token_quant_symm", "quant_mode");
    EXEC_ATB_CMD(AtbMLAPreprocess, input, gamma0, beta0, quant_scale0, quant_offset0, wdqkv, descale0, bias0, gamma1, beta1, quant_scale1, quant_offset1, wuq, descale1, bias1, gamma2, cos, sin, wuk, kv_cache,
                 kv_cache_rope, slotmapping, ctkv_scale, q_nope_scale,
                 wdq_dim, q_rope_dim, k_rope_dim, epsilon, q_rotary_coeff, k_rotary_coeff, transpose_wdq, transpose_wuq, transpose_wuk, cache_mode, quant_mode,
                 q_out0, kv_cache_out0, q_out1, kv_cache_out1);
    return std::forward_as_tuple(q_out0, kv_cache_out0, q_out1, kv_cache_out1);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_mla_preprocess(const at::Tensor &input, const at::Tensor &gamma0, const at::Tensor &beta0,
    const at::Tensor &wdqkv, const at::Tensor &descale0, const at::Tensor &gamma1, const at::Tensor &beta1,
    const at::Tensor &wuq, const at::Tensor &descale1, const at::Tensor &gamma2, const at::Tensor &cos, const at::Tensor &sin, const at::Tensor &wuk,
    const at::Tensor &kv_cache, const at::Tensor &kv_cache_rope, const at::Tensor &slotmapping,
    const c10::optional<at::Tensor>  &quant_scale0, const c10::optional<at::Tensor> &quant_offset0, const c10::optional<at::Tensor> &bias0,
    const c10::optional<at::Tensor> &quant_scale1, const c10::optional<at::Tensor>  &quant_offset1, const c10::optional<at::Tensor>  &bias1,
    const c10::optional<at::Tensor> &ctkv_scale, const c10::optional<at::Tensor> &q_nope_scale,
    c10::optional<c10::string_view> cache_mode_opt, c10::optional<c10::string_view> quant_mode_opt)
{
    const c10::OptionalDeviceGuard device_guard(device_of(input));
    uint32_t wdq_dim = 0;
    uint32_t q_rope_dim = 0;
    uint32_t k_rope_dim = 0;
    float epsilon = 1e-5;
    uint32_t q_rotary_coeff = 2;
    uint32_t k_rotary_coeff = 2;
    bool transpose_wdq = true;
    bool transpose_wuq = true;
    bool transpose_wuk = true;
    int token_num = input.size(0);
    int head_num = wuk.size(0);
    at::Tensor q_out0 = at::empty({token_num, head_num, 512}, kv_cache.options());
    at::Tensor kv_cache_out0;
    at::Tensor q_out1 = at::empty({token_num, head_num, 64}, input.options());
    at::Tensor kv_cache_out1;
    auto cache_mode = atb::utils::get_op_mode(cache_mode_map, cache_mode_opt, "krope_ctkv", "cache_mode");
    auto quant_mode = atb::utils::get_op_mode(quant_mode_map, quant_mode_opt, "per_token_quant_symm", "quant_mode");
    if (cache_mode == 2 || cache_mode == 3) {
        kv_cache_out0 = at_npu::native::empty_with_format(kv_cache.sizes(), kv_cache.options(), ACL_FORMAT_FRACTAL_NZ);
        kv_cache_out1 = at_npu::native::empty_with_format(kv_cache_rope.sizes(), kv_cache_rope.options(), ACL_FORMAT_FRACTAL_NZ);
    } else {
        kv_cache_out0 = at::empty(kv_cache.sizes(), kv_cache.options());
        kv_cache_out1 = at::empty(kv_cache_rope.sizes(), kv_cache_rope.options());
    }
    EXEC_ATB_CMD(AtbMLAPreprocess, input, gamma0, beta0, quant_scale0, quant_offset0, wdqkv, descale0, bias0, gamma1, beta1, quant_scale1, quant_offset1, wuq, descale1, bias1, gamma2, cos, sin, wuk, kv_cache,
                 kv_cache_rope, slotmapping, ctkv_scale, q_nope_scale,
                 wdq_dim, q_rope_dim, k_rope_dim, epsilon, q_rotary_coeff, k_rotary_coeff, transpose_wdq, transpose_wuq, transpose_wuk, cache_mode, quant_mode,
                 q_out0, kv_cache_out0, q_out1, kv_cache_out1);
    return std::make_tuple(q_out0, kv_cache_out0, q_out1, kv_cache_out1);
}


namespace {
TORCH_LIBRARY_FRAGMENT(atb, m)
{
    m.def("npu_mla_preprocess(Tensor input, Tensor gamma0, Tensor beta0, Tensor wdqkv, Tensor descale0, Tensor gamma1, Tensor beta1, Tensor wuq, Tensor descale1, Tensor gamma2, Tensor cos, Tensor sin, Tensor wuk, Tensor kv_cache, Tensor kv_cache_rope, Tensor slotmapping, *,  Tensor? quant_scale0=None, Tensor? quant_offset0=None, Tensor? bias0=None, Tensor? quant_scale1=None, Tensor? quant_offset1=None, Tensor? bias1=None, Tensor? ctkv_scale=None, Tensor? q_nope_scale=None, str? cache_mode=None, str? quant_mode=None) -> (Tensor, Tensor, Tensor, Tensor)");
    m.def("npu_mla_preprocess.out(Tensor input, Tensor gamma0, Tensor beta0, Tensor wdqkv, Tensor descale0, Tensor gamma1, Tensor beta1, Tensor wuq, Tensor descale1, Tensor gamma2, Tensor cos, Tensor sin, Tensor wuk, Tensor kv_cache, Tensor kv_cache_rope, Tensor slotmapping, *,  Tensor? quant_scale0=None, Tensor? quant_offset0=None, Tensor? bias0=None, Tensor? quant_scale1=None, Tensor? quant_offset1=None, Tensor? bias1=None, Tensor? ctkv_scale=None, Tensor? q_nope_scale=None, str? cache_mode=None, str? quant_mode=None, Tensor(a!) q_out0, Tensor(b!) kv_cache_out0, Tensor(c!) q_out1, Tensor(d!) kv_cache_out1) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!))");
}
}

namespace {
TORCH_LIBRARY_IMPL(atb, PrivateUse1, m)
{
    m.impl("npu_mla_preprocess", TORCH_FN(atb::npu_mla_preprocess));
    m.impl("npu_mla_preprocess.out", TORCH_FN(atb::npu_mla_preprocess_out));
}
}
}

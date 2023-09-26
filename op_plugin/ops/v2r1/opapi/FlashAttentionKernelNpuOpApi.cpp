#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/OpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_prompt_flash_attention(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &padding_mask,
    const c10::optional<at::Tensor> &atten_mask,
    c10::OptionalIntArrayRef actual_seq_lengths,
    int64_t num_heads, double scale_value,
    int64_t pre_tokens, int64_t next_tokens,
    c10::string_view input_layout, int64_t num_key_value_heads)
{
    // construct the output tensor of the NPU
    auto output = npu_preparation::apply_tensor_without_format(query);

    // convert str
    std::string input_layout_str = std::string(input_layout);
    char *input_layout_ptr = const_cast<char *>(input_layout_str.c_str());
    
    auto actSeqLen = (actual_seq_lengths.has_value()) ? actual_seq_lengths.value().vec() : std::vector<at::IntArrayRef::value_type>{};

    // dispatch hostAPI
    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnPromptFlashAttention, query, key, value, padding_mask, atten_mask, actSeqLen,
                                 num_heads, scale_value, pre_tokens, next_tokens, input_layout_ptr, num_key_value_heads, output);
    return output;
}

at::Tensor npu_incre_flash_attention(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &padding_mask, const c10::optional<at::Tensor> &atten_mask,
    c10::OptionalIntArrayRef actual_seq_lengths,
    int64_t num_heads, double scale_value, c10::string_view input_layout, int64_t num_key_value_heads)
{
    // construct the output tensor of the NPU
    auto output = npu_preparation::apply_tensor_without_format(query);

    // convert string
    std::string input_layout_str = std::string(input_layout);
    char *input_layout_ptr = const_cast<char *>(input_layout_str.c_str());

    at::TensorList keyTensors = {key};
    at::TensorList valueTensors = {value};

    auto actSeqLen = (actual_seq_lengths.has_value()) ? actual_seq_lengths.value().vec() : std::vector<at::IntArrayRef::value_type>{};

    // dispatch hostAPI
    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnIncreFlashAttention, query, keyTensors, valueTensors, padding_mask, atten_mask, actSeqLen,
                                 num_heads, scale_value, input_layout_ptr, num_key_value_heads, output);
    return output;
}
} // namespace op_api

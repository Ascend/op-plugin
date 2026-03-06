// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {

using npu_preparation = at_npu::native::OpPreparation;

constexpr int64_t MINIMUM_SHAPE_SIZE = 2;

at::Tensor npu_matmul_compress_dequant(const at::Tensor &x1, const at::Tensor &x2,
                                       const at::Tensor &compress_index, const at::Tensor &bias,
                                       const at::Tensor &scale,
                                       const c10::optional<at::Tensor> &offsetW,
                                       c10::optional<int64_t> offsetX)
{
    static const bool is_aclnn_available =
        check_aclnn_kernel_available("aclnnMatmulCompressDequant");
    TORCH_CHECK(is_aclnn_available,
                "aclnnMatmulCompressDequant or aclnnMatmulCompressDequantGetWorkspaceSize not found, "
                "please upgrade CANN.",
                OPS_ERROR(ErrCode::PARAM));

    // 约束1: offsetW 当前仅支持空指针传入
    TORCH_CHECK(!offsetW.has_value(),
                "offsetW currently only supports null/None, please do not pass offsetW.",
                OPS_ERROR(ErrCode::PARAM));

    // 约束2: offsetX 当前仅支持 0
    int64_t offset_x_val = offsetX.value_or(0);
    TORCH_CHECK(offset_x_val == 0,
                "offsetX currently only supports 0, but got ", offset_x_val,
                OPS_ERROR(ErrCode::PARAM));

    TORCH_CHECK(x1.dim() == MINIMUM_SHAPE_SIZE,
                "x1 must have 2 dimensions, but got ", x1.dim(),
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(scale.dim() == MINIMUM_SHAPE_SIZE,
                "scale must have 2 dimensions, but got ", scale.dim(),
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(bias.dim() == MINIMUM_SHAPE_SIZE,
                "bias must have 2 dimensions, but got ", bias.dim(),
                OPS_ERROR(ErrCode::PARAM));

    // 约束4: output 形状为 [M, N], M = x1 第1维, N = bias 第2维；格式 ND，数据类型 DT_FLOAT16
    int64_t M = x1.size(0);
    int64_t N = bias.size(1);
    c10::SmallVector<int64_t, op_infer::SIZE> output_size = {M, N};

    at::Tensor result = npu_preparation::apply_tensor_with_format(
        output_size, x1.options().dtype(at::kHalf), ACL_FORMAT_ND);

    // 约束3: compressInfo = {8, 8, k, n, 1}, k = x1 第2维, n = scale 第2维
    int64_t k = x1.size(1);
    int64_t n = scale.size(1);
    std::vector<int64_t> compress_info_vec = {8, 8, k, n, 1};
    at::IntArrayRef compress_info_ref(compress_info_vec);

    // offsetW 传空, offsetX 传 0 给 aclnn
    c10::optional<at::Tensor> offset_w_for_api = c10::nullopt;
    int offset_x_for_api = 0;

    EXEC_NPU_CMD(aclnnMatmulCompressDequant, x1, x2, compress_index, bias, scale,
                 offset_w_for_api, offset_x_for_api, compress_info_ref, result);

    return result;
}

}  // namespace op_api

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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
const size_t INDEX_ONE = 1;
const size_t INDEX_TWO = 2;
const size_t DIM_TREE = 3;

at::Tensor npu_moe_finalize_routing(const at::Tensor& expanded_permuted_rows, const c10::optional<at::Tensor>& skip1,
                                    const c10::optional<at::Tensor>& skip2,
                                    const c10::optional<at::Tensor>& bias,
                                    const c10::optional<at::Tensor>& scales,
                                    const at::Tensor& expanded_src_to_dst_row,
                                    const c10::optional<at::Tensor>& expert_for_source_row,
                                    const c10::optional<int64_t> drop_pad_mode)
{
    static const bool is_moe_finalize_routing_V2_available = check_aclnn_kernel_available("aclnnMoeFinalizeRoutingV2");
    if (!is_moe_finalize_routing_V2_available) {
        TORCH_CHECK(skip1.has_value(), "skip1 parameter must have value when there is no aclnnMoeFinalizeRoutingV2",
            OPS_ERROR(ErrCode::PARAM));
        at::Tensor result = npu_preparation::apply_tensor_without_format(skip1.value());
        EXEC_NPU_CMD(aclnnMoeFinalizeRouting, expanded_permuted_rows, skip1, skip2, bias, scales,
            expanded_src_to_dst_row, expert_for_source_row, result);
        return result;
    }

    at::Tensor result;
    int64_t dim0 = expanded_src_to_dst_row.size(0);
    if (scales.has_value()) {
        dim0 = scales.value().size(0);
    }
    at::SmallVector<int64_t, op_infer::SIZE> output_size;
    output_size.push_back(dim0);
    size_t dim1Index = INDEX_ONE;
    if (expanded_permuted_rows.dim() == DIM_TREE) {
        dim1Index = INDEX_TWO;
    }
    output_size.push_back(expanded_permuted_rows.size(dim1Index));
    result = npu_preparation::apply_tensor_without_format(output_size, expanded_permuted_rows.options());
    int64_t mode = c10::value_or_else(drop_pad_mode, [] { return 0; });
    EXEC_NPU_CMD(aclnnMoeFinalizeRoutingV2, expanded_permuted_rows, expanded_src_to_dst_row,
                 skip1, skip2, bias, scales, expert_for_source_row, mode, result);
    return result;
}
} // namespace op_api

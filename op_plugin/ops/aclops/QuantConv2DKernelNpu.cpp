// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
using npu_preparation = at_npu::native::OpPreparation;

constexpr int ATTRS_DIM = 2;
constexpr int TENSORS_DIM = 4;
constexpr int INPUT_H_INDEX = 2;
constexpr int INPUT_W_INDEX = 3;
constexpr int WEIGHT_W_INDEX = 3;

at::Tensor npu_quant_conv2d_out(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& scale,
                                c10::IntArrayRef stride, c10::IntArrayRef pad,
                                c10::IntArrayRef dilation, int64_t groups,
                                int64_t offset_x, c10::string_view round_mode, at::Tensor& output,
                                c10::optional<at::ScalarType> output_dtype,
                                const c10::optional<at::Tensor>& bias_opt, const c10::optional<at::Tensor>& offset)
{
    TORCH_CHECK(stride.size() >= ATTRS_DIM, "stride has to contain more than 2 elements, but got ", stride.size());
    TORCH_CHECK(pad.size() >= ATTRS_DIM, "padding has to contain more than 2 elements, but got ", pad.size());
    TORCH_CHECK(dilation.size() >= ATTRS_DIM, "dilation has to contain more than 2 elements, but got ",
        dilation.size());
    TORCH_CHECK(output_dtype == at::ScalarType::Half, "only support float16 as outputdtype");

    const at::Tensor &bias = c10::value_or_else(bias_opt, [] { return at::Tensor(); });
    c10::SmallVector<int64_t, N> strides = {1, 1, stride[0], stride[1]};
    c10::SmallVector<int64_t, N> paddings = {pad[0], pad[0], pad[1], pad[1]};
    c10::SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1]};
    at::ScalarType dtype = c10::value_or_else(output_dtype, [] {return at::ScalarType::Half;});

    int64_t dtype_enum = 0;
    if (dtype == at::ScalarType::Half) {
        dtype_enum = 1;
    }

    at_npu::native::OpCommand cmd;
    cmd.Name("QuantConv2D").Input(input, "x").Input(weight, "filter").Input(scale, "scale");
    if (bias.defined()) {
        cmd.Input(bias);
    }
    cmd.Output(output, "y")
        .Attr("dtype", dtype_enum)
        .Attr("strides", strides)
        .Attr("pads", paddings)
        .Attr("dilations", dilations)
        .Attr("groups", groups)
        .Attr("data_format", static_cast<std::string>("NCHW"))
        .Attr("offset_x", offset_x)
        .Attr("round_mode", static_cast<std::string>("rint"))
        .Run();

    return output;
}


at::Tensor npu_quant_conv2d(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& scale,
                            c10::IntArrayRef strides, c10::IntArrayRef pads,
                            c10::IntArrayRef dilations, int64_t groups,
                            int64_t offset_x, c10::string_view round_mode, c10::optional<at::ScalarType> output_dtype,
                            const c10::optional<at::Tensor>& bias, const c10::optional<at::Tensor>& offset)
{
    TORCH_CHECK(input.dim() >= TENSORS_DIM, "input has to be more than 4D, but got Tensor of dimension ", input.dim());
    TORCH_CHECK(weight.dim() >= TENSORS_DIM, "weight has to more than 4D, but got Tensor of dimension ", weight.dim());
    TORCH_CHECK(strides.size() >= ATTRS_DIM, "stride has to contain more than 2 elements, but got ", strides.size());
    TORCH_CHECK(pads.size() >= ATTRS_DIM, "padding has to contain more than 2 elements, but got ", pads.size());
    TORCH_CHECK(dilations.size() >= ATTRS_DIM, "dilation has to contain more than 2 elements, but got ",
        dilations.size());
    TORCH_CHECK(weight.size(WEIGHT_W_INDEX) != 0, "4th dim of weight cannot be 0");
    TORCH_CHECK(strides[0] * strides[1] != 0, "Stride cannot contain 0")

    int64_t N = input.size(0);
    int64_t H = input.size(INPUT_H_INDEX);
    int64_t W = input.size(INPUT_W_INDEX);
    int64_t Co = weight.size(0);
    auto kernel_size = weight.sizes().slice(2);

    int64_t Ho = (H + 2 * pads[0] - dilations[0] * (kernel_size[0] - 1) - 1) / strides[0] + 1;
    int64_t Wo = (W + 2 * pads[1] - dilations[1] * (kernel_size[1] - 1) - 1) / strides[1] + 1;

    TORCH_CHECK(Ho > 0, "Ho has to be positive, but got ", Ho);
    TORCH_CHECK(Wo > 0, "Wo has to be positive, but got ", Wo);

    c10::SmallVector<int64_t, SIZE> output_size = {N, Co, Ho, Wo};
    c10::TensorOptions options = input.options().dtype(at::kHalf);
    at::Tensor result = npu_preparation::apply_tensor_with_format(output_size, options, ACL_FORMAT_NCHW);

    acl_op::npu_quant_conv2d_out(input, weight, scale, strides, pads, dilations, groups, offset_x, round_mode,
        result, output_dtype, bias, offset);
    return result;
}
#endif
} // namespace acl_op

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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/AdvancedIndex.h"
#include "torch_npu/csrc/aten/mirror/NPUMemoryOverlap.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
void check_no_overlap(const at::Tensor& a, const at::Tensor& b)
{
    const auto overlap_status = at_npu::native::get_overlap_status(a, b);
    TORCH_CHECK(overlap_status != at_npu::native::MemOverlapStatus::PARTIAL &&
                    overlap_status != at_npu::native::MemOverlapStatus::FULL,
                "unsupported operation: some elements of the input tensor and "
                "the written-to tensor refer to a single memory location. "
                "Please clone() the tensor before performing the operation.",
                OPS_ERROR(ErrCode::NOT_SUPPORT));
}
} // namespace

at::Tensor index_put(
    const at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices,
    const at::Tensor& value,
    bool accumulate) {
    DO_COMPATIBILITY(aclnnIndexPutImpl, acl_op::index_put(self, indices, value, accumulate));
    return self.clone(at::MemoryFormat::Contiguous).index_put_(indices, value, accumulate);
}

at::Tensor& index_put_(
    at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices,
    const at::Tensor& value,
    const bool accumulate) {
    DO_COMPATIBILITY(aclnnIndexPutImpl, acl_op::index_put_(self, indices, value, accumulate));
    return at::_index_put_impl_(self, indices, value, accumulate, false);
}

inline std::tuple<bool, at::Tensor> canDispatchToMaskedFill(
    const at::Tensor& self,
    const torch::List<std::optional<at::Tensor>>& indices,
    const at::Tensor& value) {
  if (!(value.numel() == 1)) {
    return std::make_tuple(false, at::Tensor());
  }
  int64_t num_ind = 0;
  at::Tensor mask;
  auto self_device = self.device();
  for (const std::optional<at::Tensor>& i : indices) {
    if (!i.has_value() || !(*i).defined()) {
      num_ind++;
    } else {
      const at::Tensor& index = *i;
      if ((index.scalar_type() != c10::kByte && index.scalar_type() != c10::kBool) ||
          index.device() != self_device || mask.defined()) {
        return std::make_tuple(false, at::Tensor());
      } else {
        mask = index;
        for (const auto j : c10::irange(index.dim())) {
          int64_t srcIdx = num_ind + j;
          TORCH_CHECK_INDEX(
              index.size(j) == self.size(srcIdx),
              "The shape of the mask ",
              index.sizes(),
              " at index ",
              j,
              " does not match the shape of the indexed tensor ",
              self.sizes(),
              " at index ",
              srcIdx);
        }
        num_ind += mask.ndimension();
      }
    }
  }
  for ([[maybe_unused]] const auto i :
       c10::irange(num_ind, self.ndimension())) {
    mask = mask.unsqueeze(-1);
  }
  return std::make_tuple(true, mask);
}

at::Tensor& _index_put_impl_(
    at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices,
    const at::Tensor& value,
    const bool accumulate,
    const bool unsafe) {
    DO_COMPATIBILITY(aclnnIndexPutImpl, acl_op::_index_put_impl_(self, indices, value, accumulate, unsafe));
    if (self.device().type() == at::kCPU) {
        return at::native::_index_put_impl_(self, indices, value, accumulate, unsafe);
    }
    if (at_npu::native::has_internal_overlap(self) == at_npu::native::MemOverlap::YES) {
        TORCH_WARN(
            "Use of index_put_ on expanded tensors is deprecated. "
            "Please clone() the tensor before performing this operation. "
            "This also applies to advanced indexing e.g. tensor[indices] = tensor");
    }
    if (!accumulate) {
        static const std::set<at::ScalarType> masked_fill_dtypes = {
            at::kFloat, at::kHalf, at::kInt, at::kLong, at::kChar, at::kBool, at::kBFloat16};
        if (masked_fill_dtypes.count(self.scalar_type()) && masked_fill_dtypes.count(value.scalar_type())) {
            auto masked_fill_dispatch = canDispatchToMaskedFill(self, indices, value);
            if (std::get<0>(masked_fill_dispatch)) {
                return self.masked_fill_(std::get<1>(masked_fill_dispatch), value.item());
            }
        }
    }
    check_no_overlap(self, value);
    for (const c10::optional<at::Tensor>& index : indices) {
        if (index.has_value()) {
            check_no_overlap(self, *index);
        }
    }

    bool needCast = op_plugin::AdvanceIndex::checkIndexTensorTypes(indices);
    auto indices_after = op_plugin::AdvanceIndex::npu_expand_tensors(self, indices, needCast, true);
    std::vector<at::Tensor> all_defined_indices;
    at::SmallVector<int64_t, op_infer::N> zeroSize = {0};
    at::Tensor emptyTensor = npu_preparation::apply_tensor_without_format(self, zeroSize);
    c10::List<c10::optional<at::Tensor>> indices_expand_list;
    for (at::Tensor index_opt : indices_after) {
        indices_expand_list.push_back(index_opt);
    }
    auto info = op_plugin::AdvanceIndex::make_info(self, indices_expand_list);
    TORCH_CHECK(op_plugin::AdvanceIndex::is_expandable_to(value.sizes(), info.src.sizes()),
        "shape mismatch: value tensor of shape ", value.sizes(),
        " cannot be broadcast to indexing result of shape ", info.src.sizes(),
        OPS_ERROR(ErrCode::PARAM));
    for (int i = 0; i < static_cast<int>(indices_after.size()); i++) {
        if (indices_after[i].defined()) {
            all_defined_indices.emplace_back(indices_after[i]);
        continue;
        }
    all_defined_indices.emplace_back(emptyTensor);
    }

  for (auto &all_defined_indice : all_defined_indices) {
      if (all_defined_indice.device() != self.device()) {
          all_defined_indice = all_defined_indice.to(self.device());
      }
  }
  at::TensorList indices_tensor_list = all_defined_indices;
  if (self.numel() != 0 && value.numel() != 0) {
      EXEC_NPU_CMD(aclnnIndexPutImpl, self, indices_tensor_list, value, accumulate, unsafe);
  }
  return self;
}

}

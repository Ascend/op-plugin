// Copyright (c) 2025 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.

#include "op_plugin/OpApiInterface.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "op_plugin/utils/OpAdapter.h"


namespace op_api {
const int DIMENSION_2D = 2;
at::Tensor _weight_norm(
    const at::Tensor& v_in,
    const at::Tensor& g_in,
    int64_t dim)
{
    TORCH_CHECK(
        v_in.device() == g_in.device(),
        "weight_norm: expected v_in and g_in to be on the same device, but v_in is "
        "on ", v_in.device(), " and g_in is on ", g_in.device(), OPS_ERROR(ErrCode::PARAM));
    auto v = v_in.contiguous();
    auto g = g_in.contiguous();
    return v * g.div(at::norm_except_dim(v, DIMENSION_2D, dim));
}

} // namespace op_api

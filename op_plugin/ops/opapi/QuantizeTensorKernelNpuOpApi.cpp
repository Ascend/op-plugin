// Copyright (c) 2026 Huawei Technologies Co., Ltd
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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

#if VERSION_BETWEEN(V2R2, VERSION_NEWEST)
#include <ATen/native/quantized/AffineQuantizer.h>

namespace op_api {
at::Tensor dequantize(const at::Tensor& self)
{
    return at::native::dequantize_quantized(self);
}
} // namespace op_api
#endif

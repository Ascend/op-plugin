// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifndef EXTENTION_OP_TILING_H
#define EXTENTION_OP_TILING_H
#include "tiling/tiling_api.h"

namespace ascendc_path {

optiling::TCubeTiling MatmulLeakyreluGenerateTiling();
} // namespace ascendc_path


#endif

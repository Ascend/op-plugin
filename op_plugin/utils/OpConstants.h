// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#ifndef __TORCH_NPU_OP_PLUGIN_UTILS_OPCONSTANTS__
#define __TORCH_NPU_OP_PLUGIN_UTILS_OPCONSTANTS__

// smallvector max size
const int N = 32;
// npu tensor max size
const int SIZE = 8;
// HALF_MAX and HALF_MIN of NPU support
const int NPU_HALF_MAX = 65504;
const int NPU_HALF_MIN = -65504;
const int NPU_MAX_OP_EXEC_TRY_NUM = 2;

#endif

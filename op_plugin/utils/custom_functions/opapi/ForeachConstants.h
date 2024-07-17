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

#ifndef __TORCH_NPU_OP_PLUGIN_UTILS_FOREACHCONSTANTS__
#define __TORCH_NPU_OP_PLUGIN_UTILS_FOREACHCONSTANTS__

constexpr int SINGLE_FOREACH_OP_TENSOR_COUNT = 48;      // 单动态输入输出最佳切分长度1
constexpr int SINGLE_FOREACH_SCALAR_OP_TENSOR_COUNT = 50;      // 单动态输入输出最佳切分长度2
constexpr int DOUBLE_FOREACH_OP_TENSOR_COUNT = 24;      // 双动态输入输出子最佳切分长度
constexpr int TRIPLE_FOREACH_OP_TENSOR_COUNT = 16;      // 三动态输入输出最佳切分长度
constexpr int QUADRA_FOREACH_OP_TENSOR_COUNT = 12;  // 四动态输入输出最佳切分长度

#endif

// Copyright (c) 2024 Huawei Technologies Co., Ltd
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

#ifndef __TORCH_NPU_OP_PLUGIN_UTILS_UPSAMPLECONSTANTS__
#define __TORCH_NPU_OP_PLUGIN_UTILS_UPSAMPLECONSTANTS__

constexpr double MAX_SUPPORT_SCALE = 50.0;      // 最大缩放系数
constexpr double NEAREST_MAX_SCALE = 100.0;     // 最近邻最大缩放系数
constexpr double BILINEAR_MIN_SCALE = 0.02;     // 双线性最小缩放系数
constexpr double BICUBIC_MIN_SCALE = 0.03;      // 双三次最小缩放系数
constexpr int BICUBIC_MAX_SHAPE = 8192;         // 双三次最大shape
constexpr int H_INDEX = 2;                      // 高度下标
constexpr int W_INDEX = 3;                      // 宽度下标
constexpr int DIM_D = 2;
constexpr int DIM_H = 3;
constexpr int DIM_W = 4;

#endif

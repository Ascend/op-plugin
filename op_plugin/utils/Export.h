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

#ifndef OP_PLUGIN_UTILS_EXPORT
#define OP_PLUGIN_UTILS_EXPORT

#if defined(_MSC_VER)
#define OP_PLUGIN_HIDDEN
#endif // _MSC_VER
#if defined(__GNUC__)
#define OP_PLUGIN_HIDDEN __attribute__((visibility("hidden")))
#else // __GNUC__
#define OP_PLUGIN_HIDDEN
#endif // __GNUC__

#endif // OP_PLUGIN_UTILS_EXPORT

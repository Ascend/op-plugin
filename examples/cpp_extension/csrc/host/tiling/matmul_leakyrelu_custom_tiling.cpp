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

#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include "op_tiling.h"
using namespace matmul_tiling;
using namespace std;

namespace ascendc_path {

optiling::TCubeTiling MatmulLeakyreluGenerateTiling()
{
    int M = 1024;
    int N = 640;
    int K = 256;
    int baseM = 256;
    int baseN = 128;
    TPosition leftPos = TPosition::GM;
    CubeFormat leftFormat = CubeFormat::ND;
    DataType leftDtype = DataType::DT_FLOAT16;
    int transposeA = 0;

    TPosition rightPos = TPosition::GM;
    CubeFormat rightFormat = CubeFormat::ND;
    DataType rightDtype = DataType::DT_FLOAT16;
    int transposeB = 0;

    TPosition resPos = TPosition::GM;
    CubeFormat resFormat = CubeFormat::ND;
    DataType resDtype = DataType::DT_FLOAT;

    TPosition biasPos = TPosition::GM;
    CubeFormat biasFormat = CubeFormat::ND;
    DataType biasDtype = DataType::DT_FLOAT;
    bool isBias = true;
    int usedCoreNum = 2;
    optiling::TCubeTiling tilingData;
    tilingData.set_usedCoreNum(usedCoreNum);
    MultiCoreMatmulTiling tilingApi;
    tilingApi.SetDim(usedCoreNum);
    tilingApi.SetAType(leftPos, leftFormat, leftDtype, bool(transposeA));
    tilingApi.SetBType(rightPos, rightFormat, rightDtype, bool(transposeB));
    tilingApi.SetCType(resPos, resFormat, resDtype);
    tilingApi.SetBiasType(biasPos, biasFormat, biasDtype);

    tilingApi.SetOrgShape(M, N, K);
    tilingApi.SetShape(M, N, K);
    tilingApi.SetBias(isBias);
    tilingApi.SetTraverse(MatrixTraverse::FIRSTM);
    tilingApi.SetFixSplit(baseM, baseN, -1);
    tilingApi.SetBufferSpace(-1, -1, -1);
    int64_t res = tilingApi.GetTiling(tilingData);
    tilingData.set_stepM(1);
    tilingData.set_stepN(1);
    if (res == -1) {
        std::cout << "gen tiling failed" << std::endl;
    }
    return tilingData;
}
} // namespace ascendc_path

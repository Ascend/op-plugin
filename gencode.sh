#!/bin/bash

# Copyright (c) 2023 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION.
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

CDIR="$(cd "$(dirname "$0")" ; pwd -P)"

cd $CDIR/

PYTORCH_VERSION="$1"
IFS='.' read -ra version_parts <<< "$PYTORCH_VERSION"
PYTORCH_VERSION_DIR="v${version_parts[0]}r${version_parts[1]}"
python_execute="$2"

export PYTORCH_VERSION="$PYTORCH_VERSION"

${python_execute} -m codegen.gen_backend_stubs  \
  --version="$PYTORCH_VERSION" \
  --output_dir="$CDIR/op_plugin/" \
  --source_yaml="$CDIR/op_plugin/config/$PYTORCH_VERSION_DIR/op_plugin_functions.yaml" \
  --impl_path="$CDIR/torch_npu/csrc/aten"  # Used to double-check the yaml file definitions.

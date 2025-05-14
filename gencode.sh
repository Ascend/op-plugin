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

# Following updates only add incompatible ops files in corresponding branch folder.
newest_minor_version=8
for minor_version in $(seq 1 ${newest_minor_version}); do
    # Merge base info and version related info (unsupported ops)
    sed -i "1r test/unsupported_ops_info_base.yaml" test/test_v2r${minor_version}_ops/unsupported_ops_info.yaml
done

PYTORCH_VERSION="$1"
IFS='.' read -ra version_parts <<< "$PYTORCH_VERSION"
PYTORCH_VERSION_DIR="v${version_parts[0]}r${version_parts[1]}"
python_execute="$2"

export PYTORCH_VERSION="$PYTORCH_VERSION"

OUTPUT_DIR="$CDIR/op_plugin/config/$PYTORCH_VERSION_DIR"
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir "$OUTPUT_DIR"
fi

if [ "$PYTORCH_VERSION_DIR" == "v1r11" ]; then
    cp $CDIR/op_plugin/config/aclnn_derivatives.yaml $OUTPUT_DIR
fi

${python_execute} -m codegen.gen_op_plugin_functions  \
  --version="$PYTORCH_VERSION" \
  --output_dir="$OUTPUT_DIR/" \
  --source_yaml="$CDIR/op_plugin/config/op_plugin_functions.yaml"

${python_execute} -m codegen.gen_derivatives  \
  --version="$PYTORCH_VERSION" \
  --output_dir="$OUTPUT_DIR/" \
  --source_yaml="$CDIR/op_plugin/config/derivatives.yaml"

${python_execute} -m codegen.gen_backend_stubs  \
  --version="$PYTORCH_VERSION" \
  --output_dir="$CDIR/op_plugin/" \
  --source_yaml="$OUTPUT_DIR/op_plugin_functions.yaml" \
  --impl_path="$CDIR/torch_npu/csrc/aten"  # Used to double-check the yaml file definitions.

${python_execute} -m codegen.struct.gen_struct_opapi  \
  --output_dir="$CDIR/op_plugin/ops/opapi/" \
  --native_yaml="$OUTPUT_DIR/op_plugin_functions.yaml" \
  --struct_yaml="$CDIR/op_plugin/config/op_plugin_functions.yaml"

sh $CDIR/op_plugin/third_party/atb/libs/build_stub.sh

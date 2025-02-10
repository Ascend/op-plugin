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

set -e

CUR_DIR=$(dirname $(readlink -f $0))
PY_VERSION='3.8' # Default supported python version is 3.8
PYTORCH_VERSION='master' # Default supported PyTorch branch is master
DEFAULT_SCRIPT_ARGS_NUM_MAX=2 # Default max supported input parameters

# Parse arguments inside script
function parse_script_args() {
    local args_num=0

    while true; do
        if [[ "x${1}" = "x" ]]; then
            break
        fi
        if [[ "$(echo "${1}"|cut -b1-|cut -b-2)" == "--" ]]; then
            args_num=$((args_num+1))
        fi
        if [[ "x${2}" = "x" ]]; then
            break
        fi
        if [[ "$(echo "${2}"|cut -b1-|cut -b-2)" == "--" ]]; then
            args_num=$((args_num+1))
        fi
        if [[ ${args_num} -eq ${DEFAULT_SCRIPT_ARGS_NUM_MAX} ]]; then
            break
        fi

    done

    while true; do
        case "${1}" in
        --python=*)
            PY_VERSION=$(echo "${1}"|cut -d"=" -f2)
            args_num=$((args_num-1))
            shift
            ;;
        --pytorch=*)
            PYTORCH_VERSION=$(echo "${1}"|cut -d"=" -f2)
            args_num=$((args_num-1))
            shift
            ;;
        -*)
            echo "ERROR Unsupported parameters: ${1}"
            return 1
            ;;
        *)
            if [ "x${1}" != "x" ]; then
                echo "ERROR Unsupported parameters: ${1}"
                return 1
            fi
            break
            ;;
        esac
    done

    # if some "--param=value" are not parsed correctly, throw an error.
    if [[ ${args_num} -ne 0 ]]; then
        return 1
    fi
}

function checkout_pytorch_branch() {
    cd ${PYTORCH_PATH}
    current_torch_branch=$(git symbolic-ref --short HEAD)
    if [ "${current_torch_branch}" != "${PYTORCH_VERSION}" ]; then
        if [ -d ${PYTORCH_PATH}/third_party/op-plugin ]; then
            rm -r ${PYTORCH_PATH}/third_party/op-plugin
        fi
        echo "checkout to torch expected-branch[ ${PYTORCH_VERSION} ] "
        git checkout "${PYTORCH_VERSION}" --recurse-submodules;
        git checkout .;git clean -fdx;
    fi
    cd ${CUR_DIR}/../
}

function main()
{
    if ! parse_script_args "$@"; then
        echo "Failed to parse script args. Please check your inputs."
        exit 1
    fi

    CODE_ROOT_PATH=${CUR_DIR}/../
    # clone torch_adapter
    BUILD_PATH=${CODE_ROOT_PATH}/build
    PYTORCH_PATH=${BUILD_PATH}/pytorch
    if [ ! -d ${PYTORCH_PATH} ]; then
        if [ -d ${BUILD_PATH} ]; then
            rm -r ${BUILD_PATH}
        fi
        git clone -b ${PYTORCH_VERSION} https://gitee.com/ascend/pytorch.git ${PYTORCH_PATH}
    fi
    checkout_pytorch_branch

    # download third_party of torch_npu
    cd ${PYTORCH_PATH}
    git submodule update --init --depth=1 --recursive

    # copy op_plugin to torch_adapter/third_party
    PYTORCH_THIRD_PATH=${PYTORCH_PATH}/third_party/op-plugin
    if [ -d ${PYTORCH_THIRD_PATH}/op_plugin ]; then
        rm -r ${PYTORCH_THIRD_PATH}/*
    else
        mkdir -p ${PYTORCH_THIRD_PATH}
    fi

    cp -rf ${CODE_ROOT_PATH}/op_plugin ${PYTORCH_THIRD_PATH}/
    cp -rf ${CODE_ROOT_PATH}/codegen ${PYTORCH_THIRD_PATH}/
    cp -rf ${CODE_ROOT_PATH}/*.sh ${PYTORCH_THIRD_PATH}/
    cp -rf ${CODE_ROOT_PATH}/test ${PYTORCH_THIRD_PATH}/

    # compile torch_adapter
    export BUILD_WITHOUT_SHA=1
    if [[ "${PYTORCH_VERSION}" == v1.11.0* ]] || [[ "${PYTORCH_VERSION}" == v2.0.1* ]]; then
        bash ${PYTORCH_PATH}/ci/build.sh --python=${PY_VERSION}
    else
        bash ${PYTORCH_PATH}/ci/build.sh --python=${PY_VERSION} --disable_torchair --disable_rpc
    fi

    # copy dist/torch_npu.whl from torch_adapter to op_plugin
    if [ -d ${CODE_ROOT_PATH}/dist ]; then
        rm -r ${CODE_ROOT_PATH}/dist
    fi
    cp -rf ${PYTORCH_PATH}/dist ${CODE_ROOT_PATH}

    exit 0
}

main "$@"

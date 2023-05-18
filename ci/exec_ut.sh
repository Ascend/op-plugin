#!/bin/bash

# Copyright (c) 2023 Huawei Technologies Co., Ltd
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
SUPPORTED_PY_VERSION=(3.8 3.9)
SUPPORTED_PYTORCH_VERSION=('master' 'v2.0.0')
PY_VERSION='3.8' # Default supported python version is 3.8
PYTORCH_VERSION='master' # Default supported PyTorch version is master
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

function check_python_version() {
    matched_py_version='false'
    for ver in ${SUPPORTED_PY_VERSION[*]}; do
        if [ "${PY_VERSION}" = "${ver}" ]; then
            matched_py_version='true'
            return 0
        fi
    done
    if [ "${matched_py_version}" = 'false' ]; then
        echo "${PY_VERSION} is an unsupported python version, we suggest ${SUPPORTED_PY_VERSION[*]}"
        exit 1
    fi
}

function check_pytorch_version() {
    matched_pytorch_version='false'
    for ver in ${SUPPORTED_PY_VERSION[*]}; do
        if [ "${PYTORCH_VERSION}" = "${ver}" ]; then
            matched_py_version='true'
            return 0
        fi
    done
    if [ "${matched_py_version}" = 'false' ]; then
        echo "${PYTORCH_VERSION} is an unsupported pytorch version, we suggest ${SUPPORTED_PYTORCH_VERSION[*]}"
        exit 1
    fi
}

function main()
{
    if ! parse_script_args "$@"; then
        echo "Failed to parse script args. Please check your inputs."
        exit 1
    fi
    check_python_version
    check_pytorch_version

    # clone torch_adapter for ops ut
    PYTORCH_PATH=${CUR_DIR}/../pytorch_ut
    if [ -d ${PYTORCH_PATH} ]; then
        rm -r ${PYTORCH_PATH}
    fi
    git clone -b ${PYTORCH_VERSION} https://gitee.com/ascend/pytorch.git ${PYTORCH_PATH}

    # copy modify_files.txt to torch_adapter/ci
    cp ${CUR_DIR}/../modify_files.txt ${PYTORCH_PATH}/

    # exec ut
    cd ${PYTORCH_PATH}/ci
    python"${PY_VERSION}" access_control_test.py

    exit 0
}

main "$@"
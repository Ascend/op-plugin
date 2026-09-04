#!/bin/bash
set -e

ARCH=$(uname -m)
CUR_DIR=$(dirname $(readlink -f $0))
PY_VERSION='3.8' # Default supported python version is 3.8
PYTORCH_VERSION='master' # Default supported PyTorch version is master
DEFAULT_SCRIPT_ARGS_NUM_MAX=3 # Default max supported input parameters
PR_ID='1'
export LD_PRELOAD=$LD_PRELOAD:/lib64/libgomp.so.1

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
        if [[ "x${3}" = "x" ]]; then
            break
        fi
        if [[ "$(echo "${3}"|cut -b1-|cut -b-2)" == "--" ]]; then
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
        --pr_id=*)
            PR_ID=$(echo "${1}"|cut -d"=" -f2)
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

    if [[ ${args_num} -ne 0 ]]; then
        return 1
    fi
}

function main()
{
    if ! parse_script_args "$@"; then
        echo "Failed to parse script args. Please check your inputs."
        exit 1
    fi
    cd ${CUR_DIR}
    python"${PY_VERSION}" access_control_test.py --rank=${TEST_RANK} --world_size=${TEST_WORLD_SIZE}

    PYTORCH_PATH=${CUR_DIR}/../pytorch_ut
    if [ ! -d ${PYTORCH_PATH} ]; then
        bash "${WORKSPACE}"/cie/.gitcode/workflows/scripts/common/retry_command.sh "git clone -b ${PYTORCH_VERSION} https://gitcode.com/Ascend/pytorch.git ${PYTORCH_PATH}"
        cd ${PYTORCH_PATH}
        bash "${WORKSPACE}"/cie/.gitcode/workflows/scripts/common/retry_command.sh "git submodule update --init --recursive --force third_party/op-plugin  third_party/torchair/torchair"
        if [ -f "${WORKSPACE}/CODE/test/allowlist_for_publicAPI.json" ];then
           cp -f ${WORKSPACE}/CODE/test/allowlist_for_publicAPI.json ./third_party/op-plugin/test/allowlist_for_publicAPI.json
        fi
        cd -
        if [ "${PYTORCH_VERSION}" \> "v2.0.1" ] || [ "${PYTORCH_VERSION}" == "master" ]; then
            python"${PY_VERSION}" delete_torchair_base_schema.py -s "${PYTORCH_PATH}"/test/torch_npu_schema.json
            if [ -f "${PYTORCH_PATH}/third_party/torchair/torchair/tests/st/torch_npu_schema.json" ];then
               python"${PY_VERSION}" delete_torchair_base_schema.py -s "${PYTORCH_PATH}"/third_party/torchair/torchair/tests/st/torch_npu_schema.json
            fi
        fi
    fi
    # GitCode runner 工作目录含仓库名 op-plugin，导致 test/npu 用例被误判为 op-plugin 用例
    if [ -f "${PYTORCH_PATH}/ci/access_control_test.py" ]; then
        sed -i "s/'op-plugin' in str(Path(ut_file))/Path(ut_file).is_relative_to(NETWORK_OPS_DIR)/g" \
            "${PYTORCH_PATH}"/ci/access_control_test.py
    fi
    if [ -f "${PYTORCH_PATH}/ci/split_by_time.py" ]; then
        sed -i "s/'op-plugin' in str(Path(ut_file))/Path(ut_file).is_relative_to(NETWORK_OPS_DIR)/g" \
            "${PYTORCH_PATH}"/ci/split_by_time.py
    fi

    # copy modify_files.txt to torch_adapter/ci
    cp ${CUR_DIR}/../modify_files.txt ${PYTORCH_PATH}/
    echo "copy common_files.txt for not master branch!"
    if [ "${PYTORCH_VERSION:0:6}" != "master" ]; then
        cp ${CUR_DIR}/../common_files.txt ${PYTORCH_PATH}/
    fi

    cd "${PYTORCH_PATH}"
    echo "sync with github"
    branch_path=${PYTORCH_VERSION%%-*}
    branch_path=${branch_path%%_*}
    echo "branch:${branch_path}"
    temp_folder="testcase_temp"
    rm -rf "${temp_folder}"
    mkdir -p "${temp_folder}"
    wget --no-host-directories -c -q --no-check-certificate https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/test/common_testcase/${branch_path}/testcase.tar.gz -O "${temp_folder}/testcase.tar.gz"
    tar -zxf "${temp_folder}/testcase.tar.gz" -C "${temp_folder}"

    if [ -e "${PYTORCH_PATH}"/test/testfiles_synchronized.txt ]; then
        cd "${temp_folder}"
        cat "${PYTORCH_PATH}"/test/testfiles_synchronized.txt | while IFS= read -r line; do
            line=$(echo "${line}" | tr -d '\r')
            if [ "${line}" != "" ]; then
                mkdir -p "${PYTORCH_PATH}"/"$(dirname "${line}")"
                cp -rf "${line}" "${PYTORCH_PATH}"/"${line}" || echo "testfiles copy failed: ${line}"
            fi
        done
        cd "${PYTORCH_PATH}"
    fi
    if [ -e "${PYTORCH_PATH}"/test/testfolder_synchronized.txt ]; then
        cd "${temp_folder}"
        cat "${PYTORCH_PATH}"/test/testfolder_synchronized.txt | while IFS= read -r line; do
            line=$(echo "${line}" | tr -d '\r')
            if [ "${line}" != "" ]; then
                cp -rf "${line}" "${PYTORCH_PATH}"/"${line}" || echo "testfolder copy failed: ${line}"
            fi
        done
        cd "${PYTORCH_PATH}"
    fi
    rm -rf "${temp_folder}"

    echo "Skip several test cases due to bug of py39."
    rm -rf ${PYTORCH_PATH}/test/test_jit.py
    rm -rf ${PYTORCH_PATH}/test/test_jit*.py
    rm -rf ${PYTORCH_PATH}/test/jit
    sed -i "s/'test_jit_fuser_te',/ /g" ${PYTORCH_PATH}/ci/access_control/constants.py

    rm -rf ${PYTORCH_PATH}/test/distributed/test_register_sharding.py
    rm -rf ${PYTORCH_PATH}/test/distributed/fsdp2
    if [ "${ARCH}" == "aarch64" ]; then
        rm -rf ${PYTORCH_PATH}/test/trans_contiguous
    fi

    if [ "${PYTORCH_VERSION}" == "master" ]; then
        export DISABLED_TESTS_FILE=${PYTORCH_PATH}/test/unsupported_test_cases/.pytorch-disabled-tests.json
    fi
    rm -rf ${PYTORCH_PATH}/test/dynamo/*
    rm -rf ${PYTORCH_PATH}/test/_inductor/*

    cd ${PYTORCH_PATH}/ci
    echo "Start to run access_control_test in PTA."
    python"${PY_VERSION}" access_control_test.py --rank=${TEST_RANK} --world_size=${TEST_WORLD_SIZE}
    exit 0
}

main "$@"

#!/bin/bash
# 用于op-plugin门禁UT 脚本
# shellcheck source=/dev/null

pytorch_version=$1
py_version=$2
pr_id=$3
WORKSPACE=$4
TARGET_BRANCH=$5
CUR_DIR=$(dirname $(readlink -f $0))
ARCH=$(uname -m)
source /etc/profile
export OMP_NUM_THREADS=32
export PTA_UT_EXEC_TIMEOUT=4800
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

python_path=$(dirname $(dirname $(readlink -f $(which python$py_version))))

function torch_install_other() {
    if [ "${ARCH}" == "x86_64" ]; then
        if [ "${1}" == "2.7.1" ] || [ "${1}" == "2.8.0" ] || [ "${1}" == "2.9.0" ] || [ "${1}" == "2.9.1" ] || [ "${1}" == "2.10.0" ] || [ "${1}" == "2.11.0" ] || [ "${1}" == "2.12.0" ]; then
            wget --no-check-certificate --no-verbose -P "${WORKSPACE}" https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/v${1}/torch-${1}%2Bcpu-cp${2}-cp${2}-manylinux_2_28_x86_64.whl
        else
            wget --no-check-certificate --no-verbose -P "${WORKSPACE}" https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/v${1}/torch-${1}%2Bcpu-cp${2}-cp${2}-linux_x86_64.whl
        fi
    elif [ "${ARCH}" == "aarch64" ]; then
        if [ "${1}" == "2.6.0" ] || [ "${1}" == "2.7.1" ] || [ "${1}" == "2.8.0" ] || [ "${1}" == "2.9.0" ] || [ "${1}" == "2.10.0" ]; then
            wget --no-check-certificate --no-verbose -P "${WORKSPACE}" https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/v${1}/torch-${1}-cp${2}-cp${2}-manylinux_2_28_aarch64.whl
        elif [ "${1}" == "2.9.1" ] || [ "${1}" == "2.11.0" ] || [ "${1}" == "2.12.0" ]; then
            wget --no-check-certificate --no-verbose -P "${WORKSPACE}" https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/v${1}/torch-${1}%2Bcpu-cp${2}-cp${2}-manylinux_2_28_aarch64.whl
        else
            wget --no-check-certificate --no-verbose -P "${WORKSPACE}" https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/v${1}/torch-${1}-cp${2}-cp${2}-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
        fi
    fi
}

function torch_install_master() {
    if [ "${ARCH}" == "x86_64" ]; then
        wget --no-check-certificate --no-verbose -P ${WORKSPACE} https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/master/${nightly_date}/torch-${nightly_version}%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl
    elif [ "${ARCH}" == "aarch64" ]; then
        wget --no-check-certificate --no-verbose -P ${WORKSPACE} https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/master/${nightly_date}/torch-${nightly_version}%2Bcpu-cp310-cp310-manylinux_2_28_aarch64.whl
    fi
}

function torch_install_master_2() {
    test_version=${nightly_version:0:6}
    if [ "${ARCH}" == "x86_64" ]; then
        wget --no-check-certificate --no-verbose -P "${WORKSPACE}" https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/v${test_version}/torch-${test_version}%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl
    elif [ "${ARCH}" == "aarch64" ]; then
        wget --no-check-certificate --no-verbose -P "${WORKSPACE}" https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/v${test_version}/torch-${test_version}%2Bcpu-cp310-cp310-manylinux_2_28_aarch64.whl
    fi
}

function torch_npu_install() {
    pip"${py_version}" install -U "${WORKSPACE}"/torch-*${ARCH}.whl --force-reinstall
    ls "${WORKSPACE}"
    pip"${py_version}" install -U "${WORKSPACE}"/torch_npu*${ARCH}.whl --force-reinstall --no-deps
}

function main() {
    pip"${py_version}" install https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/Triton_Innersource/triton-ascend/20260428190056/triton_ascend-3.2.1-cp310-cp310-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl
    cd ${WORKSPACE}
    rm -rf "${python_path}"/lib/python"${py_version}"/site-packages/torch
    rm -rf "${python_path}"/lib/python"${py_version}"/site-packages/~orch
    rm -rf "${python_path}"/lib/python"${py_version}"/site-packages/-orch
    rm -rf "${python_path}"/lib/python"${py_version}"/site-packages/torch*.dist-info
    rm -rf "${python_path}"/lib/python"${py_version}"/site-packages/torchgen
    rm -rf "${python_path}"/lib/python"${py_version}"/site-packages/~orchgen
    rm -rf "${python_path}"/lib/python"${py_version}"/site-packages/-orchgen
    rm -rf "${python_path}"/lib/python"${py_version}"/site-packages/functorch
    rm -rf "${python_path}"/lib/python"${py_version}"/site-packages/~unctorch
    rm -rf "${python_path}"/lib/python"${py_version}"/site-packages/-unctorch
    rm -rf "${python_path}"/lib/python"${py_version}"/site-packages/torch_npu
    rm -rf "${python_path}"/lib/python"${py_version}"/site-packages/torch_npu*.dist-info
    if [ -d "/root/.cache/torch_extensions/py${py_version//./}_cpu/custom_add" ];then
        rm -rf /root/.cache/torch_extensions/py"${py_version//./}"_cpu/custom_add
    fi


    if [ ${pytorch_version:0:6} == "v2.1.0" ] || [ ${pytorch_version:0:6} == "v2.6.0" ] || [ ${pytorch_version:0:6} == "v2.7.1" ] || [ ${pytorch_version:0:6} == "v2.8.0" ] || [ ${pytorch_version:0:6} == "v2.9.0" ]; then
        torch_install_other "${pytorch_version:1:5}" "${py_version//./}"
    elif [ ${pytorch_version:0:7} == "v2.10.0" ] || [ ${pytorch_version:0:7} == "v2.11.0" ]|| [ ${pytorch_version:0:7} == "v2.12.0" ] ; then
        torch_install_other "${pytorch_version:1:6}" "${py_version//./}"
    elif [ "${pytorch_version}" = "master" ]; then
        bash ${WORKSPACE}/cie/.gitcode/workflows/scripts/common/retry_command.sh "rm -rf ${WORKSPACE}/tmp_pytorch && git clone --depth=1 https://gitcode.com/Ascend/pytorch.git ${WORKSPACE}/tmp_pytorch"
        nightly_version=`cat ${WORKSPACE}/tmp_pytorch/requirements.txt | grep torch== | cut -c 8-25`
        nightly_date=${nightly_version:10:8}
        echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        echo "${nightly_version}"
        echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        if [ ${#nightly_date} -ne 8 ]; then
            torch_install_master_2
        else
            torch_install_master
        fi
        pip"${py_version}" install -r ${WORKSPACE}/tmp_pytorch/test/requirements.txt
    fi
    torch_npu_install
    pip${py_version} install einops
    pip${py_version} install tlparse==0.3.18
    pip${py_version} install tabulate==0.9.0
    pip${py_version} list

    cp -rf modify_files.txt ./CODE
    if [ "${pytorch_version}" != "master" ]; then
        cp -f ${WORKSPACE}/cie/.gitcode/workflows/scripts/UT/op-plugin/common_files.txt ./CODE
    fi
    cd ${WORKSPACE}/CODE
    rm -rf ci/exec_ut.sh
    cp -f ${WORKSPACE}/cie/.gitcode/workflows/scripts/UT/op-plugin/exec_ut.sh ${WORKSPACE}/CODE/ci/exec_ut.sh
    cp -f ${WORKSPACE}/cie/.gitcode/workflows/scripts/UT/op-plugin/delete_torchair_base_schema.py ${WORKSPACE}/CODE/ci/delete_torchair_base_schema.py

    pkill -f /usr/local/bin/python3 || echo "IGNORE KILL ERROR" > /dev/null
    pkill -f /opt/_internal/cpython || echo "IGNORE KILL ERROR" > /dev/null
    pkill -f "${python_path}"/bin/python"${py_version}" || echo "IGNORE KILL ERROR" > /dev/null
    pkill -f /usr/local/bin/python"${py_version}" || echo "IGNORE KILL ERROR" > /dev/null
    pkill -f python"${py_version}" || echo "IGNORE KILL ERROR" > /dev/null
    ps -ef | grep "/usr/local/bin/python3" | grep -v grep | awk '{print $2}' | xargs kill -9 2> /dev/null || echo "IGNORE KILL ERROR" > /dev/null
    ps -ef | grep "/opt/_internal/cpython" | grep -v grep | awk '{print $2}' | xargs kill -9 2> /dev/null || echo "IGNORE KILL ERROR" > /dev/null

    export PATH="${python_path}"/bin:$PATH
    python"${py_version}" -c "import acl;print('npu info:',acl.get_soc_name())" || echo "npu info  get error"

    echo "------------------------------------- ENV PATH -------------------------------------"
    echo ${PATH}
    echo "------------------------------------------------------------------------------------"

    if [ "${TARGET_BRANCH}" == "master" ]; then
        sh ci/exec_ut.sh --python=${py_version} --pytorch=${pytorch_version} --pr_id=${pr_id}
    else
        sh ci/exec_ut.sh --python=${py_version} --pytorch=${pytorch_version}-${TARGET_BRANCH} --pr_id=${pr_id}
    fi
}

set -e

echo "start execute ut"
main
echo "end execute ut"

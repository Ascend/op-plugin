#!/bin/bash
# ============================================================================
# 并行编译 + 安装 + 测试 examples 下的 9 个工程
#
# 用法:
#   bash parallel_build_test.sh [最大并发数]
#
# 说明:
#   - 9 个工程以独立后台任务方式并行执行：whl 编译 -> 安装 -> test 脚本执行
#   - 各工程日志输出到 parallel_logs/<工程名>.log
#   - pip install 阶段使用 flock 串行化 (pip 不支持并发安装)
#   - op_extension 模块由 3 个工程共享 (cpp_extension + kernel_extension_aclgraph
#     的 pybind/torch_library)，其「安装+测试」经 mod_op_extension.lock 串行，
#     避免某工程测试期间 .so 被另一工程 --force-reinstall 覆盖
#   - cpp_extension_full 与 kernel_extension_aclgraph 各含 2 个子工程，
#     子工程在各自任务内串行执行
# ============================================================================

set -u

EXAMPLES_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${EXAMPLES_DIR}/parallel_logs"
PIP_LOCK="${LOG_DIR}/pip.lock"
# 同名模块工程(op_extension)的「安装+测试」互斥锁
# cpp_extension 与 kernel_extension_aclgraph/{pybind,torch_library} 都产出
# 模块 op_extension，需串行「安装+测试」以免 --force-reinstall 互相覆盖 .so
OP_EXT_MOD_LOCK="${LOG_DIR}/mod_op_extension.lock"
SUMMARY_LOG="${LOG_DIR}/summary.log"

mkdir -p "${LOG_DIR}"
: > "${SUMMARY_LOG}"

# 并发上限，默认 9
MAX_JOBS="${1:-9}"

# Python / pip 命令，可通过环境变量覆盖
PYTHON="${PYTHON:-python3}"
PIP="${PIP:-pip3}"

# 颜色输出
COLOR_RED='\033[0;31m'
COLOR_GREEN='\033[0;32m'
COLOR_YELLOW='\033[0;33m'
COLOR_RESET='\033[0m'

# ----------------------------------------------------------------------------
# 串行化 pip install，避免并行安装同名包时产生竞态
# 参数: $1 = whl 文件 glob 模式 (相对当前目录)
# ----------------------------------------------------------------------------
pip_install_locked() {
    local whl_glob="$1"
    (
        flock 9
        ${PIP} install ${whl_glob} --no-deps --force-reinstall
    ) 9>"${PIP_LOCK}"
}

# ----------------------------------------------------------------------------
# 执行前清理: 卸载上次运行可能残留的自定义算子包
# 目的: 避免 build 失败时测试 import 到「旧 .so」造成假 PASS；保证每次从干净态开始
# ----------------------------------------------------------------------------
clean_env() {
    echo "[INFO] pre-run cleanup: uninstall leftover custom ops packages (if any)"
    local pkgs=(aclnn_extension op_extension cpp_extension_asc cpp_extension_base \
                cpp_extension_full cpp_extension_pybind cpp_extension_structured custom_ops)
    for p in "${pkgs[@]}"; do
        if ${PIP} show "$p" >/dev/null 2>&1; then
            ${PIP} uninstall -y "$p" >/dev/null 2>&1 && echo "  - removed: $p"
        fi
    done
}

# ----------------------------------------------------------------------------
# 单个测试脚本执行 (带结果打印)
# 参数: $1 = 测试脚本路径 (相对当前目录)
# ----------------------------------------------------------------------------
run_test_script() {
    local script="$1"
    if [ ! -f "$script" ]; then
        echo "[WARN]: test script not found: $script"
        return 1
    fi
    echo "[INFO]: run test: $script"
    ${PYTHON} "$script"
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[ERROR]: test failed: $script (rc=${rc})"
    else
        echo "[INFO]: test passed: $script"
    fi
    return $rc
}

# ----------------------------------------------------------------------------
# 安装 whl + 执行测试 (单个工程内)
# 参数:
#   $1 工程目录 (绝对路径)
#   $2 whl 文件 glob 模式
#   $3 模块锁文件 (为空则不加锁；同名模块工程间传入同一文件以串行「安装+测试」)
#   $4... 测试脚本列表 (相对工程目录 test/ 下)
# 说明: mod_lock 非空时，「安装+测试」整体在锁内执行，确保一个同名工程的测试
#       完成后、下一个同名工程才开始安装，避免 .so 被 --force-reinstall 覆盖。
# ----------------------------------------------------------------------------
install_and_test() {
    local dir="$1"
    local whl_glob="$2"
    local mod_lock="$3"
    shift 3
    local tests=("$@")
    local fail=0

    if [ -n "${mod_lock}" ]; then
        (
            flock 8
            cd "${dir}/dist"
            pip_install_locked "${whl_glob}"
            cd "${dir}/test"
            for t in "${tests[@]}"; do
                run_test_script "$t" || fail=1
            done
            exit $fail
        ) 8>"${mod_lock}"
        return $?
    else
        cd "${dir}/dist"
        pip_install_locked "${whl_glob}"
        cd "${dir}/test"
        for t in "${tests[@]}"; do
            run_test_script "$t" || fail=1
        done
        return $fail
    fi
}

# ----------------------------------------------------------------------------
# 单个工程任务: 编译 whl + 安装 + 执行 test 脚本
# 参数:
#   $1 工程名
#   $2 工程目录 (绝对路径)
#   $3 pre_build 命令 (可为空字符串)
#   $4 build 命令
#   $5 install whl glob
#   $6 模块锁文件 (可为空字符串；同名模块工程间传入同一文件)
#   $7... 测试脚本列表 (相对工程目录 test/ 下，可空)
# ----------------------------------------------------------------------------
run_project() {
    local name="$1"
    local dir="$2"
    local pre_build="$3"
    local build_cmd="$4"
    local whl_glob="$5"
    local mod_lock="$6"
    shift 6
    local tests=("$@")

    local log="${LOG_DIR}/${name}.log"
    local start_ts=$(date +%s)

    echo "[$(date '+%H:%M:%S')] [START] ${name}" | tee -a "${SUMMARY_LOG}"

    (
        set -e
        cd "${dir}"
        rm -rf dist build *.egg-info
        echo "=========== [${name}] workdir: $(pwd) ==========="

        # 1. 预处理 (如 gen.sh)
        if [ -n "${pre_build}" ]; then
            echo "----------- [${name}] pre-build: ${pre_build} -----------"
            eval "${pre_build}"
        fi

        # 2. 编译 whl
        echo "----------- [${name}] build: ${build_cmd} -----------"
        eval "${build_cmd}"

        # 3. 安装 whl + 4. 执行测试 (同名模块经 mod_lock 串行)
        echo "----------- [${name}] install: ${whl_glob} -----------"
        install_and_test "${dir}" "${whl_glob}" "${mod_lock}" "${tests[@]}"
        exit $?
    ) > "${log}" 2>&1

    local rc=$?
    local end_ts=$(date +%s)
    local duration=$((end_ts - start_ts))

    if [ $rc -eq 0 ]; then
        echo -e "[$(date '+%H:%M:%S')] [${COLOR_GREEN}PASS${COLOR_RESET}] ${name} (${duration}s)" | tee -a "${SUMMARY_LOG}"
    else
        echo -e "[$(date '+%H:%M:%S')] [${COLOR_RED}FAIL${COLOR_RESET}] ${name} (${duration}s) -> ${log}" | tee -a "${SUMMARY_LOG}"
    fi
    return $rc
}

# ----------------------------------------------------------------------------
# 9 个工程定义
# ----------------------------------------------------------------------------
# 注意:
#   - cpp_extension_full: 包含 module 与 torch_lib_impl 两个子工程 (同名包 cpp_extension_full)
#   - kernel_extension_aclgraph: 包含 pybind 与 torch_library 两个子工程 (同名包 op_extension)
#     这两个工程的子工程在其任务内串行执行，避免同名包冲突

# 子工程辅助函数: 在一个后台任务内串行执行多个子工程
# 参数: name dir pre_build build_cmd whl_glob mod_lock test1 test2 ...
run_subproject() {
    local name="$1"
    local dir="$2"
    local pre_build="$3"
    local build_cmd="$4"
    local whl_glob="$5"
    local mod_lock="$6"
    shift 6
    local tests=("$@")

    (
        set -e
        cd "${dir}"
        rm -rf dist build *.egg-info
        echo "=========== [${name}] workdir: $(pwd) ==========="

        if [ -n "${pre_build}" ]; then
            echo "----------- [${name}] pre-build: ${pre_build} -----------"
            eval "${pre_build}"
        fi

        echo "----------- [${name}] build: ${build_cmd} -----------"
        eval "${build_cmd}"

        echo "----------- [${name}] install: ${whl_glob} -----------"
        install_and_test "${dir}" "${whl_glob}" "${mod_lock}" "${tests[@]}"
    )
    return $?
}

# 工程任务包装器: 把整个工程任务写到独立日志
task_aclnn_extension() {
    run_project "aclnn_extension" \
        "${EXAMPLES_DIR}/aclnn_extension" \
        "bash gen.sh npu_custom.yaml" \
        "${PYTHON} setup.py bdist_wheel" \
        "aclnn_extension*.whl" \
        "" \
        "test_npu_fast_gelu_custom.py"
}

task_cpp_extension() {
    run_project "cpp_extension" \
        "${EXAMPLES_DIR}/cpp_extension" \
        "" \
        "${PYTHON} setup.py bdist_wheel" \
        "op_extension*.whl" \
        "${OP_EXT_MOD_LOCK}" \
        "test.py"
}

task_cpp_extension_asc() {
    # add/trig_aclgraph_test.py 为图相关用例 (依赖 npugraph_ex backend)，
    # 当前环境不支持，不看护；本工程仅看护「编译 whl + 安装」
    run_project "cpp_extension_asc" \
        "${EXAMPLES_DIR}/cpp_extension_asc" \
        "" \
        "${PYTHON} setup.py bdist_wheel" \
        "cpp_extension_asc*.whl" \
        ""
}

task_cpp_extension_base() {
    run_project "cpp_extension_base" \
        "${EXAMPLES_DIR}/cpp_extension_base" \
        "" \
        "${PYTHON} setup.py build bdist_wheel" \
        "*.whl" \
        "" \
        "test_add_custom.py"
}

task_cpp_extension_full() {
    # module 与 torch_lib_impl 共享包名 cpp_extension_full，串行执行
    # test_add_custom_graph.py 为图相关用例 (依赖 npugraph_ex backend)，不看护
    local log="${LOG_DIR}/cpp_extension_full.log"
    local start_ts=$(date +%s)
    echo "[$(date '+%H:%M:%S')] [START] cpp_extension_full" | tee -a "${SUMMARY_LOG}"
    (
        set +e
        run_subproject "cpp_extension_full/module" \
            "${EXAMPLES_DIR}/cpp_extension_full/module" \
            "" \
            "${PYTHON} setup.py build bdist_wheel" \
            "*.whl" \
            "" \
            "test_add_custom.py"
        local rc1=$?

        run_subproject "cpp_extension_full/torch_lib_impl" \
            "${EXAMPLES_DIR}/cpp_extension_full/torch_lib_impl" \
            "" \
            "${PYTHON} setup.py build bdist_wheel" \
            "*.whl" \
            "" \
            "test_add_custom.py"
        local rc2=$?

        [ $rc1 -eq 0 ] && [ $rc2 -eq 0 ]
    ) > "${log}" 2>&1
    local rc=$?
    local end_ts=$(date +%s)
    local duration=$((end_ts - start_ts))
    if [ $rc -eq 0 ]; then
        echo -e "[$(date '+%H:%M:%S')] [${COLOR_GREEN}PASS${COLOR_RESET}] cpp_extension_full (${duration}s)" | tee -a "${SUMMARY_LOG}"
    else
        echo -e "[$(date '+%H:%M:%S')] [${COLOR_RED}FAIL${COLOR_RESET}] cpp_extension_full (${duration}s) -> ${log}" | tee -a "${SUMMARY_LOG}"
    fi
    return $rc
}

task_cpp_extension_pybind() {
    run_project "cpp_extension_pybind" \
        "${EXAMPLES_DIR}/cpp_extension_pybind" \
        "" \
        "${PYTHON} setup.py build bdist_wheel" \
        "*.whl" \
        "" \
        "test_add_custom.py"
}

task_cpp_extension_structured() {
    run_project "cpp_extension_structured" \
        "${EXAMPLES_DIR}/cpp_extension_structured" \
        "bash gen.sh npu_custom.yaml" \
        "${PYTHON} setup.py bdist_wheel" \
        "cpp_extension_structured*.whl" \
        "" \
        "test_npu_fast_gelu_custom.py"
}

task_framwork_cpp_extension() {
    # 注意: test_add_custom_graph.py 依赖 torchair (import torchair /
    # torchair.ge.custom_op / torchair.get_npu_backend)，当前环境未安装 torchair，
    # 故该用例不看护、脚本中不执行；本工程仅看护「编译 whl + 安装」+ test_add_custom.py
    run_project "framwork_cpp_extension" \
        "${EXAMPLES_DIR}/framwork_cpp_extension" \
        "" \
        "${PYTHON} setup.py build bdist_wheel" \
        "*.whl" \
        "" \
        "test_add_custom.py"
}

task_kernel_extension_aclgraph() {
    # pybind 与 torch_library 共享包名 op_extension，串行执行
    local log="${LOG_DIR}/kernel_extension_aclgraph.log"
    local start_ts=$(date +%s)
    echo "[$(date '+%H:%M:%S')] [START] kernel_extension_aclgraph" | tee -a "${SUMMARY_LOG}"
    (
        set +e
        # pybind/torch_library 的 add/trig_aclgraph_test.py 为图相关用例 (依赖 npugraph_ex)，
        # 当前环境不支持，不看护；本工程仅看护「编译 whl + 安装」
        run_subproject "kernel_extension_aclgraph/pybind" \
            "${EXAMPLES_DIR}/kernel_extension_aclgraph/pybind" \
            "" \
            "${PYTHON} setup.py bdist_wheel" \
            "op_extension*.whl" \
            "${OP_EXT_MOD_LOCK}"
        local rc1=$?

        run_subproject "kernel_extension_aclgraph/torch_library" \
            "${EXAMPLES_DIR}/kernel_extension_aclgraph/torch_library" \
            "" \
            "${PYTHON} setup.py bdist_wheel" \
            "op_extension*.whl" \
            "${OP_EXT_MOD_LOCK}"
        local rc2=$?

        [ $rc1 -eq 0 ] && [ $rc2 -eq 0 ]
    ) > "${log}" 2>&1
    local rc=$?
    local end_ts=$(date +%s)
    local duration=$((end_ts - start_ts))
    if [ $rc -eq 0 ]; then
        echo -e "[$(date '+%H:%M:%S')] [${COLOR_GREEN}PASS${COLOR_RESET}] kernel_extension_aclgraph (${duration}s)" | tee -a "${SUMMARY_LOG}"
    else
        echo -e "[$(date '+%H:%M:%S')] [${COLOR_RED}FAIL${COLOR_RESET}] kernel_extension_aclgraph (${duration}s) -> ${log}" | tee -a "${SUMMARY_LOG}"
    fi
    return $rc
}

# ----------------------------------------------------------------------------
# 调度器: 限制并发数
# ----------------------------------------------------------------------------
declare -a PIDS=()
declare -a NAMES=()

schedule() {
    local name="$1"
    local fn="$2"
    # 等待空位
    while [ $(jobs -r | wc -l) -ge ${MAX_JOBS} ]; do
        sleep 1
    done
    echo "[$(date '+%H:%M:%S')] [LAUNCH] ${name}" | tee -a "${SUMMARY_LOG}"
    ${fn} &
    PIDS+=($!)
    NAMES+=("${name}")
}

# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
TOTAL_START_TS=$(date +%s)

echo "========================================================================" | tee -a "${SUMMARY_LOG}"
echo "Parallel build + install + test for 9 example projects" | tee -a "${SUMMARY_LOG}"
echo "Examples dir : ${EXAMPLES_DIR}" | tee -a "${SUMMARY_LOG}"
echo "Logs dir     : ${LOG_DIR}" | tee -a "${SUMMARY_LOG}"
echo "Max jobs     : ${MAX_JOBS}" | tee -a "${SUMMARY_LOG}"
echo "Python       : ${PYTHON}" | tee -a "${SUMMARY_LOG}"
echo "Pip          : ${PIP}" | tee -a "${SUMMARY_LOG}"
echo "Start time   : $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${SUMMARY_LOG}"
echo "========================================================================" | tee -a "${SUMMARY_LOG}"

# 执行前清理环境，避免上次残留的自定义包导致假 PASS
clean_env

# 9 个工程并行调度
schedule "aclnn_extension"         task_aclnn_extension
schedule "cpp_extension"           task_cpp_extension
schedule "cpp_extension_asc"       task_cpp_extension_asc
schedule "cpp_extension_base"      task_cpp_extension_base
schedule "cpp_extension_full"      task_cpp_extension_full
schedule "cpp_extension_pybind"    task_cpp_extension_pybind
# schedule "cpp_extension_structured" task_cpp_extension_structured
schedule "framwork_cpp_extension" task_framwork_cpp_extension
schedule "kernel_extension_aclgraph" task_kernel_extension_aclgraph

# ----------------------------------------------------------------------------
# 等待所有任务完成并汇总
# ----------------------------------------------------------------------------
OVERALL_RC=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        OVERALL_RC=1
    fi
done

TOTAL_END_TS=$(date +%s)
TOTAL_DURATION=$((TOTAL_END_TS - TOTAL_START_TS))
TOTAL_MIN=$((TOTAL_DURATION / 60))
TOTAL_SEC=$((TOTAL_DURATION % 60))

echo "========================================================================" | tee -a "${SUMMARY_LOG}"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${SUMMARY_LOG}"
echo -e "Total duration: ${COLOR_YELLOW}${TOTAL_MIN}m ${TOTAL_SEC}s${COLOR_RESET} (wall clock)" | tee -a "${SUMMARY_LOG}"
echo "------------------------------------------------------------------------" | tee -a "${SUMMARY_LOG}"
echo "Per-project duration (sorted by start order):" | tee -a "${SUMMARY_LOG}"
echo "------------------------------------------------------------------------" | tee -a "${SUMMARY_LOG}"
# 从 summary.log 抽出每个任务的 PASS/FAIL 行（含耗时）—— 先剥离 ANSI 色码再匹配
# 注意: 只输出到 stdout，不再 tee 回 summary.log，否则会让计数 grep 翻倍
sed 's/\x1b\[[0-9;]*m//g' "${SUMMARY_LOG}" | grep -E "\[(PASS|FAIL)\]"
echo "------------------------------------------------------------------------" | tee -a "${SUMMARY_LOG}"
# 最长任务耗时（取自各任务 duration）
MAX_DUR=$(grep -oE "\([0-9]+s\)" "${SUMMARY_LOG}" | tr -d '()s' | sort -n | tail -1)
if [ -n "${MAX_DUR}" ]; then
    MAX_MIN=$((MAX_DUR / 60))
    MAX_SEC=$((MAX_DUR % 60))
    echo "Longest single task: ${MAX_MIN}m ${MAX_SEC}s" | tee -a "${SUMMARY_LOG}"
fi
PASS_CNT=$(sed 's/\x1b\[[0-9;]*m//g' "${SUMMARY_LOG}" | grep -cE "\[PASS\]" || true)
FAIL_CNT=$(sed 's/\x1b\[[0-9;]*m//g' "${SUMMARY_LOG}" | grep -cE "\[FAIL\]" || true)
echo "Result: ${PASS_CNT} passed, ${FAIL_CNT} failed" | tee -a "${SUMMARY_LOG}"
echo "========================================================================" | tee -a "${SUMMARY_LOG}"

if [ ${OVERALL_RC} -eq 0 ]; then
    echo -e "${COLOR_GREEN}[ALL PASS] All scheduled projects passed.${COLOR_RESET}" | tee -a "${SUMMARY_LOG}"
else
    echo -e "${COLOR_RED}[SOME FAIL] See ${SUMMARY_LOG} and per-project logs in ${LOG_DIR}/${COLOR_RESET}" | tee -a "${SUMMARY_LOG}"
fi
echo "Per-project logs:"
ls -1 "${LOG_DIR}"/*.log 2>/dev/null | sed 's/^/  /'

exit ${OVERALL_RC}

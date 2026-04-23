#!/bin/bash

BASE_DIR=$(pwd)

# 编译wheel包
python3 setup.py build bdist_wheel

# 安装wheel包
cd ${BASE_DIR}/dist
pip3 install *.whl --no-deps --force-reinstall


# 运行测试用例
cd ${BASE_DIR}/test
python3 test_add_custom.py
if [ $? -ne 0 ]; then
    echo "[ERROR]: Run add_custom test failed!"
else
    echo "[INFO]: Run add_custom test success!"
fi
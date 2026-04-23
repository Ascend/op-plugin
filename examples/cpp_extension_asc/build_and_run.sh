#!/bin/bash

BASE_DIR=$(pwd)

# 编译wheel包
python setup.py bdist_wheel

# 安装wheel包
cd ${BASE_DIR}/dist
pip3 install *.whl


# 运行测试用例
cd ${BASE_DIR}/test
python add_aclgraph_test.py
if [ $? -ne 0 ]; then
    echo "[ERROR]: Run add_aclgraph_test test failed!"
else
    echo "[INFO]: Run add_aclgraph_test test success!"
fi

python trig_aclgraph_test.py
if [ $? -ne 0 ]; then
    echo "[ERROR]: Run trig_aclgraph_test test failed!"
else
    echo "[INFO]: Run trig_aclgraph_test test success!"
fi
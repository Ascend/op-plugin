#!/bin/bash

BASE_DIR=$(pwd)


# 1. 先执行 gen.sh，生成适配代码
bash gen.sh npu_custom.yaml
echo "[INFO]: bash gen.sh success!"

# 2. 再执行 setup.py 构建 whl 包并安装
python3 setup.py bdist_wheel
cd ${BASE_DIR}/dist
pip install cpp_extension_structured*.whl

# 3. 运行测试验证
cd ${BASE_DIR}/test
python3 test_npu_fast_gelu_custom.py
if [ $? -ne 0 ]; then
    echo "[ERROR]: Run add_custom test failed!"
else
    echo "[INFO]: Run add_custom test success!"
fi


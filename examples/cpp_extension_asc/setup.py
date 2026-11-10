import os

from setuptools import find_packages, setup
from torch_npu.utils.cpp_extension import BuildExtension, NpuExtension


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

setup(
    name="cpp_extension_asc",
    version="0.1",
    ext_modules=[
        NpuExtension(
            name="cpp_extension_asc.custom_ops_lib",
            sources=[
                os.path.join(BASE_DIR, "csrc", "add_custom.asc"),
                os.path.join(BASE_DIR, "csrc", "trig_inplace_custom.asc"),
            ],
        ),
    ],
    packages=find_packages(),
    cmdclass={"build_ext": BuildExtension},
)

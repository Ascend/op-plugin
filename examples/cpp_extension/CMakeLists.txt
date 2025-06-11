cmake_minimum_required(VERSION 3.16.0)
project(Ascend_C)
set(LINUX TRUE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
message(STATUS "Determined _GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI}")
set(_GLIBCXX_USE_CXX11_ABI ${GLIBCXX_USE_CXX11_ABI})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI}")


# user-defined configuration
set(SOC_VERSION "Ascendxxxyy" CACHE STRING "system on chip type")
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "path for install()" FORCE)
if(DEFINED ASCEND_CANN_PACKAGE_PATH)
elseif(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "ASCEND CANN package installation directory" FORCE)
endif()


if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the cann package is installed.")
endif()

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# ascendc_library use to add kernel file to generate ascendc library

FILE(GLOB _OPPLUGIN_PYBIND_OPS_SRCS
   csrc/host/*.cpp
   csrc/host/tiling/*.cpp)

set(_OPPLUGIN_PYBIND_NAME op_extension)

ascendc_library(no_workspace_kernel STATIC
    csrc/kernel/add_custom.cpp
)

ascendc_library(workspace_kernel STATIC
    csrc/kernel/matmul_leakyrelu_custom.cpp
)

ascendc_compile_definitions(workspace_kernel PRIVATE
  -DHAVE_WORKSPACE
  -DHAVE_TILING
)

add_library(${_OPPLUGIN_PYBIND_NAME} SHARED ${_OPPLUGIN_PYBIND_OPS_SRCS})

target_link_libraries(${_OPPLUGIN_PYBIND_NAME} PRIVATE
  workspace_kernel
  no_workspace_kernel
  torch_npu
  ascendcl
  tiling_api
  register
  platform
  ascendalog
  dl
)

target_link_directories(${_OPPLUGIN_PYBIND_NAME} PRIVATE
  ${TORCH_PATH}/lib
  ${TORCH_NPU_PATH}/lib
)
target_include_directories(${_OPPLUGIN_PYBIND_NAME} PRIVATE
  ${TORCH_NPU_PATH}/include
  ${TORCH_PATH}/include
  ${TORCH_PATH}/include/torch/csrc/api/include
)

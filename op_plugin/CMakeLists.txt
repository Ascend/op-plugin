if (NOT DEFINED TORCH_VERSION)
    message(FATAL_ERROR "torch version is not defined.")
endif()

FILE(GLOB _OPS_PLUGIN_SRCS
    ops/custom/*.cpp
    ops/official/*.cpp)

FILE(GLOB _OPPLUGIN_BASE_OPS_SRCS
    ops/base_ops/aclops/*.cpp
    ops/base_ops/opapi/*.cpp)

FILE(GLOB _OPPLUGIN_UTILS_SRCS
    *.cpp
    utils/*.cpp
    utils/custom_functions/aclops/*.cpp
    utils/custom_functions/opapi/*.cpp)

LIST(APPEND OPS_PLUGIN_SRCS ${_OPS_PLUGIN_SRCS} ${_OPPLUGIN_BASE_OPS_SRCS} ${_OPPLUGIN_UTILS_SRCS})

string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)" MATCHED_TORCH_VERSION "${TORCH_VERSION}")
set(OPS_VERSION_NAME "v${CMAKE_MATCH_1}r${CMAKE_MATCH_2}")

FILE(GLOB _OPPLUGIN_VERSION_SRCS
    ops/${OPS_VERSION_NAME}/aclops/*.cpp
    ops/${OPS_VERSION_NAME}/opapi/*.cpp
    ops/${OPS_VERSION_NAME}/opapi/sparse/*.cpp
    ops/${OPS_VERSION_NAME}/*.cpp)
LIST(APPEND OPS_PLUGIN_SRCS ${_OPPLUGIN_VERSION_SRCS})

# Pass to parents
set(OPS_PLUGIN_SRCS ${OPS_PLUGIN_SRCS} PARENT_SCOPE)

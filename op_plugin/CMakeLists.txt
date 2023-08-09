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
    utils/*.cpp
    utils/custom_functions/aclops/*.cpp
    utils/custom_functions/opapi/*.cpp)

LIST(APPEND OPS_PLUGIN_SRCS ${_OPS_PLUGIN_SRCS} ${_OPPLUGIN_BASE_OPS_SRCS} ${_OPPLUGIN_UTILS_SRCS})

if (("${TORCH_VERSION}" VERSION_GREATER_EQUAL "1.11") AND ("${TORCH_VERSION}" VERSION_LESS "1.12"))
    set(OPS_VERSION_NAME "v1r11")
elseif(("${TORCH_VERSION}" VERSION_GREATER_EQUAL "2.0") AND ("${TORCH_VERSION}" VERSION_LESS "2.1"))
    set(OPS_VERSION_NAME "v2r0")
elseif("${TORCH_VERSION}" VERSION_GREATER_EQUAL "2.1")
    set(OPS_VERSION_NAME "v2r1")
else()
    message(FATAL_ERROR "unsupported torch version: ${TORCH_VERSION}.")
endif()

FILE(GLOB _OPPLUGIN_VERSION_SRCS
    ops/${OPS_VERSION_NAME}/aclops/*.cpp
    ops/${OPS_VERSION_NAME}/opapi/*.cpp)
LIST(APPEND OPS_PLUGIN_SRCS ${_OPPLUGIN_VERSION_SRCS})

# Pass to parents
set(OPS_PLUGIN_SRCS ${OPS_PLUGIN_SRCS} PARENT_SCOPE)

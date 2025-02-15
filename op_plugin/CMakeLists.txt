if (NOT DEFINED TORCH_VERSION)
    message(FATAL_ERROR "torch version is not defined.")
endif()

FILE(GLOB _OPS_PLUGIN_SRCS
    ops/custom/*.cpp
    ops/official/*.cpp)

FILE(GLOB _OPPLUGIN_BASE_OPS_SRCS
    ops/aclops/*.cpp
    ops/opapi/*.cpp)

FILE(GLOB _OPPLUGIN_UTILS_SRCS
    *.cpp
    utils/*.cpp
    utils/custom_functions/aclops/*.cpp
    utils/custom_functions/opapi/*.cpp)

LIST(APPEND OPS_PLUGIN_SRCS ${_OPS_PLUGIN_SRCS} ${_OPPLUGIN_BASE_OPS_SRCS} ${_OPPLUGIN_UTILS_SRCS})

string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)" MATCHED_TORCH_VERSION "${TORCH_VERSION}")
set(OPS_VERSION_NAME "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}")

if (${OPS_VERSION_NAME} VERSION_GREATER_EQUAL "2.1")
    FILE(GLOB _OPPLUGIN_SPARSE_SRCS
        ops/opapi/sparse/*.cpp)
    LIST(APPEND OPS_PLUGIN_SRCS ${_OPPLUGIN_SPARSE_SRCS})
endif()

# Pass to parents
set(OPS_PLUGIN_SRCS ${OPS_PLUGIN_SRCS} PARENT_SCOPE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCURRENT_VERSION=V${CMAKE_MATCH_1}R${CMAKE_MATCH_2}" PARENT_SCOPE)

set(_OPPLUGIN_ATB_NAME op_plugin_atb)

FILE(GLOB _OPPLUGIN_UTILS_SRCS
    ops/atb/*.cpp
    utils/custom_functions/atb/*.cpp)

add_library(${_OPPLUGIN_ATB_NAME} SHARED
    ${_OPPLUGIN_UTILS_SRCS}
)

target_link_libraries(${_OPPLUGIN_ATB_NAME} PUBLIC torch_npu)
target_link_libraries(${_OPPLUGIN_ATB_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/third_party/op-plugin/op_plugin/third_party/atb/libs/libatb.so)

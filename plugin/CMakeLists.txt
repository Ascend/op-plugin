FILE(GLOB _OPS_PLUGIN_SRCS
    ops/kernel/*.cpp
    framework/utils/*.cpp)

LIST(APPEND OPS_PLUGIN_SRCS ${_OPS_PLUGIN_SRCS})

# Pass to parent
set(OPS_PLUGIN_SRCS ${OPS_PLUGIN_SRCS} PARENT_SCOPE)
FILE(GLOB _OPS_PLUGIN_SRCS
    ops/custom/*.cpp
    ops/official/*.cpp
    utils/*.cpp)

LIST(APPEND OPS_PLUGIN_SRCS ${_OPS_PLUGIN_SRCS})

# Pass to parent
set(OPS_PLUGIN_SRCS ${OPS_PLUGIN_SRCS} PARENT_SCOPE)
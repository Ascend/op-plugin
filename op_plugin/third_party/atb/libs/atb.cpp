// Implementation of the compiled staking function, in order to compile libatb.so
#include "context.h"
#include "operation.h"

namespace atb {
Status CreateContext(Context **context)
{
    return 0;
}

Status DestroyContext(Context *context)
{
    return 0;
}

template <typename OpParam> Status CreateOperation(const OpParam &opParam, Operation **operation)
{
    return 0;
}

Status DestroyOperation(Operation *operation)
{
    return 0;
}
} // namespace atb

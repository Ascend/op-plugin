// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
    using npu_preparation = at_npu::native::OpPreparation;

    at::Tensor attention_worker_scheduler(const at::Tensor & self)
    {
        auto output_size_0 = self.sizes();
        auto output_dtype_0 = self.scalar_type();
        at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0, self.options().dtype(output_dtype_0));

        out.copy_(self);
        EXEC_NPU_CMD(aclnnInplaceAttentionWorkerScheduler, out);
        return out;
    }
}
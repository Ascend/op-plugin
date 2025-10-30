// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
    using npu_preparation = at_npu::native::OpPreparation;

    at::Tensor ffn_worker_scheduler(const at::Tensor & self, int64_t sync_group_size, int64_t execute_mode)
    {
        auto output_size_0 = self.sizes();
        auto output_dtype_0 = self.scalar_type();
        at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0, self.options().dtype(output_dtype_0));

        out.copy_(self);
        EXEC_NPU_CMD(aclnnInplaceFfnWorkerScheduler, out, sync_group_size, execute_mode);
        return out;
    }
}
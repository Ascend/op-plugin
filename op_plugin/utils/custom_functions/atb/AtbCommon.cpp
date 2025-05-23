// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "AtbCommon.h"

namespace atb {
atb::Tensor AtTensor2AtbTensor(const at::Tensor atTensor)
{
    static std::map<at::ScalarType, aclDataType> dtypeMap = {
        {at::ScalarType::Bool, ACL_BOOL},   {at::ScalarType::Byte, ACL_UINT8},
        {at::ScalarType::Char, ACL_INT8},   {at::ScalarType::Half, ACL_FLOAT16},
        {at::ScalarType::Float, ACL_FLOAT}, {at::ScalarType::Int, ACL_INT32},
        {at::ScalarType::Long, ACL_INT64},  {at::ScalarType::BFloat16, ACL_BF16},
        {at::ScalarType::Double, ACL_DOUBLE}, {at::ScalarType::Short, ACL_INT16},
        {at::ScalarType::ComplexHalf, ACL_COMPLEX32}, {at::ScalarType::ComplexFloat, ACL_COMPLEX64},
        {at::ScalarType::ComplexDouble, ACL_COMPLEX128},
    };

    TORCH_CHECK(atTensor.is_contiguous(), "atTensor is not contiguous");
    atb::Tensor tensor;
    tensor.desc.format = ACL_FORMAT_ND;
    if (atTensor.device().type() == at::kCPU) {
        tensor.hostData = atTensor.data_ptr();
    } else {
        tensor.deviceData = atTensor.data_ptr();
    }

    tensor.desc.shape.dimNum = atTensor.sizes().size();
    for (uint64_t i = 0; i < atTensor.sizes().size(); i++) {
        tensor.desc.shape.dims[i] = atTensor.sizes()[i];
    }

    auto dtypeIterator = dtypeMap.find(atTensor.scalar_type());
    TORCH_CHECK(dtypeIterator != dtypeMap.end(), "not support dtype: ", atTensor.scalar_type());
    tensor.desc.dtype = dtypeIterator->second;

    tensor.dataSize = atb::Utils::GetTensorSize(tensor);

    return tensor;
}


void RunAtbCmd(atb::Operation *op, const ParamSetter &paramsetter, const std::string &name)
{
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    auto acl_call = [=]() -> int {
        auto contextPtr = GetContext(stream);
        uint64_t workspaceSize = OperationSetup(paramsetter.variantPack, op, contextPtr);
        at::Tensor workspaceTensor;
        void *workspacePtr = nullptr;
        if (workspaceSize != 0) {
            workspaceTensor = GetWorkspaceTensor(workspaceSize, stream);
            workspacePtr = const_cast<void *>(workspaceTensor.storage().data());
        }
        auto st = op->Execute(paramsetter.variantPack, (uint8_t *)workspacePtr, workspaceSize, contextPtr);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2(name, acl_call);
}


ParamSetter& ParamSetter::Input(const at::Tensor &tensor)
{
    if (!tensor.defined()) {
        variantPack.inTensors.push_back(atb::Tensor());
        return *this;
    }
    at::Tensor newTensor = tensor;
    if (torch_npu::utils::is_npu(newTensor)) {
        newTensor = atb::utils::FormatTrans(tensor);
    }

    if (!newTensor.is_contiguous()) {
        newTensor = newTensor.contiguous();
    }
    auto AtTensor = AtTensor2AtbTensor(newTensor);

    variantPack.inTensors.push_back(AtTensor);
    return *this;
}


ParamSetter& ParamSetter::Input(const c10::optional<at::Tensor> &tensor)
{
    if (!tensor.has_value()) {
        variantPack.inTensors.push_back(atb::Tensor());
        return *this;
    }
    return Input(tensor.value());
}


ParamSetter& ParamSetter::Output(at::Tensor &output)
{
    auto AtTensor = AtTensor2AtbTensor(output);
    variantPack.outTensors.push_back(AtTensor);
    return *this;
}


uint64_t OperationSetup(atb::VariantPack variantPack, atb::Operation *operation, atb::Context* contextPtr)
{
    uint64_t workspaceSize = 0;
    atb::Status status = operation->Setup(variantPack, workspaceSize, contextPtr);
    TORCH_CHECK(status == 0, "setup failed!");
    return workspaceSize;
}


at::Tensor GetWorkspaceTensor(uint64_t workspaceSize, aclrtStream stream)
{
    at::Tensor workspaceTensor = at_npu::native::allocate_workspace(workspaceSize, stream);
    return workspaceTensor;
}


ContextManager& ContextManager::GetInstance()
{
    static ContextManager instance;
    return instance;
}


ContextManager::ContextManager() : atbContext(nullptr) {}


ContextManager::~ContextManager()
{
    if (atbContext) {
        auto status = atb::DestroyContext(atbContext);
        TORCH_CHECK(status == 0, "destroy context failed!");
        atbContext = nullptr;
    }
}


atb::Context* ContextManager::GetContext(aclrtStream stream)
{
    std::call_once(createFlag, [this]() {
        auto status = atb::CreateContext(&atbContext);
        TORCH_CHECK(status == 0, "create context failed!");
    });

    atbContext->SetExecuteStream(stream);
    return atbContext;
}


atb::Context* GetContext(aclrtStream stream)
{
    return ContextManager::GetInstance().GetContext(stream);
}

} // namespace atb
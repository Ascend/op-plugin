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

#ifndef INC_EXTERNAL_GRAPH_INFERENCE_CONTEXT_H_
#define INC_EXTERNAL_GRAPH_INFERENCE_CONTEXT_H_

#include <memory>
#include <string>
#include <vector>

#include "./tensor.h"
#include "./types.h"
#include "ascend_string.h"

namespace ge {
class InferenceContext;
using InferenceContextPtr = std::shared_ptr<InferenceContext>;

class ShapeAndTypeImpl;
class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ShapeAndType {
public:
    ShapeAndType();
    ~ShapeAndType() = default;

    ShapeAndType(const Shape &shape, DataType dataType);

    void SetShape(const Shape &shape);

    void SetType(DataType dataType);

    Shape GetShape() const;

    DataType GetDataType() const;

private:
    std::shared_ptr<ShapeAndTypeImpl> shape_and_type_impl_;
};

class InferenceContextImpl;
class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY InferenceContext {
public:
    ~InferenceContext() = default;
    InferenceContext(const InferenceContext &context) = delete;
    InferenceContext(const InferenceContext &&context) = delete;
    InferenceContext &operator=(const InferenceContext &context) = delete;
    InferenceContext &operator=(const InferenceContext &&context) = delete;

    void SetInputHandleShapesAndTypes(std::vector<std::vector<ShapeAndType>> &&shapes_and_types);
    const std::vector<std::vector<ShapeAndType>> &GetInputHandleShapesAndTypes() const;
    const std::vector<std::vector<ShapeAndType>> &GetOutputHandleShapesAndTypes() const;
    void SetOutputHandleShapesAndTypes(const std::vector<std::vector<ShapeAndType>> &shapes_and_types);
    void SetOutputHandleShapesAndTypes(std::vector<std::vector<ShapeAndType>> &&shapes_and_types);

    ATTRIBUTED_DEPRECATED(void SetMarks(const std::vector<AscendString> &))
    void SetMarks(const std::vector<std::string> &marks);
    void SetMarks(const std::vector<AscendString> &marks);

    ATTRIBUTED_DEPRECATED(void GetMarks(std::vector<AscendString> &) const)
    const std::vector<std::string> &GetMarks() const;
    void GetMarks(std::vector<AscendString> &marks) const;

    static std::unique_ptr<InferenceContext> Create();

private:
    explicit InferenceContext(std::unique_ptr<InferenceContextImpl> &impl);
    std::shared_ptr<InferenceContextImpl> inference_context_impl_;
};
}  // namespace ge
#endif  // INC_EXTERNAL_GRAPH_INFERENCE_CONTEXT_H_

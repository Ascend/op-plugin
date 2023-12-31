# Copyright (c) 2023 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION. 
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Sequence, Union

from codegen import local
from codegen.api import cpp
from codegen.api.types import (
    ArgName,
    BaseCType,
    Binding,
    boolT,
    ConstRefCType,
    CType,
    deviceT,
    layoutT,
    ListCType,
    MutRefCType,
    NamedCType,
    OptionalCType,
    scalarT,
    scalarTypeT,
    tensorT,
)
from codegen.model import (
    Argument,
    FunctionSchema,
    Return,
    SelfArgument,
    TensorOptionsArguments,
    Type,
)
from codegen.utils import assert_never

# This file describes the translation of JIT schema to the native functions API.
# This looks a lot like the C++ API (which makes historical sense, because the
# idea was you wrote native functions to implement functions in the C++ API),
# but over time we have evolved the C++ API without actually changing our
# native:: kernels.  The intention is to make native API and dispatcher API
# line up as closely as possible, since this results in the least overhead
# (no translation is needed from dispatcher API to native API).
#
# NB: this is symint aware, you will get the non-SymInt variant for some
# dispatch entries and SymInt for others.


def name(func: FunctionSchema) -> str:
    func_name = str(func.name.name)
    if func.is_out_fn():
        func_name += "_out"
    if func.name.overload_name:
        func_name += f"_{func.name.overload_name}"
    return func_name


def argumenttype_type(
    t: Type, *, mutable: bool, binds: ArgName, symint: bool
) -> NamedCType:
    if str(t) == "Tensor?":
        tensor_type: OptionalCType = OptionalCType(BaseCType(tensorT))
        if mutable and not local.use_const_ref_for_mutable_tensors():
            return NamedCType(binds, MutRefCType(tensor_type))
        else:
            return NamedCType(binds, ConstRefCType(tensor_type))
    elif str(t) == "Tensor?[]":
        return NamedCType(
            binds, ConstRefCType(ListCType(OptionalCType(BaseCType(tensorT))))
        )
    elif str(t) == "Scalar":
        return NamedCType(binds, ConstRefCType(BaseCType(scalarT)))
    elif str(t) == "Scalar?":
        return NamedCType(binds, ConstRefCType(OptionalCType(BaseCType(scalarT))))
    return cpp.argumenttype_type(t, mutable=mutable, binds=binds, symint=symint)


def returns_type(rs: Sequence[Return], *, symint: bool) -> CType:
    return cpp.returns_type(rs, symint=symint)


def argument_type(a: Argument, *, binds: ArgName, symint: bool) -> NamedCType:
    return argumenttype_type(a.type, mutable=a.is_write, binds=binds, symint=symint)


def argument(
    a: Union[Argument, SelfArgument, TensorOptionsArguments],
    *,
    is_out: bool,
    symint: bool,
) -> List[Binding]:
    # Ideally, we NEVER default native functions.  However, there are a number
    # of functions that call native:: directly and rely on the defaulting
    # existing.  So for BC, we generate defaults for non-out variants (but not
    # for out variants, where it is impossible to generate an appropriate
    # default)
    # print("aaaaaaa======================================================")
    # print(a)

    should_default = not is_out
    if isinstance(a, Argument):
        default: Optional[str] = None
        if should_default and a.default is not None:
            default = cpp.default_expr(a.default, a.type, symint=symint)
        return [
            Binding(
                nctype=argument_type(a, binds=a.name, symint=symint),
                name=a.name,
                default=default,
                argument=a,
            )
        ]
    elif isinstance(a, SelfArgument):
        # Erase SelfArgument from the distinction
        return argument(a.argument, is_out=is_out, symint=symint)
    elif isinstance(a, TensorOptionsArguments):
        default = None
        if should_default:
            default = "{}"

        return [
            Binding(
                nctype=NamedCType("dtype", OptionalCType(BaseCType(scalarTypeT))),
                name="dtype",
                default=default,
                argument=a,
            ),
            Binding(
                nctype=NamedCType("layout", OptionalCType(BaseCType(layoutT))),
                name="layout",
                default=default,
                argument=a,
            ),
            Binding(
                nctype=NamedCType("device", OptionalCType(BaseCType(deviceT))),
                name="device",
                default=default,
                argument=a,
            ),
            Binding(
                nctype=NamedCType("pin_memory", OptionalCType(BaseCType(boolT))),
                name="pin_memory",
                default=default,
                argument=a,
            ),
        ]
    else:
        assert_never(a)


def arguments(func: FunctionSchema, *, symint: bool) -> List[Binding]:
    args: List[Union[Argument, TensorOptionsArguments, SelfArgument]] = []
    args.extend(func.arguments.non_out)
    args.extend(func.arguments.out)
    return [
        r for arg in args for r in argument(arg, symint=symint, is_out=func.is_out_fn())
    ]

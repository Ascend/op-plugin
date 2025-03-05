# Copyright (c) 2024 Huawei Technologies Co., Ltd
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

from typing import List, Dict, Sequence
import copy
from dataclasses import dataclass

from codegen.model import (BaseTy, SchemaKind, BaseType,
                           Argument, NativeFunction, ListType)
from codegen.context import native_function_manager
from codegen.api.types import NativeSignature
from codegen.api import cpp


def filt_input_tensor(arguments: Sequence[Argument]) -> List[str]:
    input_tensors = list()
    for arg in arguments:
        if isinstance(arg.type, BaseType) and arg.type.name == BaseTy.Tensor:
            input_tensors.append(arg.name)
        if isinstance(arg.type, ListType) and arg.type.elem.name == BaseTy.Tensor:
            input_tensors.append(f'{arg.name}[0]')
    return input_tensors


@dataclass(frozen=True)
class ResInfo:
    name: str
    size: str
    dtype: str
    option: str
    infer_name: str = None

    @staticmethod
    def parse(results: Dict[str, Dict[str, str]], f: 'NativeFunction') -> List['ResInfo']:
        kind = f.func.kind()
        tensor_number = sum(map(lambda x: x.type.name ==
                            BaseTy.Tensor, f.func.returns))
        if len(results) != tensor_number and kind == SchemaKind.functional:
            raise RuntimeError(f"The number of result info in yaml is {len(results)}."
                               f"That does not match {f.func.name}'s returns tensor number {tensor_number}")

        arguments = f.func.arguments.flat_all

        input_tensors = filt_input_tensor(arguments)
        res_infos = []

        if kind == SchemaKind.out:
            result_names = cpp.return_names(f)
        else:
            result_names = [k for k in results.keys()]

        for name in result_names:
            info = results.get(name, None)
            if kind == SchemaKind.out and info is None:
                size, dtype = name, name
                infer_name = None
            else:
                size = info.pop('size', None)
                dtype = info.pop('dtype', None)
                size = name if size is None and kind == SchemaKind.out else size
                dtype = name if dtype is None and kind == SchemaKind.out else dtype
                if size is None:
                    raise RuntimeError(f"The {name}'s size  is None in {f.func.name}")
                if dtype is None:
                    raise RuntimeError(f"The {name}'s dtype  is None in {f.func.name}")
                infer_name = info.pop('name', None)

            if size in input_tensors:
                size_formula = f'{size}.sizes()'
            else:
                size_formula = size
            if dtype in input_tensors:
                dtype_formula = f'{dtype}.scalar_type()'
            else:
                dtype_formula = dtype

            res_infos.append(
                ResInfo(name=name, size=size_formula, dtype=dtype_formula,
                        option=input_tensors[0], infer_name=infer_name)
            )

        return res_infos


@dataclass(frozen=True)
class StructInfo:
    name: str
    structured_inherit: 'NativeFunction' = None
    aclnn_name: str = None
    func: 'NativeFunction' = None
    cmd_args: str = None
    return_args: str = None
    acl_op: bool = None
    results: List[ResInfo] = None
    new_params: Dict[str, str] = None

    @staticmethod
    def from_yaml(
        es: Sequence[Dict[str, object]],
        native_functions: Sequence[NativeFunction],
    ) -> "List[StructInfo]":
        '''
        Parse a StructInfo from a dictionary as directly parsed 
        from op_plugin_functions.yaml
        '''
        functions_by_schema: Dict[str, NativeFunction] = {}

        for function in native_functions:
            if str(function.func) in functions_by_schema:
                raise RuntimeError(
                    f"{function.func} has multiple definitions in op_plugin_functions.yaml")

            functions_by_schema[str(function.func)] = function

        def gen_func_name(schema: str) -> str:
            return schema.split('(')[0]

        funcname_map: Dict[str, NativeFunction] = {}
        struct_map: Dict[str, Dict] = {}
        for e in es:
            e.pop('acl_op', None)
            e.pop('op_api', None)
            schema_str = e.get('func', None)
            op_name = gen_func_name(schema_str)

            funcname_map[op_name] = functions_by_schema.get(schema_str)
            struct_map[op_name] = copy.deepcopy(e)

        struct_infos: List[StructInfo] = []
        for e in es:
            schema_str = e.pop('func', None)
            defn_name = gen_func_name(schema_str)
            schema_function = functions_by_schema.get(schema_str)

            if not schema_function:
                avail = "\n".join(
                    k for k in functions_by_schema.keys() if gen_func_name(k) == defn_name
                )
                raise RuntimeError(
                    f"could not find ATen function for schema: {schema_str} "
                    f".  Available signatures:\n{avail}"
                )

            func_kind = schema_function.func.kind()

            if 'op_api' not in schema_function.impl_ns:
                raise RuntimeError(f"The Aten function {schema_str} has no op_opi"
                                   " implement in op_plugin_functions yaml")
            acl_op = 'acl_op' in schema_function.impl_ns

            gen_opapi_info = e.get('gen_opapi')
            structured_inherit = gen_opapi_info.pop('structured_inherit', None)

            if structured_inherit is not None:
                if func_kind == SchemaKind.inplace:
                    struct_info = StructInfo(
                        name=defn_name,
                        func=schema_function,
                        structured_inherit=funcname_map.get(structured_inherit),
                        acl_op=acl_op,
                    )
                    struct_infos.append(struct_info)
                    continue
                else:
                    gen_opapi_info = struct_map.get(structured_inherit)
                    if gen_opapi_info is None:
                        raise RuntimeError(f'The structured_inherit func {structured_inherit} is None')
                    gen_opapi_info = gen_opapi_info.get('gen_opapi')

            aclnn_arguments = gen_opapi_info.pop('exec', None)

            aclnn_arguments_list = [argument.strip()
                               for argument in aclnn_arguments.split(',')]
            aclnn_name = aclnn_arguments_list[0]
            new_params_dict = gen_opapi_info.pop('new_params', dict())

            cmd_args_expand = False
            if len(aclnn_arguments_list) == 1:
                cmd_args_expand = True
                with native_function_manager(schema_function):
                    sig = NativeSignature(
                        schema_function.func, prefix='', symint=False)
                    for a in sig.arguments():
                        aclnn_arguments_list.append(a.name)

            aclnn_arguments = ', '.join(aclnn_arguments_list)

            if func_kind == SchemaKind.out:
                output_names = cpp.return_names(schema_function)
                for key in gen_opapi_info.keys():
                    if key not in output_names:
                        raise ValueError(
                            f"Result infomations contains invalid key: {key} in {schema_str}")

            results = ResInfo.parse(gen_opapi_info, schema_function)

            if func_kind == SchemaKind.inplace:
                return_argument = [
                    schema_function.func.arguments.self_arg.argument.name
                ]
            else:
                return_argument = [result.name for result in results]

            if len(return_argument) == 0:
                return_args = ''
            else:
                if len(return_argument) == 1:
                    return_args = return_argument[0]
                elif func_kind == SchemaKind.out:
                    return_args = f"std::forward_as_tuple({', '.join(return_argument)})"
                else:
                    move_args = ', '.join(
                        f'std::move({arg})' for arg in return_argument)
                    return_args = f'std::make_tuple({move_args})'
                return_args = ''.join([' ', return_args])

            if cmd_args_expand and func_kind == SchemaKind.functional:
                aclnn_arguments = f"{aclnn_arguments}, {', '.join(return_argument)}"

            struct_info = StructInfo(
                name=defn_name,
                aclnn_name=aclnn_name,
                cmd_args=aclnn_arguments,
                func=schema_function,
                results=results,
                return_args=return_args,
                acl_op=acl_op,
                new_params=new_params_dict,
            )
            struct_infos.append(struct_info)

        return struct_infos

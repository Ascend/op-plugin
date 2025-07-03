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

import os
from collections import defaultdict
from typing import List, Dict, Sequence, Tuple
import yaml

from codegen.code_template import CodeTemplate
from codegen.gen import FileManager
from codegen.model import (BaseTy, SchemaKind, BaseType, NativeFunction)
from codegen.utils import concatMap, PathManager
from codegen.context import native_function_manager
from codegen.api.types import NativeSignature
from codegen.api import cpp
from .model import StructInfo, ResInfo, filt_input_tensor


USED_KEYS = ['official', 'custom', 'autograd']


PYTORCH_VERSION = os.environ.get('PYTORCH_VERSION').split('.')


ACLNN_FUNCTIONS_DEFINITION = CodeTemplate("""\
${return_type} ${func_name}(${args_str})
{
    ${do_compatibility}
    ${new_params_def}
    ${define_size_or_dtype}
    ${check_or_apply_tensor}
    EXEC_NPU_CMD(${aclnnargs});
    ${infer_name}
    return${result};
}
""")


ACLNN_FUNCTIONS_DELEGATE = CodeTemplate("""\
${return_type} ${func_name}(${args_str})
{
    return op_api::${func_name_out}(${args_str_out});
}

""")


DO_COMPATIBILITY = CodeTemplate("""\
DO_COMPATIBILITY(${aclnnname}, acl_op::${func_name}(${args_exprs_str}));
""")


APPLY_TENSOR = CodeTemplate("""\
at::Tensor ${result_name} = npu_preparation::apply_tensor_without_format(${size},
                                                            ${dtype});
""")


CHECK_TENSOR = CodeTemplate("""\
npu_preparation::check_tensor({${input}}, ${result_name}, ${dtype}, ${size});
""")


INFER_NAME = CodeTemplate("""\
at::namedinference::propagate_names(${result_name}, ${infer_func});
""")


def filt_op_branch(struct_ops: Dict) -> Dict:
    support_ops = []
    for key in USED_KEYS:
        support_ops += struct_ops[key]

    version = f"v{PYTORCH_VERSION[0]}.{PYTORCH_VERSION[1]}"

    def filt(op) -> bool:
        op_api_version = op.get('op_api', None)
        if op_api_version is None:
            is_support_version = False
        elif op_api_version == 'all_version':
            is_support_version = True
        elif isinstance(op_api_version, list):
            is_support_version = version >= op_api_version[0]
        else:
            is_support_version = version in op_api_version

        return 'gen_opapi' in op.keys() and is_support_version

    filt_ops = list(filter(lambda op: filt(op), support_ops))
    return filt_ops


def remove_empty_lines(text):
    lines = text.split('\n')
    struct_code = '\n'.join(line for line in lines if line.strip())
    return struct_code + '\n\n'


def parse_struct_yaml(path, native_functions: Sequence[NativeFunction]) -> List[StructInfo]:
    path = os.path.realpath(path)
    PathManager.check_directory_path_readable(path)
    with open(path, 'r') as struct_file:
        struct_ops = yaml.safe_load(struct_file)

    filt_ops = filt_op_branch(struct_ops)
    struct_infos = StructInfo.from_yaml(filt_ops, native_functions)

    return struct_infos


def gen_size_dtype_map(resinfos: List['ResInfo']) -> Tuple[Dict[str, str], Dict[str, str]]:
    size_map: Dict[str, str] = defaultdict(None)
    dtype_map: Dict[str, str] = defaultdict(None)
    for resinfo in resinfos:
        if size_map.get(resinfo.size) is None:
            size_map[resinfo.size] = f"output_size_{len(size_map)}"
        if dtype_map.get(resinfo.dtype) is None:
            dtype_map[resinfo.dtype] = f"output_dtype_{len(dtype_map)}"
    return size_map, dtype_map


def compute_op_api_definition(struct: StructInfo):
    f = struct.func

    with native_function_manager(f):
        kind = f.func.kind()
        sig = NativeSignature(f.func, prefix='', symint=False)
        name = cpp.name(f.func)
        args = sig.arguments()
        args_str = ', '.join(a.defn() for a in args)
        args_exprs_str = ', '.join(a.name for a in args)
        return_type = cpp.returns_type(f.func.returns).cpp_type()

        res_infos = struct.results

        # The inplace interface inherits out interface and implements a direct call to the out interface.
        if struct.structured_inherit is not None and kind == SchemaKind.inplace:
            delegate_function = struct.structured_inherit
            delegate_name = cpp.name(delegate_function.func)
            delegate_args_exprs_str = f'{args_exprs_str}, {f.func.arguments.self_arg.argument.name}'
            return [ACLNN_FUNCTIONS_DELEGATE.substitute(
                    return_type=return_type,
                    func_name=name,
                    args_str=args_str,
                    func_name_out=delegate_name,
                    args_str_out=delegate_args_exprs_str,)]

        do_compatibility = DO_COMPATIBILITY.substitute(aclnnname=struct.aclnn_name,
                                                       func_name=name,
                                                       args_exprs_str=args_exprs_str) if struct.acl_op else ""

        tensor_arguments = ", ".join(filt_input_tensor(f.func.arguments.flat_non_out))

        new_params_def = "".join(
            [f"auto {para_name} = {para_def};\n" for para_name, para_def in struct.new_params.items()])

        valid_param_set = set(struct.new_params.keys()) | set(map(lambda arg: arg.name, args)) | \
                          set(map(lambda res: res.name, res_infos))
        aclnn_params_set = set(map(lambda arg: arg.strip(), struct.cmd_args.split(',')[1:]))

        if not aclnn_params_set.issubset(valid_param_set):
            raise RuntimeError(f"exec configuration field contains invalid parameters"
                               f"{aclnn_params_set - valid_param_set}")

        size_map, dtype_map = gen_size_dtype_map(res_infos)

        define_size = "".join(
            [f"auto {name} = {size};\n" for size, name in size_map.items()])
        define_dtype = "".join(
            [f"auto {name} = {dtype};\n" for dtype, name in dtype_map.items()])
        define_size_or_dtype = "".join([define_size, define_dtype])

        if kind == SchemaKind.out:
            apply_tensor_list = list(concatMap(
                lambda res_info:
                [CHECK_TENSOR.substitute(input=tensor_arguments,
                                         result_name=res_info.name,
                                         size=size_map[res_info.size],
                                         dtype=dtype_map[res_info.dtype])],
                res_infos))
        elif kind == SchemaKind.inplace:
            apply_tensor_list = []
        else:
            apply_tensor_list = list(concatMap(
                lambda res_info:
                [APPLY_TENSOR.substitute(result_name=res_info.name,
                                         size=size_map[res_info.size],
                                         dtype=f'{res_info.option}.options().dtype({dtype_map[res_info.dtype]})')],
                res_infos))

        infer_name_list = list(concatMap(
            lambda res_info:
            INFER_NAME.substitute(result_name=res_info.name,
                                  infer_func=res_info.infer_name)
            if res_info.infer_name else "",
            res_infos))

        infer_name = "".join(infer_name_list)

        apply_tensor = "".join(apply_tensor_list)

    return [remove_empty_lines(
            ACLNN_FUNCTIONS_DEFINITION.substitute(
                return_type=return_type,
                func_name=name,
                new_params_def=new_params_def,
                define_size_or_dtype=define_size_or_dtype,
                args_str=args_str,
                check_or_apply_tensor=apply_tensor,
                do_compatibility=do_compatibility,
                aclnnargs=struct.cmd_args,
                result=struct.return_args,
                infer_name=infer_name))]


def gen_op_api(
    fm: FileManager,
    struct_functions: Sequence[StructInfo]
) -> None:

    fm.write_with_template(
        f'StructKernelNpuOpApi.cpp', f'StructKernelNpuOpApi.cpp', lambda: {
            'op_api_definition': list(concatMap(
                lambda f: compute_op_api_definition(f),
                struct_functions
            ))}
    )

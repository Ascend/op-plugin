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

from torchnpugen.code_template import CodeTemplate
from torchnpugen.gen import FileManager
from torchnpugen.model import BaseTy, SchemaKind, BaseType, NativeFunction
from torchnpugen.op_codegen_utils import concatMap, PathManager
from torchnpugen.context import native_function_manager
from torchnpugen.api.types import NativeSignature
from torchnpugen.api import cpp
from .model import STRUCTURED_GEN_OPAPI_ALLOWED_KEYS, StructInfo, ResInfo, filt_input_tensor


USED_KEYS = ['official', 'custom', 'autograd']
SYMINT_OPS = set()

_torch_ver = os.environ.get('PYTORCH_VERSION')
if not _torch_ver:
    raise RuntimeError('PYTORCH_VERSION environment variable is required')
PYTORCH_VERSION = _torch_ver.split('.')
if len(PYTORCH_VERSION) < 2:
    raise RuntimeError('PYTORCH_VERSION must contain major and minor components')


ACLNN_FUNCTIONS_DEFINITION = CodeTemplate("""\
${return_type} ${func_name}(${args_str})
{
    ${integral_identity_guard}
    ${do_compatibility}
    ${new_params_def}
    ${cpu_scalar_h2d_code}
    ${compute_names}
    ${define_size_or_dtype}
    ${check_or_apply_tensor}
    ${cpu_scalar_op_code}
    ${exec_cmd}(${aclnnargs});
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


INTEGRAL_IDENTITY_GUARD = CodeTemplate("""\
if (at::isIntegralType(${tensor}.scalar_type(), /*includeBool=*/false)) {
    return acl_op::${func_name}(${args_exprs_str});
}

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

INFER_NAME_GROUP = CodeTemplate("""\
at::namedinference::propagate_names_if_nonempty(${result_name}, maybe_names);
""")

COMPUTE_NAME_GROUP = CodeTemplate("""\
std::vector<at::Tensor> tensor_list = {${tensor_list}};
auto maybe_names = op_plugin::utils::compute_names_npu(tensor_list);
""")


def is_support_version(op):
    op_api_version = op.get('op_api', None)
    version = f"v{PYTORCH_VERSION[0]}.{PYTORCH_VERSION[1]}"

    def parse_version(v):
        return tuple(map(int, v.lstrip('v').split('.')))

    if op_api_version is None:
        is_support = False
    elif op_api_version == 'all_version':
        is_support = True
    elif isinstance(op_api_version, list):
        is_support = parse_version(version) >= parse_version(op_api_version[0])
    else:
        is_support = version in op_api_version
    return is_support


def filt_op_branch(struct_ops: Dict) -> Dict:
    support_ops = []
    for key in USED_KEYS:
        if struct_ops[key] is not None:
            support_ops += struct_ops[key]

    def filt_gen_opapi(op) -> bool:
        if 'gen_opapi' not in op.keys() or not is_support_version(op):
            return False
        gen_opapi_info = op.get('gen_opapi')
        if not isinstance(gen_opapi_info, dict):
            raise RuntimeError(f"{op.get('func')} has invalid gen_opapi configuration")
        if op.get('structured_delegate') is not None:
            raise RuntimeError(
                f"{op.get('func')} specifies structured_delegate, so gen_opapi is not allowed"
            )
        if bool(op.get('structured', False)):
            invalid_keys = set(gen_opapi_info.keys()) - STRUCTURED_GEN_OPAPI_ALLOWED_KEYS
            if invalid_keys:
                raise RuntimeError(
                    f"{op.get('func')} is marked structured, so gen_opapi only supports "
                    f"{sorted(STRUCTURED_GEN_OPAPI_ALLOWED_KEYS)}. Unsupported keys: {sorted(invalid_keys)}"
                )
        return True

    if 'symint' in struct_ops and struct_ops['symint']:
        for op in struct_ops['symint']:
            if is_support_version(op):
                SYMINT_OPS.add(op['func'].split("(")[0])

    filt_ops = [op for op in support_ops if filt_gen_opapi(op)]
    return filt_ops


def remove_empty_lines(text):
    lines = text.split('\n')
    struct_code = '\n'.join(line for line in lines if line.strip())
    return struct_code + '\n\n'


def parse_struct_yaml(path, native_functions: Sequence[NativeFunction]) -> List[StructInfo]:
    path = os.path.realpath(path)
    PathManager.check_directory_path_readable(path)
    with open(path, 'r', encoding='utf-8') as struct_file:
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


def _gen_cpu_scalar_op_code(struct: StructInfo, args, exec_cmd: str) -> str:
    if not struct.cpu_scalar_op:
        return ""
    branch_code_parts = []
    copy_set = set(struct.cpu_scalar_h2d) if struct.cpu_scalar_h2d else set()
    result_names = [res.name for res in struct.results] if struct.results else []
    cmd_args_list = [arg.strip() for arg in struct.cmd_args.split(',')]
    original_params = cmd_args_list[1:]
    # Collect scalar params that need .item() to avoid binding rvalue to lvalue reference
    scalar_params = {}
    for op in struct.cpu_scalar_op:
        if op.param not in scalar_params:
            scalar_params[op.param] = f"{op.param}_scalar"
    # Generate scalar variable declarations
    scalar_decls = "".join(
        f"at::Scalar {var} = {param}.item();\n"
        for param, var in scalar_params.items()
    )
    for op in struct.cpu_scalar_op:
        params_list = list(original_params)
        for i, p in enumerate(params_list):
            if p == op.param:
                params_list[i] = scalar_params[op.param]
            elif p in copy_set:
                params_list[i] = f"{p}_cp"
        for result_name in result_names:
            for i, p in enumerate(params_list):
                if p == "out":
                    params_list[i] = result_name
        params_str = ", ".join(params_list)
        condition = f"npu_preparation::IsCPUScalar({op.param})"
        branch_code_parts.append(
            f"if ({condition}) {{\n"
            f"    {scalar_decls}"
            f"    {exec_cmd}({op.exec_cmd}, {params_str});\n"
            f"    return{struct.return_args};\n"
            f"}}\n"
        )
    return "".join(branch_code_parts)


def _gen_cpu_scalar_h2d_code(struct: StructInfo, args) -> str:
    if not struct.cpu_scalar_h2d:
        return ""
    copy_code_parts = []
    arg_names = [a.name for a in args]
    tensor_arg_names = [a.name for a in args
                        if hasattr(a, 'argument') and hasattr(a.argument, 'type')
                        and isinstance(a.argument.type, BaseType) and a.argument.type.name == BaseTy.Tensor]
    copy_set = set(struct.cpu_scalar_h2d)
    for param in struct.cpu_scalar_h2d:
        if param not in arg_names:
            raise RuntimeError(f"cpu_scalar_h2d contains invalid parameter: {param}")
        non_copy_tensor_args = [n for n in tensor_arg_names if n not in copy_set]
        other_copy_params = [p for p in struct.cpu_scalar_h2d if p != param]
        if non_copy_tensor_args:
            device_expr = f"{non_copy_tensor_args[0]}.device()"
        elif other_copy_params:
            device_expr = f"{other_copy_params[0]}.device()"
        else:
            device_expr = "c10::Device(c10::DeviceType::PrivateUse1)"
        copy_code_parts.append(
            f"at::Tensor {param}_cp = {param};\n"
            f"if (npu_preparation::IsCPUScalar({param})) {{\n"
            f"    at::Scalar {param}_scalar = {param}.item();\n"
            f"    {param}_cp = npu_preparation::copy_scalar_to_device({param}_scalar, {param}.scalar_type(), {device_expr});\n"
            f"}}\n"
        )
    return "".join(copy_code_parts)


def _replace_cmd_args_with_copy(struct: StructInfo, args) -> str:
    if not struct.cpu_scalar_h2d:
        return struct.cmd_args
    cmd_args_list = [arg.strip() for arg in struct.cmd_args.split(',')]
    scalar_set = set(struct.cpu_scalar_h2d)
    new_args_list = []
    for arg in cmd_args_list:
        if arg in scalar_set:
            new_args_list.append(f"{arg}_cp")
        else:
            new_args_list.append(arg)
    return ", ".join(new_args_list)


def _replace_scalar_refs_in_expr(expr: str, cpu_scalar_h2d: List[str]) -> str:
    if not cpu_scalar_h2d or expr is None:
        return expr
    result = expr
    for param in cpu_scalar_h2d:
        result = result.replace(f"{param}.sizes()", f"{param}_cp.sizes()")
        result = result.replace(f"{param}.scalar_type()", f"{param}_cp.scalar_type()")
        result = result.replace(f"{param}.device()", f"{param}_cp.device()")
        result = result.replace(f"{param}.options()", f"{param}_cp.options()")
        result = result.replace(f"({param},", f"({param}_cp,")
        result = result.replace(f", {param},", f", {param}_cp,")
        result = result.replace(f", {param})", f", {param}_cp)")
        result = result.replace(f"({param})", f"({param}_cp)")
    if result == expr:
        for param in cpu_scalar_h2d:
            if result == param:
                result = f"{param}_cp"
                break
    return result


def compute_op_api_definition(struct: StructInfo, env_aclnn_extension_switch: bool):
    f = struct.func
    is_symint = struct.name in SYMINT_OPS
    with native_function_manager(f):
        kind = f.func.kind()
        sig = NativeSignature(f.func, prefix='', symint=is_symint)
        name = cpp.name(f.func)
        name = name + '_symint' if is_symint else name
        args = sig.arguments()
        args_str = ', '.join(a.defn() for a in args)
        args_exprs_str = ', '.join(a.name for a in args)
        return_type = cpp.returns_type(f.func.returns).cpp_type()

        res_infos = struct.results

        # The inplace interface inherits out interface and implements a direct call to the out interface.
        if struct.structured_inherit is not None and kind == SchemaKind.inplace:
            delegate_function = struct.structured_inherit
            delegate_name = cpp.name(delegate_function.func)
            delegate_name = delegate_name + '_symint' if is_symint else delegate_name
            delegate_args_exprs_str = f'{args_exprs_str}, {f.func.arguments.self_arg.argument.name}'
            return [
                ACLNN_FUNCTIONS_DELEGATE.substitute(
                    return_type=return_type,
                    func_name=name,
                    args_str=args_str,
                    func_name_out=delegate_name,
                    args_str_out=delegate_args_exprs_str,
                )
            ]

        do_compatibility = (
            DO_COMPATIBILITY.substitute(aclnnname=struct.aclnn_name, func_name=name, args_exprs_str=args_exprs_str)
            if struct.acl_op
            else ""
        )

        integral_identity_guard = ""
        if struct.integral_identity_tensor is not None:
            integral_identity_guard = INTEGRAL_IDENTITY_GUARD.substitute(
                tensor=struct.integral_identity_tensor, func_name=name, args_exprs_str=args_exprs_str
            )

        integral_identity_guard = ""
        if struct.integral_identity_tensor is not None:
            integral_identity_guard = INTEGRAL_IDENTITY_GUARD.substitute(
                tensor=struct.integral_identity_tensor,
                func_name=name,
                args_exprs_str=args_exprs_str)

        if struct.cpu_scalar_op and struct.acl_op:
            for op in struct.cpu_scalar_op:
                if op.exec_cmd != struct.aclnn_name:
                    do_compatibility += DO_COMPATIBILITY.substitute(aclnnname=op.exec_cmd,
                                                                     func_name=name,
                                                                     args_exprs_str=args_exprs_str)

        tensor_arguments_list = filt_input_tensor(f.func.arguments.flat_non_out)
        if struct.cpu_scalar_h2d:
            scalar_set = set(struct.cpu_scalar_h2d)
            tensor_arguments_list = [f"{t}_cp" if t in scalar_set else t for t in tensor_arguments_list]
        tensor_arguments = ", ".join(tensor_arguments_list)

        new_params_def = "".join(
            [f"auto {para_name} = {para_def};\n" for para_name, para_def in struct.new_params.items()]
        )

        exec_cmd = "EXEC_NPU_CMD_EXT" if env_aclnn_extension_switch else "EXEC_NPU_CMD"

        cpu_scalar_op_code = _gen_cpu_scalar_op_code(struct, args, exec_cmd)
        cpu_scalar_h2d_code = _gen_cpu_scalar_h2d_code(struct, args)
        cmd_args = _replace_cmd_args_with_copy(struct, args)

        valid_param_set = (
            set(struct.new_params.keys())
            | set(map(lambda arg: arg.name, args))
            | set(map(lambda res: res.name, res_infos))
        )
        aclnn_params_set = set(map(lambda arg: arg.strip(), struct.cmd_args.split(',')[1:]))

        if not aclnn_params_set.issubset(valid_param_set):
            raise RuntimeError(
                f"exec configuration field contains invalid parameters{aclnn_params_set - valid_param_set}"
            )

        size_map, dtype_map = gen_size_dtype_map(res_infos)

        if struct.cpu_scalar_h2d:
            new_res_infos = []
            for ri in res_infos:
                new_size = _replace_scalar_refs_in_expr(ri.size, struct.cpu_scalar_h2d)
                new_dtype = _replace_scalar_refs_in_expr(ri.dtype, struct.cpu_scalar_h2d)
                new_option = _replace_scalar_refs_in_expr(ri.option, struct.cpu_scalar_h2d)
                new_res_infos.append(ResInfo(
                    name=ri.name, size=new_size, dtype=new_dtype,
                    option=new_option, infer_name=ri.infer_name))
            res_infos = new_res_infos
            size_map, dtype_map = gen_size_dtype_map(res_infos)

        define_size = "".join([f"auto {name} = {size};\n" for size, name in size_map.items()])
        define_dtype = "".join([f"auto {name} = {dtype};\n" for dtype, name in dtype_map.items()])

        define_size_or_dtype = "" if kind == SchemaKind.inplace else "".join([define_size, define_dtype])

        if kind == SchemaKind.out:
            apply_tensor_list = list(
                concatMap(
                    lambda res_info: [
                        CHECK_TENSOR.substitute(
                            input=tensor_arguments,
                            result_name=res_info.name,
                            size=size_map[res_info.size],
                            dtype=dtype_map[res_info.dtype],
                        )
                    ],
                    res_infos,
                )
            )
        elif kind == SchemaKind.inplace:
            apply_tensor_list = []
        else:
            apply_tensor_list = list(
                concatMap(
                    lambda res_info: [
                        APPLY_TENSOR.substitute(
                            result_name=res_info.name,
                            size=size_map[res_info.size],
                            dtype=f'{res_info.option}.options().dtype({dtype_map[res_info.dtype]})',
                        )
                    ],
                    res_infos,
                )
            )

        compute_name_list = []
        infer_name_list = []
        for res_info in res_infos:
            if res_info.infer_name is not None:
                tensor_list = res_info.infer_name.split(", ")
                if len(tensor_list) == 1:
                    names = INFER_NAME.substitute(result_name=res_info.name, infer_func=res_info.infer_name)
                    infer_name_list.append(names)
                    compute_name_list.append("")
                else:
                    name_list = COMPUTE_NAME_GROUP.substitute(tensor_list=res_info.infer_name)
                    names = INFER_NAME_GROUP.substitute(result_name=res_info.name)
                    infer_name_list.append(names)
                    compute_name_list.append(name_list)
            else:
                infer_name_list.append("")
                compute_name_list.append("")

        compute_names = "".join(compute_name_list)

        infer_name = "".join(infer_name_list)

        apply_tensor = "".join(apply_tensor_list)

        # use EXEC_NPU_CMD or EXEC_NPU_CMD_EXT according to environment variable
        exec_cmd = "EXEC_NPU_CMD_EXT" if env_aclnn_extension_switch else "EXEC_NPU_CMD"

    return [
        remove_empty_lines(
            ACLNN_FUNCTIONS_DEFINITION.substitute(
                return_type=return_type,
                func_name=name,
                integral_identity_guard=integral_identity_guard,
                new_params_def=new_params_def,
                define_size_or_dtype=define_size_or_dtype,
                args_str=args_str,
                check_or_apply_tensor=apply_tensor,
                do_compatibility=do_compatibility,
                compute_names=compute_names,
                exec_cmd=exec_cmd,
                aclnnargs=cmd_args,
                result=struct.return_args,
                infer_name=infer_name,
                cpu_scalar_op_code=cpu_scalar_op_code,
                cpu_scalar_h2d_code=cpu_scalar_h2d_code
            )
        )
    ]


def gen_op_api(fm: FileManager, struct_functions: Sequence[StructInfo], env_aclnn_extension_switch: bool) -> None:
    # 根据环境变量生成不同的include语句
    if env_aclnn_extension_switch:
        includes = """#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/extension.h>
#include <ATen/native/TypeProperties.h>
#include "op_plugin/include/npu_cpp_extension.h" """
    else:
        includes = """#include <ATen/native/TypeProperties.h>
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h" """

    fm.write_sharded(
        'StructKernelNpuOpApi.cpp',
        struct_functions,
        key_fn=lambda s: str(s.func.func),
        base_env={'includes': includes},
        env_callable=lambda s: {
            'op_api_definition': compute_op_api_definition(s, env_aclnn_extension_switch),
        },
        num_shards=32,  # Empirical value; reduces compilation time from ~10min to ~40s+ per file after sharding
        sharded_keys={'op_api_definition'},
    )

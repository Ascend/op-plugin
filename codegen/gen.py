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

import os
import stat
import functools
import hashlib
from typing import (List, Dict, Optional, Set, Callable, Any,
                    Union, TypeVar, Iterable, Tuple)
from collections import defaultdict
import yaml

from codegen.code_template import CodeTemplate
from codegen.model import (NativeFunction, SelfArgument,
                           TensorOptionsArguments,
                           assert_never)
from codegen.api.types.signatures import NativeSignature
from codegen.context import native_function_manager
from codegen.utils import concatMap, context



T = TypeVar('T')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                           RUN IT ALL
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

@functools.lru_cache(maxsize=None)
def _read_template(template_fn: str) -> CodeTemplate:
    return CodeTemplate.from_file(template_fn)


# String hash that's stable across different executions, unlike builtin hash
def string_stable_hash(s: str) -> int:
    sha256 = hashlib.sha256(s.encode('latin1')).digest()
    return int.from_bytes(sha256, byteorder='little')

# A small abstraction for writing out generated files and keeping track
# of what files have been written (so you can write out a list of output
# files)
class FileManager:
    install_dir: str
    template_dir: str
    dry_run: bool
    filenames: Set[str]

    def __init__(self, install_dir: str, template_dir: str, dry_run: bool) -> None:
        self.install_dir = install_dir
        self.template_dir = template_dir
        self.filenames = set()
        self.dry_run = dry_run

    @staticmethod
    def _remove_path_safety(filepath: str) -> None:
        if os.path.islink(filepath):
            raise RuntimeError(f"Invalid path is a soft chain: {filepath}")
        if os.path.exists(filepath):
            os.remove(filepath)

    @staticmethod
    def _write_if_changed(filename: str, contents: str) -> None:
        old_contents: Optional[str]
        filepath = os.path.realpath(filename)
        try:
            with open(filepath, 'r') as f:
                old_contents = f.read()
        except IOError:
            old_contents = None
        if contents != old_contents:
            FileManager._remove_path_safety(filepath)
            with os.fdopen(os.open(filepath, os.O_RDWR | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR), "w") as f:
                f.write(contents)
            os.chmod(filepath, stat.S_IRUSR | stat.S_IEXEC | stat.S_IRGRP | stat.S_IXGRP)

    def write_with_template(self, filename: str, template_fn: str,
                            env_callable: Callable[[], Union[str, Dict[str, Any]]]) -> None:
        filename = '{}/{}'.format(self.install_dir, filename)
        if filename in self.filenames:
            raise ValueError(f"duplicate file write {filename}")
        self.filenames.add(filename)
        if not self.dry_run:
            env = env_callable()
            if isinstance(env, dict):
                if 'generated_comment' not in env:
                    comment = "@" + "generated by tools/codegen/gen.py"
                    comment += " from {}".format(os.path.basename(template_fn))
                    env['generated_comment'] = comment
                env['legacy_th_headers'] = []
                template = _read_template(os.path.join(self.template_dir, template_fn))
                self._write_if_changed(filename, template.substitute(env))
            elif isinstance(env, str):
                self._write_if_changed(filename, env)
            else:
                assert_never(env)


    def write(self, filename: str, env_callable: Callable[[], Union[str, Union[str, Dict[str, Any]]]]) -> None:
        self.write_with_template(filename, filename, env_callable)

    def write_sharded(
            self,
            filename: str,
            items: Iterable[T],
            *,
            key_fn: Callable[[T], str],
            env_callable: Callable[[T], Dict[str, List[str]]],
            num_shards: int,
            base_env: Optional[Dict[str, Any]] = None,
            sharded_keys: Set[str]
    ) -> None:

        everything: Dict[str, Any] = {'shard_id': 'Everything'}
        shards: List[Dict[str, Any]] = [{'shard_id': f'_{i}'} for i in range(num_shards)]
        all_shards = [everything] + shards

        if base_env is not None:
            for shard in all_shards:
                shard.update(base_env)

        for key in sharded_keys:
            for shard in all_shards:
                if key in shard:
                    if not isinstance(shard[key], list):
                        raise TypeError("sharded keys in base_env must be a list")
                    shard[key] = shard[key].copy()
                else:
                    shard[key] = []


        def merge_env(into: Dict[str, List[str]], from_: Dict[str, List[str]]) -> None:
            for k, v in from_.items():
                if k not in sharded_keys:
                    raise KeyError("undeclared sharded key {k}")
                into[k] += v

        for item in items:
            key = key_fn(item)
            sid = string_stable_hash(key) % num_shards
            env = env_callable(item)

            merge_env(shards[sid], env)
            merge_env(everything, env)

        dot_pos = filename.rfind('.')
        if dot_pos == -1:
            dot_pos = len(filename)
        base_filename = filename[:dot_pos]
        extension = filename[dot_pos:]

        for shard in all_shards:
            shard_id = shard['shard_id']
            self.write_with_template(f"{base_filename}{shard_id}{extension}",
                                     filename,
                                     lambda: shard)

        # filenames is used to track compiled files, but FooEverything.cpp isn't meant to be compiled
        self.filenames.discard(
            f"{self.install_dir}/{base_filename}Everything{extension}")

    def write_outputs(self, filename: str) -> None:
        """Write a file containing the list of all outputs which are
        generated by this script.
        """
        self._write_if_changed(
            filename,
            ''.join(name + ";" for name in sorted(self.filenames)))


SYMINT_SET = set()


def parse_native_yaml_struct(
    es: object,
) -> List[NativeFunction]:

    rs: List[NativeFunction] = []
    if not es:
        return rs

    if 'symint' not in es:
        raise AssertionError("Can't find symint in yaml.")
    if 'official' not in es:
        raise AssertionError("Can't find official in yaml.")
    if 'custom' not in es:
        raise AssertionError("Can't find custom in yaml.")

    if es['symint']:
        for e in es['symint']:
            global SYMINT_SET
            SYMINT_SET.add(e['func'].split("(")[0])

    all_funcs = []
    if es['official']:
        all_funcs += es['official']
    if es['custom']:
        all_funcs += es['custom']
    if ('quant' in es) and es['quant']:
        all_funcs += es['quant']

    if not isinstance(all_funcs, list):
        raise TypeError("all_funcs must be a list")

    for e in all_funcs:
        funcs = e.get("func")
        with context(lambda: f"in:\n  {funcs}"):
            func, m = NativeFunction.from_yaml(e)
            rs.append(func)

    return rs


def gen_function_declaration(
    f: NativeFunction,
    backend_decalarations: Dict,
):
    with native_function_manager(f):
        has_symint = False
        op_name = str(f.func.name.name)
        global SYMINT_SET
        if f.func.is_out_fn():
            op_name += "_out"
        if str(f.func.name) in SYMINT_SET:
            op_name += "_symint"
            has_symint = True

        sig = NativeSignature(f.func, prefix='', symint=has_symint)
        sig_str = f"OP_PLUGIN_HIDDEN {sig.decl(name=op_name)};"
        backend_decalarations["op_api"].append(sig_str)
        backend_decalarations["acl_op"].append(sig_str)
        if f.sparse is not None:
            op_name += "_sparse"
            backend_decalarations["sparse"].append(f"OP_PLUGIN_HIDDEN {sig.decl(name=op_name)};")


def gen_return(
    f: NativeFunction,
    deprecated_dict: Dict,
) -> List[Optional[str]]:
    ret = []
    with native_function_manager(f):
        has_symint = False
        op_name_with_overload = str(f.func.name)
        op_name = str(f.func.name.name)
        global SYMINT_SET
        if f.func.is_out_fn():
            op_name += "_out"
        if str(f.func.name) in SYMINT_SET:
            op_name += "_symint"
            has_symint = True

        sig = NativeSignature(f.func, prefix='', symint=has_symint)
        args_exprs_str = ', '.join(a.name for a in sig.arguments())

        impl_name = f.impl_name
        if not f.impl_name:
            impl_name = op_name
        
        deprecated_warn = ""
        if op_name_with_overload in deprecated_dict.keys():
            deprecated_func = f'torch_npu.{str(f.func.name.name)}'
            deprecated_replace = deprecated_dict[op_name_with_overload]
            if deprecated_replace is not None:
                deprecated_warn += f'TORCH_WARN_ONCE("{deprecated_func} is deprecated and will be removed in future version. \
Use {deprecated_replace} instead.");'
            else:
                deprecated_warn += f'TORCH_WARN_ONCE("{deprecated_func} is deprecated and will be removed in future version.");'

        format_check = []
        format_display = []
        place_holder = []
        format_for_args = []
        is_aclnn_only = "c10_npu::IsAclnnOnly()"
        for a in sig.arguments():
            argument = a.argument
            if isinstance(a.argument, SelfArgument):
                argument = a.argument.argument
            if not isinstance(a.argument, TensorOptionsArguments) and argument.type.is_tensor_like():
                format_for_args.append(
                    f"    bool {a.name}_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat({a.name});\n")
                format_check.append(f" && {a.name}_base_format")
                format_display.append(f", !{a.name}_base_format")
                place_holder.append(f", {a.name} is internal format: %d")

        if "op_api" in f.impl_ns and "acl_op" in f.impl_ns:
            if not f.internal_format_opapi:
                ret.append(f"""{sig.defn(name=op_name)}{{
    {deprecated_warn}
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
{"".join(format_for_args)}
    ASCEND_LOGI("{impl_name} exec with jit compile: %d{"".join(place_holder)}",
                !is_jit_disable{"".join(format_display)});
    if (is_jit_disable{"".join(format_check)}) {{
        return op_api::{impl_name}({args_exprs_str});
    }} else {{
        if ({is_aclnn_only}) {{
            TORCH_CHECK(false,
                "Current device only support aclnn operator, and current operator {impl_name} do not support internal format.",
                PTA_ERROR(ErrCode::NOT_SUPPORT));
        }}
        return acl_op::{impl_name}({args_exprs_str});
    }}
}}
""")
            else:
                ret.append(f"""{sig.defn(name=op_name)}{{
    {deprecated_warn}
    bool is_jit_disable = at_npu::native::env::CheckJitDisable();
    ASCEND_LOGI("{impl_name} exec with jit compile: %d", !is_jit_disable);
    if (is_jit_disable) {{
        return op_api::{impl_name}({args_exprs_str});
    }} else {{
        return acl_op::{impl_name}({args_exprs_str});
    }}
}}
""")
        elif "op_api" in f.impl_ns:
            ns = f.impl_ns[0]
            if f.internal_format_opapi:
                ret.append(f"""{sig.defn(name=op_name)}{{
    {deprecated_warn}
    return {ns}::{impl_name}({args_exprs_str});
}}
""")
            else:
                ret.append(f"""{sig.defn(name=op_name)}{{
    {deprecated_warn}
{"".join(format_for_args)}
    if ({("".join(format_check)).replace(" && ", " || !").replace(" || ", "", 1)}) {{
        TORCH_CHECK(false,
            "Current operator {impl_name} do not support internal format. ",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }}
    return {ns}::{impl_name}({args_exprs_str});
}}
""")
        elif "acl_op" in f.impl_ns:
            ns = f.impl_ns[0]
            ret.append(f"""{sig.defn(name=op_name)}{{
    {deprecated_warn}
    if ({is_aclnn_only}) {{
        TORCH_CHECK(false,
            "Current device only support aclnn operator, "
            "but current operator {impl_name} do not have aclnn implementation",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }}
    return {ns}::{impl_name}({args_exprs_str});
}}
""")
        if f.sparse is not None:
            ret.append(f"""{sig.defn(name=op_name + "_sparse")}{{
    {deprecated_warn}
    return sparse::{impl_name}_sparse({args_exprs_str});
}}
""")
    return ret


def parse_native_yaml(
    path: str,
    deprecate_path: str,
) -> Tuple[Dict[str, list], List[Optional[str]]]:

    with open(path, "r") as f:
        es = yaml.safe_load(f)
    
    with open(deprecate_path, "r") as f:
        dp = yaml.safe_load(f)

    res = parse_native_yaml_struct(es)
    backend_declarations = defaultdict(list)
    for f in res:
        gen_function_declaration(f, backend_declarations)
    deprecated = dp["deprecated"]
    deprecated_dict = {}
    for item in deprecated:
        deprecated_dict[item.get("name")] = item.get("replace", None)
    dispatch_registrations_body = sorted(set(concatMap(lambda f: gen_return(f, deprecated_dict), res)))

    return backend_declarations, dispatch_registrations_body

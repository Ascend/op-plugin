import pathlib
import argparse
import os
import stat
import re
from collections import namedtuple, Counter, defaultdict
import itertools
from dataclasses import dataclass, field
from typing import List, Dict, Union, Sequence, Optional, Set, Callable
import yaml

import torchgen
from torchgen.code_template import CodeTemplate
from torchgen.gen import (parse_tags_yaml, FileManager, parse_native_yaml,
                          get_grouped_native_functions, error_check_native_functions)
from torchgen.model import (BackendIndex, DispatchKey,
                            NativeFunction, NativeFunctionsGroup, OperatorName,
                            BackendMetadata, is_cuda_dispatch_key)
from torchgen.context import with_native_function
from torchgen.native_function_generation import add_generated_native_functions
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import Target, concatMap, context, NamespaceHelper
import torchgen.dest as dest
import torchgen.api.dispatcher as dispatcher
import torchgen.api.native as native
from torchgen.api.cpp import JIT_TO_CPP_DEFAULT
from torchgen.gen_backend_stubs import gen_dispatchkey_nativefunc_headers

from torchgen.code_template import CodeTemplate
from torchgen.gen import (parse_tags_yaml, FileManager, cpp_string, error_check_native_functions)
from torchgen.model import (BackendIndex, DispatchKey, Variant,
                            NativeFunction, OperatorName, BackendMetadata, TensorOptionsArguments)
from torchgen.utils import concatMap, mapMaybe
from torchgen.context import with_native_function, native_function_manager, method_with_native_function, with_native_function_and_index
from torchgen.api.types import DispatcherSignature
from torchgen.api import cpp
from torchgen.dest.register_dispatch_key import RegisterDispatchKey
from torch_npu_gen.custom_functions import parse_custom_yaml, gen_custom_functions_dispatch
from torch_npu_gen.utils import rename_privateuse1_dispatch_key, get_torchgen_dir


METHOD_DEFINITION = CodeTemplate("""\
${return_type} ${name}(${args_str}) {
  ${device_check}
  ${device_guard}
  ${type_definition_body}
}

""")


CUSTOM_DISPATCH = CodeTemplate("""\
return ${impl_name}(${args_exprs_str});""")


@with_native_function
def compute_op_definition(f: NativeFunction):
    out_num = len(f.func.arguments.out)
    sig = DispatcherSignature.from_schema(f.func, prefix=f'wrapper_{f.func.name.overload_name}_')
    name = sig.name()
    args = sig.arguments()
    args_str = ', '.join(a.defn() for a in args)

    args_exprs_str = ', '.join(a.name for a in args)

    impl_name = f"at_npu::native::NPUNativeFunctions::{cpp.name(f.func)}"

    check_out = [f'TORCH_CHECK(out.size() == {out_num}, "expected tuple of {out_num} elements but got ", out.size(), '
                 f'OPS_ERROR(ErrCode::PARAM));']
    unpack_out = check_out + [f'at::Tensor {args[-out_num + i].name} = out[{i}];' for i in range(out_num)] \
        if out_num > 1 else ''
    out_return_type = '::std::tuple<{}>'.format(', '.join(['at::Tensor'] * out_num))

    has_tensor_options = any(
        isinstance(a, TensorOptionsArguments)
        for a in f.func.arguments.non_out
    )

    # There is precedence for which argument we use to do
    # device guard.  This describes the precedence order.
    self_arg = (
        [f.func.arguments.self_arg.argument]
        if f.func.arguments.self_arg is not None
        else []
    )
    candidate_args = itertools.chain(
        self_arg,
        f.func.arguments.out,
        f.func.arguments.flat_positional,
    )
    candidate_tensor_args = []
    for a in candidate_args:
        if a.type.is_tensor_like():
            candidate_tensor_args.append(f"{a.name}")

    candidate_args = itertools.chain(
        f.func.arguments.out,
        f.func.arguments.flat_positional,
        f.func.arguments.flat_kwarg_only,
    )
    device_check = RegisterDispatchKey.gen_device_check(
        f.device_check, list(candidate_args), name
    )

    candidate_args = itertools.chain(
        self_arg,
        f.func.arguments.out,
        f.func.arguments.flat_positional,
    )
    # Only tensor like arguments are eligible
    device_of = next(
        (
            f"{a.name}"
            for a in candidate_args
            if a.type.is_tensor_like()
        ),
        None,
    )

    device_guard = ""
    if has_tensor_options and device_of is not None:
        device_guard = f"""
c10::OptionalDeviceGuard device_guard(device_of({device_of}));
if (device.has_value()) {{
device_guard.reset_device(device_or_default(device));
}}
"""
    elif has_tensor_options:
        # kernel is creating a tensor
        device_guard = """
const c10::DeviceGuard device_guard(device_or_default(device));"""
    elif device_of is not None:
        # kernel is operating on existing tensors
        device_guard = f"const c10::OptionalDeviceGuard device_guard(device_of({device_of}));"

    return [METHOD_DEFINITION.substitute(
        return_type=out_return_type if out_num > 1 else cpp.returns_type(f.func.returns).cpp_type(),
        name=name,
        args_str=','.join(a.defn() for a in args[:-out_num]) + ', at::TensorList out' if out_num > 1 else args_str,
        unpack_out=unpack_out,
        device_check=device_check,
        device_guard=device_guard,
        type_definition_body=[CUSTOM_DISPATCH.substitute(impl_name=impl_name, args_exprs_str=args_exprs_str)]
    )]


@dataclass(frozen=True)
class RegisterCustomSchema:
    known_tags: Dict[str, int] = field(default_factory=dict)

    @method_with_native_function
    def __call__(self, f: NativeFunction):
        out_num = len(f.func.arguments.out)
        if out_num > 1:
            decl = re.compile(r"(?P<name>[^\(]+)\((?P<args>.*)\) -> (?P<returns>.*)").findall(str(f.func))[0]
            func_schema = decl[0] + '(' + ','.join(decl[1].split(',')[:-out_num]) + ', Tensor[] out) -> (' + ', '.join(
                ['Tensor'] * out_num) + ')'
        else:
            func_schema = str(f.func)

        tags = "{" + ", ".join(f"at::Tag::{tag}" for tag in sorted(f.tags)) + "}"
        maybe_tags = ""
        if tags not in self.known_tags:
            idx = len(self.known_tags)
            self.known_tags[tags] = idx
            maybe_tags = f"const std::vector<at::Tag> tags_{idx} = {tags};\n"
        tag_index = f", tags_{self.known_tags[tags]}"
        if tags == "{}":
            tag_index = ""

        pattern = r'\bself\b(?=[,\)])'
        func_schema = re.sub(pattern, 'input', func_schema)

        if f.has_composite_explicit_autograd_kernel:
            name = DispatcherSignature.from_schema(f.func, prefix=f'wrapper_{f.func.name.overload_name}_').name()
            return f'{maybe_tags}m.def({cpp_string(func_schema)}, TORCH_FN(at_npu::native::{name}){tag_index});\n'
        else:
            return f'{maybe_tags}m.def({cpp_string(func_schema)}{tag_index});\n'


@with_native_function_and_index
def compute_register_impl(f: NativeFunction, backend_index):
    if (backend_index is not None) and (backend_index.get_kernel(f) is None):
        return []

    if f.has_composite_explicit_autograd_kernel:
        return []
    else:
        name = DispatcherSignature.from_schema(f.func, prefix=f'wrapper_{f.func.name.overload_name}_').name()
        return [f'm.impl("{f.func.name}", TORCH_FN(at_npu::native::{name}));\n']


def gen_custom_register(fm: FileManager, custom_functions: Sequence[NativeFunction], custom_backend_indices):

    fm.write_with_template(f'CustomRegisterSchema.cpp', 'CustomRegisterSchema.cpp', lambda: {
        'custom_op_definitions': list(concatMap(
            lambda f: compute_op_definition(f),
            custom_functions
        )),
        'custom_schema_registrations': list(mapMaybe(
            RegisterCustomSchema(),
            custom_functions
        )),
        'custom_impl_registrations': list(concatMap(
            lambda f: compute_register_impl(f, None),
            custom_functions
        )),
    })


def get_torch_npu_gen_dir():
    # get path of torchgen, then get tags.yaml and native_functions.yaml
    try:
        import torch_npu_gen
        return os.path.dirname(os.path.realpath(torch_npu_gen.__file__))
    except Exception:
        _, _, exc_traceback = sys.exc_info()
        frame_summary = traceback.extract_tb(exc_traceback)[-1]
        return os.path.dirname(frame_summary.filename)


def run(source_yaml: str, output_dir: str, dry_run: bool) -> None:
    torchgen_path = get_torchgen_dir()
    tags_yaml_path = os.path.join(torchgen_path, 'packaged/ATen/native/tags.yaml')
    custom_functions, custom_backend_indices = parse_custom_yaml(source_yaml, tags_yaml_path)
    pta_template_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "templates")
    fm = FileManager(install_dir=output_dir, template_dir=pta_template_dir, dry_run=dry_run)
    gen_custom_register(fm, custom_functions, custom_backend_indices)
    pta_template_dir = os.path.join(get_torch_npu_gen_dir(), "templates")
    fm = FileManager(install_dir=output_dir, template_dir=pta_template_dir, dry_run=dry_run)
    gen_custom_functions_dispatch(fm, custom_functions)


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate backend stub files')
    parser.add_argument(
        '-s',
        '--source_yaml',
        help='path to source yaml file containing operator external definitions')
    parser.add_argument(
        '-o', '--output_dir', help='output directory')
    parser.add_argument(
        '--dry_run', type=bool, default=False, help='output directory')
    options = parser.parse_args()

    run(options.source_yaml, options.output_dir, options.dry_run)


if __name__ == '__main__':
    main()
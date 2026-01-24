import argparse
import os
import re
from typing import List, Dict
import yaml

from torchgen.packaged.autograd.gen_inplace_or_view_type import gen_inplace_or_view_type
from torchgen.packaged.autograd.gen_autograd_functions import gen_autograd_functions_lib

import torchgen.gen
from torchgen.code_template import CodeTemplate
from torchgen.api.autograd import NativeFunctionWithDifferentiabilityInfo, DifferentiabilityInfo, match_differentiability_info
from torchgen.model import NativeFunction, FunctionSchema
from torchgen.packaged.autograd.gen_inplace_or_view_type import gen_inplace_or_view_type_env
from torchgen.packaged.autograd.gen_autograd_functions import process_function
from torch_npu_gen.utils import get_torchgen_dir
from torch_npu_gen.custom_functions import parse_custom_yaml

from .gen_variable_type import gen_variable_type, gen_variable_type_head    


def parse_native_and_custom_yaml_(*args, **kwargs):
    return parse_custom_yaml(*args, **kwargs)


def gen_inplace_or_view_type_env_for_npu(
    fn: NativeFunctionWithDifferentiabilityInfo,
) -> Dict[str, List[str]]:
    gen_code = gen_inplace_or_view_type_env(fn)

    if len(gen_code['inplace_or_view_method_definitions']):
        gen_code['ops_headers'] = []
        method_definitions = re.sub(pattern=r"at::_ops::(\w+)::redispatch",
                                    repl=r'at_npu::redispatch::\1',
                                    string=gen_code['inplace_or_view_method_definitions'][0])
        gen_code['inplace_or_view_method_definitions'] = [method_definitions]
    return gen_code


def apply_autograd_patches():
    torchgen.gen.parse_native_yaml = parse_native_and_custom_yaml_
    torchgen.packaged.autograd.gen_inplace_or_view_type.gen_inplace_or_view_type_env = \
        gen_inplace_or_view_type_env_for_npu


apply_autograd_patches()


def filt_npu_autograd_functions(
    native_functions_path: str,
    funcs_with_diff_infos: List[NativeFunctionWithDifferentiabilityInfo]
):
    npu_funcs_with_diff_infos: List[NativeFunctionWithDifferentiabilityInfo] = []
    torch_functions = set()
    with open(native_functions_path, 'r') as f:
        es = yaml.safe_load(f)
    for e in es:
        torch_functions.add(e.get('func').split('(')[0])

    npu_autograd_functions = set()
    torch_derivatives_functions = set()
    for f in funcs_with_diff_infos:
        name = str(f.func.func.name)
        # f.info is differentiabilityinfo. Existence of variants ops with a differentiabilityinfo of none.
        if f.info and name not in torch_functions:
            npu_funcs_with_diff_infos.append(f)
            npu_autograd_functions.add(name)
        if f.info and name in torch_functions:
            torch_derivatives_functions.add(name)
    return npu_funcs_with_diff_infos, npu_autograd_functions, torch_derivatives_functions


def filter_out_native_autograd_function(
    native_funcs: List[NativeFunction],
    differentiability_infos: Dict[FunctionSchema, Dict[str, DifferentiabilityInfo]],
):
    result: List[NativeFunction] = []
    derivatives_name_list: List[str] = []

    for diffinfo_dict in differentiability_infos.values():
        for info in diffinfo_dict.values():
            derivatives_name_list.append(str(info.func.func.name))
    for funcs in native_funcs:
        func_name = str(funcs.func.name)
        func_base_name = str(funcs.func.name.name.base)
        if (func_name in derivatives_name_list) or (func_base_name in derivatives_name_list):
            result.append(funcs)
    return result


def parse_derivatives(native_functions_path: str, tags_path: str, autograd_dir: str, npu_native_functions_path: str):
    from torchgen.packaged.autograd.load_derivatives import load_derivatives
    derivatives_path = os.path.join(autograd_dir, 'derivatives.yaml')
    differentiability_infos, _ = load_derivatives(derivatives_path, npu_native_functions_path, tags_path)
    native_funcs = parse_custom_yaml(npu_native_functions_path, tags_path).native_functions
    funcs = filter_out_native_autograd_function(native_funcs, differentiability_infos)
    funcs_with_diff_infos: List[NativeFunctionWithDifferentiabilityInfo] = []
    funcs_with_diff_infos = match_differentiability_info(funcs, differentiability_infos)

    return (differentiability_infos, native_funcs, funcs_with_diff_infos)


def gen_autograd(
    native_functions_path: str,
    tags_path: str,
    out: str,
    autograd_dir: str,
    npu_native_functions_path: str
) -> None:
    differentiability_infos, _, funcs_with_diff_infos =\
        parse_derivatives(native_functions_path, tags_path, autograd_dir, npu_native_functions_path)
    npu_funcs_with_diff_infos, _, _ = filt_npu_autograd_functions(native_functions_path, funcs_with_diff_infos)
    template_path = os.path.join(autograd_dir, 'templates')

    # Generate VariableType.cpp
    gen_variable_type(out, funcs_with_diff_infos, template_path)

    # Generate VariableType.h
    gen_variable_type_head(out, funcs_with_diff_infos, template_path)

    # Generate ADInplaceOrViewType.cpp
    gen_inplace_or_view_type(out, native_functions_path, tags_path, npu_funcs_with_diff_infos, template_path)

    # Generate Functions.h/cpp
    gen_autograd_functions_lib(out, differentiability_infos, template_path)



def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate autograd C++ files script')
    parser.add_argument('--out_dir', metavar='OUT',
                        help='path to output directory')
    parser.add_argument('--autograd_dir', metavar='AUTOGRAD',
                        help='path to autograd directory')
    parser.add_argument('--npu_native_function_dir',
                        help='path to npu_native_functions.yaml')
    args = parser.parse_args()

    torchgen_path = get_torchgen_dir()
    tags_yaml_path = os.path.join(torchgen_path, 'packaged/ATen/native/tags.yaml')
    native_yaml_path = os.path.join(torchgen_path, 'packaged/ATen/native/native_functions.yaml')

    gen_autograd(native_yaml_path,
                 tags_yaml_path,
                 args.out_dir,
                 args.autograd_dir,
                 args.npu_native_function_dir)


if __name__ == '__main__':
    main()
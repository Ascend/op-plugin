import os
import argparse

from torchnpugen.gen import parse_native_yaml, FileManager
from torchnpugen.op_codegen_utils import concatMap, PathManager


def main() -> None:

    parser = argparse.ArgumentParser(description='Generate backend stub files')
    parser.add_argument(
        '--to_cpu', type=str, default="TRUE", help='move op which npu does not support to cpu')
    parser.add_argument(
        '-s',
        '--source_yaml',
        help='path to source yaml file containing operator external definitions')
    parser.add_argument(
        '--deprecate_yaml',
        help='path to yaml file containing functions which is deprecated.')
    parser.add_argument(
        '-o', '--output_dir', help='output directory')
    parser.add_argument(
        '--dry_run', type=bool, default=False, help='output directory')
    parser.add_argument(
        '--version', type=str, default=None, help='pytorch version')
    parser.add_argument(
        '--impl_path', type=str, default=None, help='path to the source C++ file containing kernel definitions')
    options = parser.parse_args()

    source_yaml_path = os.path.realpath(options.source_yaml)
    deprecate_yaml_path = os.path.realpath(options.deprecate_yaml)
    PathManager.check_directory_path_readable(source_yaml_path)
    PathManager.check_directory_path_readable(deprecate_yaml_path)
    backend_declarations, dispatch_registrations_body = parse_native_yaml(source_yaml_path, deprecate_yaml_path)

    env_aclnn_extension_switch = os.getenv('ACLNN_EXTENSION_SWITCH')
    if env_aclnn_extension_switch:
        # Use absolute path relative to current script for templates
        script_dir = os.path.dirname(os.path.abspath(__file__))
        template_dir_make_file_manager = os.path.join(script_dir, "templates")
    else:
        template_dir_make_file_manager = "torchnpugen/templates"

    def make_file_manager(install_dir: str) -> FileManager:
        return FileManager(
            install_dir=install_dir, template_dir=template_dir_make_file_manager, dry_run=False
        )

    fm = make_file_manager(options.output_dir)

    pytorch_version = os.environ.get('PYTORCH_VERSION').split('.')
    torch_dir = f"v{pytorch_version[0]}r{pytorch_version[1]}"

    all_functions = sorted(set(concatMap(lambda f: [f],
                                         set(v for sublist in backend_declarations.values() for v in sublist))))

    fm.write_with_template(
        "OpInterface.h",
        "Interface.h",
        lambda: {
            "torch_dir": torch_dir,
            "namespace": "op_plugin",
            "declarations": all_functions,
        },
    )

    header_files = {
        "op_api": "OpApiInterface.h",
        "acl_op": "AclOpsInterface.h",
        "sparse": "SparseOpsInterface.h",
        "lazy_fusion": "DvmOpsInterface.h",
    }
    for op_type, file_name in header_files.items():
        fm.write_with_template(
            file_name,
            "Interface.h",
            lambda: {
                "torch_dir": torch_dir,
                "namespace": op_type,
                "declarations": backend_declarations[op_type],
            },
        )

    dvm_includes = (
        '#include "op_plugin/DvmOpsInterface.h"\n'
        '#include "op_plugin/ops/dvm/lazy_fusion_kernel.h"\n'
    )
    # When ACLNN_EXTENSION_SWITCH is set, use simplified includes (no FormatHelper/op_log) for OpInterface.cpp
    if env_aclnn_extension_switch:
        includes_block = f'''#include "torch_npu/csrc/framework/interface/EnvVariables.h"
// #include "torch_npu/csrc/framework/FormatHelper.h"
#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/SparseOpsInterface.h"
{dvm_includes}// #include "op_plugin/utils/op_log.h"
#include "op_plugin/OpInterface.h"
'''
    else:
        includes_block = f'''#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "torch_npu/csrc/framework/FormatHelper.h"
#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/SparseOpsInterface.h"
{dvm_includes}#include "op_plugin/utils/op_log.h"
#include "op_plugin/OpInterface.h"
'''

    fm.write_sharded(
        "OpInterface.cpp",
        dispatch_registrations_body,
        key_fn=lambda decl: decl,
        base_env={
            "namespace": "op_plugin",
            "includes_block": includes_block,
        },
        env_callable=lambda decl: {"declarations": [decl]},
        num_shards=32,  # Empirical value; reduces compilation time from ~150s to ~20s+ per file after sharding
        sharded_keys={"declarations"},
    )


if __name__ == '__main__':
    main()


import argparse

from codegen.gen import parse_native_yaml, FileManager


def main() -> None:

    parser = argparse.ArgumentParser(description='Generate backend stub files')
    parser.add_argument(
        '--to_cpu', type=str, default="TRUE", help='move op which npu does not support to cpu')
    parser.add_argument(
        '-s',
        '--source_yaml',
        help='path to source yaml file containing operator external definitions')
    parser.add_argument(
        '-o', '--output_dir', help='output directory')
    parser.add_argument(
        '--dry_run', type=bool, default=False, help='output directory')
    parser.add_argument(
        '--version', type=str, default=None, help='pytorch version')
    parser.add_argument(
        '--impl_path', type=str, default=None, help='path to the source C++ file containing kernel definitions')
    options = parser.parse_args()

    backend_declarations, dispatch_registrations_body = parse_native_yaml(options.source_yaml)

    def make_file_manager(install_dir: str) -> FileManager:
        return FileManager(
            install_dir=install_dir, template_dir="codegen/templates", dry_run=False
        )

    fm = make_file_manager(options.output_dir)

    fm.write_with_template(
        "OpInterface.h",
        "Interface.h",
        lambda: {
            "namespace": "op_plugin",
            "declarations": backend_declarations,
        },
    )
    fm.write_with_template(
        "OpAPIInterface.h",
        "Interface.h",
        lambda: {
            "namespace": "op_api",
            "declarations": backend_declarations,
        },
    )
    fm.write_with_template(
        "AclOpsInterface.h",
        "Interface.h",
        lambda: {
            "namespace": "acl_op",
            "declarations": backend_declarations,
        },
    )

    fm.write_with_template(
        "OpInterface.cpp",
        "OpInterface.cpp",
        lambda: {
            "namespace": "op_plugin",
            "declarations": dispatch_registrations_body,
        },
    )


if __name__ == '__main__':
    main()

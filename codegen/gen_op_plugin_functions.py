import re
import argparse
import os
import stat
import yaml

from codegen.utils import PathManager, get_version


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', type=str, default=None, help='pytorch version')
    parser.add_argument('-o', '--output_dir', help='output directory')
    parser.add_argument('-s', '--source_yaml', type=str, default=None, help='source yaml')
    args = parser.parse_args()
    pytorch_version = args.version.split('.')
    version = f"v{pytorch_version[0]}.{pytorch_version[1]}"
    output_dir = args.output_dir
    source_yaml = args.source_yaml

    source_yaml_path = os.path.realpath(source_yaml)
    PathManager.check_directory_path_readable(source_yaml_path)
    with open(source_yaml_path, 'r') as f:
        old_yaml = yaml.safe_load(f)

    new_yaml = {'official': [], 'custom': [], 'symint': [], 'tocpu': [], 'unsupported': [], 'quant': [], 'autograd': []}

    string = ['official', 'custom', 'symint', 'quant', 'autograd']
    all_version = old_yaml['all_version']

    # parse 'official', 'custom', 'symint', 'quant', 'autograd'
    for key in string:
        for item in old_yaml[key]:
            item.pop('gen_opapi', None)

            acl_op_sup_ver = get_version(item.get('acl_op', None), all_version)
            op_api_sup_ver = get_version(item.get('op_api', None), all_version)
            sparse = get_version(item.get('sparse', None), all_version)
            internal_format_opapi = get_version(item.get('internal_format_opapi', None), all_version)
            item.update({'acl_op':acl_op_sup_ver})
            item.update({'op_api':op_api_sup_ver})
            item.update({'sparse':sparse})
            item.update({'internal_format_opapi':internal_format_opapi})

            if version in (acl_op_sup_ver + op_api_sup_ver + sparse):
                new_item = item.copy()
                impl_ns = []
                for i in ['acl_op', 'op_api']:
                    if version in item.get(i, ''):
                        impl_ns.append(i)
                    new_item.pop(i, None)
                if len(acl_op_sup_ver) > 0 or len(op_api_sup_ver) > 0:
                    new_item['impl_ns'] = ', '.join(impl_ns)

                if version in item.get('sparse', ''):
                    new_item['sparse'] = 'op_api'
                else:
                    new_item.pop('sparse', None)
                if version in get_version(item.get('exposed', None), all_version):
                    new_item['exposed'] = True
                else:
                    new_item.pop('exposed', None)
                if version in item.get('internal_format_opapi', ''):
                    new_item['internal_format_opapi'] = True
                else:
                    new_item.pop('internal_format_opapi', None)

                new_yaml.get(key).append(new_item)

    # parse 'tocpu' and 'unsupported', only v1.11 and v2.0 supported
    if version in ['v1.11', 'v2.0']:
        for key in ['tocpu', 'unsupported']:
            for func in old_yaml[key]:
                if version in func['version']:
                    new_yaml.get(key).append(func['func'])
    else:
        del new_yaml['tocpu'], new_yaml['unsupported']

    if version in ['v1.11', 'v2.0', 'v2.1']:
        del new_yaml['quant']

    # save to new yaml
    flags = os.O_WRONLY | os.O_CREAT
    modes = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(f'{output_dir}/op_plugin_functions.yaml', flags, modes), 'w') as f:
        yaml.dump(data=new_yaml, stream=f, width=2000, sort_keys=False)

if __name__ == '__main__':
    main()

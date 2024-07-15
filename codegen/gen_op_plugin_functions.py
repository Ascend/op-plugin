import re
import argparse
import os
import stat
import yaml

OP_API = 'op_api'
ACL_OP = 'acl_op'


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

    if os.path.islink(source_yaml):
        raise RuntimeError(f'Invalid path is a soft chain: {source_yaml}')
    if os.path.exists(source_yaml):
        with open(source_yaml, 'r') as f:
            old_yaml = yaml.safe_load(f)

    new_yaml = {'official':[], 'custom':[], 'symint':[], 'tocpu':[], 'unsupported':[], 'quant':[]}

    string = ['official', 'custom', 'symint', 'quant']

    # official,custom,symint
    for key in string:
        for item in old_yaml[key]:
            if version in item['version']:
                new_item = item
                if item.get('impl_ns') and item.get('version'):
                    impl_ns_list = re.split(', ', item['impl_ns'])
                    version_list = re.split(', ', item['version'])
                    version_idx = [idx for idx, val in enumerate(version_list) if val == version]
                    support_version = [impl_ns_list[i] for i in version_idx]
                    new_item['impl_ns'] = ', '.join(support_version)
                del new_item['kernel'], new_item['version']
                if (version == 'v1.11' or version == 'v2.0') and item.get('sparse'):
                    del new_item['sparse']

                if item.get('exposed') and version in version_list:
                    new_item['exposed'] = True
                elif item.get('exposed'):
                    del new_item['sparse']

                new_yaml.get(key).append(new_item)

    # tocpu and unsupported
    if version in ['v1.11', 'v2.0']:
        for key in ['tocpu', 'unsupported']:
            for func in old_yaml[key]:
                if version in func['version']:
                    new_yaml.get(key).append(func['func'])
    else:
        del new_yaml['tocpu'], new_yaml['unsupported']

    if version in ['v1.11', 'v2.0', 'v2.1']:
        del new_yaml['quant']

    # save to yaml
    flags = os.O_WRONLY | os.O_CREAT
    modes = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(f'{output_dir}/new_op_plugin_functions.yaml', flags, modes), 'w') as f:
        yaml.dump(data=new_yaml, stream=f, width=2000, sort_keys=False)

if __name__ == '__main__':
    main()

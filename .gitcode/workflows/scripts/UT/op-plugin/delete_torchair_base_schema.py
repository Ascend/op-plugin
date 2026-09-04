import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--source_json', type=str, default=None)
args = parser.parse_args()
source_json = args.source_json

with open(source_json, 'r') as f:
    data = json.load(f)
filtered_data = {key: value for key, value in data.items() if not key.startswith("torch_npu.dynamo.")}

with open(source_json, 'w') as f:
    json.dump(filtered_data, f, indent=2)

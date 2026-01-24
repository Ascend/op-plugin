CDIR="$(cd "$(dirname "$0")" ; pwd -P)"

cd $CDIR

python3 -m codegen.gen \
    --output_dir="$CDIR"            \
    --source_yaml="$CDIR/test.yaml" \

python3 -m codegen.struct \
    --output_dir="$CDIR"            \
  --native_yaml="$CDIR/test.yaml" \
  --struct_yaml="$CDIR/test.yaml" \

python3 -m codegen.autograd.gen_autograd \
  --out_dir="$CDIR"            \
  --autograd_dir="$CDIR/codegen/autograd" \
  --npu_native_function_dir="$CDIR/test.yaml" \
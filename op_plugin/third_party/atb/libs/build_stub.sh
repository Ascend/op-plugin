CDIR="$(cd "$(dirname "$0")" ; pwd -P)"

cd ${CDIR}

gcc -fPIC -shared -o libatb.so -I../inc -I../../../../../acl/inc atb.cpp

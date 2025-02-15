CDIR="$(cd "$(dirname "$0")" ; pwd -P)"

cd ${CDIR}

gcc -fPIC -shared -o libatb.so -I./ atb.cpp

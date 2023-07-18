#!/bin/bash
cd "$(dirname "$0")"

if [ -d ./workspace ]; then
    rm -rf ./workspace
fi

mkdir ./workspace
cp -r ../fastsam ./workspace
cp -r ../images ./workspace
cp -r ../mobile_sam ./workspace
cp -r ../new_images ./workspace
cp -r ../scripts ./workspace
cp -r ../serve ./workspace
cp -r ../weights ./workspace
cp ../*.* ./workspace

docker build --no-cache -t  mobilesam_dev:1.0 .


docker run -itd -p 10027:10027 --name mobilesam_dev mobilesam_dev:1.0

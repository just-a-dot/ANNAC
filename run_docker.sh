#!/bin/sh

docker build -t keras-project docker/
docker run -it --runtime=nvidia -u $(id -u):$(id -g) -v $(pwd):/data keras-project bash


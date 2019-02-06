#!/bin/sh

docker build -t keras-project .
docker run -it -u $(id -u):$(id -g) -v $(pwd):/data keras-project bash


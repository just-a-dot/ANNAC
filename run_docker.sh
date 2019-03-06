#!/bin/sh

docker build -t keras-project:gpu .
sudo docker run --runtime=nvidia --rm -it -u $(id -u):$(id -g) -v $(pwd):/data keras-project:gpu bash


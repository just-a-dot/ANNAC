#!/bin/sh

if [ "$1" == 'gpu' ]
then
    docker build -t keras-project:gpu .
    sudo docker run --runtime=nvidia --rm -it -u $(id -u):$(id -g) -v $(pwd):/data keras-project:gpu bash
elif [ "$1" == 'cpu' ]
then
    docker build -t keras-project .
    sudo docker run --rm -it -u $(id -u):$(id -g) -v $(pwd):/data keras-project bash
else
    echo "ERROR: missing parameter (either gpu or cpu)"
fi


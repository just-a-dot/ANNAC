FROM tensorflow/tensorflow:latest-py3

RUN pip install keras
RUN apt-get install -y sox
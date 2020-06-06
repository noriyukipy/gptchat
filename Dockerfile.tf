FROM tensorflow/tensorflow:2.1.0-gpu-py3

COPY . /app
WORKDIR /app

RUN apt-get update && \
    apt-get install -y wget unzip language-pack-ja mecab libmecab-dev mecab-ipadic-utf8 jq

RUN pip install -e ./

WORKDIR /work
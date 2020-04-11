FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime

RUN apt-get update && \
    apt-get install -y language-pack-ja mecab libmecab-dev mecab-ipadic-utf8

ENV LC_ALL=ja_JP.UTF-8

# Install
COPY ./ /app
WORKDIR /app
RUN pip install ./

WORKDIR /work
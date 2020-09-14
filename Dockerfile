FROM tensorflow/tensorflow:2.3.0-gpu

RUN apt-get update && \
    apt-get install -y wget unzip language-pack-ja jq

COPY . /app
WORKDIR /app

RUN pip install -e ./

WORKDIR /work
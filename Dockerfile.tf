FROM tensorflow/tensorflow:2.1.0-gpu-py3

RUN apt-get update && \
    apt-get install -y wget unzip language-pack-ja mecab libmecab-dev mecab-ipadic-utf8 jq

RUN pip install jupyter==1.0.0 papermill==2.1.1 transformers==2.9.0 attrdict==2.0.1

# Create cache directory used by Black
RUN mkdir -p /root/.cache/black/19.10b0

# GPTChat

GPTChat provides conversation modeling CLI based on GPT-2.

This repository uses GPT-2 models provided by powerful and exciting OSS [🤗 Transformers](https://github.com/huggingface/transformers) by HuggingFace.

## Installation

Build Docker imageb.py.

```sh
$ git clone https://github.com/noriyukipy/gptchat
$ cd gptchat
$ docker image build -t gptchat .
```

Change directory to working directory.

```sh
$ cd config
```

## Tokenizer

Train your tokenizer with `gptchat.tokenizer.train`

```sh
$ docker container run  -v $(pwd):/work -w /work --rm -it gptchat python -m gptchat.tokenizer.train --config=tokenizer/config.yaml
```

## LM Model

**LM** model is a language model which aims to be used as a pretrained model for latter models.

### Preparation of Dataset

Before training, prepare language corpus such as Wikipedia.

### Training

```sh
$ docker container run --gpus all -v $(pwd):/work -w /work --rm -it gptchat python -m gptchat.lm.train --config=lm/config.yaml
```

### Serve API

```sh
$ docker container run -v $(pwd):/work -w /work -p 8000:8000 --rm -it gptchat python -m gptchat.lm.serve_api --config=lm/config.yaml --host=0.0.0.0 --port=8000
```

## ChatLM Model

**ChatLM** model is a sequence to sequence model by fine-tuning pretrained GPT-2 to generate a **response** from a **context** given by a user.

Context and response are given to the model separeted by the `[SEP]` token.
The `[CLS]` token is also appended at the end of the tokens.

To distinguish context and response, [token_type_id](https://huggingface.co/transformers/glossary.html#token-type-ids) is also given to the model.

|                | 1       | 2       | 3     | 5       | 6     |
| ---            | ---     | ---     | ---   | ---     | ---   |
| Token          | Good    | morning | [SEP] | Hi      | [CLS] |
| Token Type IDs | 0       | 0       | 1     | 1       | 1     |
| Attention mask | 1       | 1       | 1     | 1       | 1     |
| Target or not  | 1       | 1       | 1     | 1       | 1     |

This model is trained with [CrossEntropyLoss](https://pytorch.org/docs/stable/nn.html#crossentropyloss) of the only response tokens.

### Dataset Preparation

Prepare a chatting corpus which has two columns separated by a TAB character.
First column is a context, and the second is a response to the first column.

### Training

Train the model with

```sh
$ docker container run --gpus all -v $(pwd):/work -w /work --rm -it gptchat python -m gptchat.chatlm.train --config=chatlm/config.yaml
```

### Serve API

```sh
$ docker container run -v $(pwd):/work -w /work -p 8000:8000 --rm -it gptchat python -m gptchat.chatlm.serve_api --config=chatlm/config.yaml --host=0.0.0.0 --port=8000
```

## Logs

To monitor training progress, use tensorboard. Go to the output directory, and then run tensorboard.

```sh
$ docker container run -v $(pwd):/work -w /work -p 6006:6006 --rm gptchat tensorboard --logdir . --host=0.0.0.0
```

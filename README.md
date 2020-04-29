# GPTChat

GPTChat provides conversation modeling CLI based on GPT-2 for Japanese.

This repository uses GPT-2 models provided by powerful and exciting OSS [🤗 Transformers](https://github.com/huggingface/transformers) by HuggingFace.

## Installation

Build Docker image.

```sh
$ git clone https://github.com/noriyukipy/gptchat
$ cd gptchat
$ docker image build -t gptchat .
```

## Pretrained Model

### Preparation of Dataset

Before training, prepare Japanese language corpus such as Wikipedia.

### Training

Run `notebooks/gpt/train.ipynb` with [papermill](https://github.com/nteract/papermill).
Papermill is useful for recording your specified parameters and environments.

```sh
$ docker container run -v $(pwd):/work --gpus all --rm jupyter papermill notebooks/gpt/train.ipynb notebooks/gpt/output/output.ipynb -p n_ctx 512 -p block_size 512 -p data_dir notebooks/gpt/data -p output_dir notebooks/gpt/output
```

To check all the available argument, see the [notebook](notebooks/chat/train.ipynb).

After finish training, the trained model is in the `${output_dir}/model`


```sh
$ ls -1 notebooks/gpt/output/model/
added_tokens.json
config.json
pytorch_model.bin
special_tokens_map.json
tokenizer_config.json
vocab.txt
```

## ChatLM Model

GPTChat provides **ChatLM** Model.
The model is a sequence to sequence model by fine-tuning pretrained GPT-2 to generate a **response** from a **context** given by a user.

Context and response are given to the model separeted by the `[SEP]` token.
The `[CLS]` token is also appended at the end of the tokens.

To distinguish context and response, [token_type_id](https://huggingface.co/transformers/glossary.html#token-type-ids) is also given to the model.

|                | 1       | 2       | 3     | 5       | 6     |
| ---            | ---     | ---     | ---   | ---     | ---   |
| Token          | Good    | morning | [SEP] | Hi      | [CLS] |
| Token Type IDs | [CTX]   | [CTX]   | [RES] | [RES]   | [RES] |

This model is trained with [CrossEntropyLoss](https://pytorch.org/docs/stable/nn.html#crossentropyloss) of the only response tokens.

### Dataset Preparation

Prepare a chatting corpus which has two columns separated by a TAB character.
First column is a context, and the second is a response to the first column.

### Training

Run `notebooks/chatlm/train.ipynb` with [papermill](https://github.com/nteract/papermill).

Train the model with `

```sh
$ docker container run -v $(pwd):/work --gpus all --rm gptchat papermill notebooks/chatlm/train.ipynb notebooks/chatlm/output/output.ipynb -p pretrained_dir notebooks/gpt/output/model -p data_dir notebooks/chatlm/data -p output_dir notebooks/chatlm/output -p batch_size 32
```

To check all the available argument, see the [notebook](notebooks/chatlm/train.ipynb).

After finish training, the trained model is in the `${output_dir}/model`

```sh
$ ls -1 notebooks/chatlm/output/model/
added_tokens.json
config.json
pytorch_model.bin
special_tokens_map.json
tokenizer_config.json
vocab.txt
```

### Response Generation

Run `gptchat.cmd.serve_chatlm_api` to serve API server to generate response.

```sh
$ docker container run -v $(pwd):/work -p 8000:8000 --rm gptchat python -m gptchat.cmd.serve_chatlm_api --model_dir=output/chatlm/model --address=0.0.0.0 --port=8000 --top_p=0.95 --top_k=50 --max_len=20 --num_cands=3
```

Then you can give request to the server.

```sh
$ curl localhost:8000/chat -d '{"context": "これで完成！"}' -H"content-type:applicaiton/json" | jq
{
  "request": {
    "context": "これで完成！"
  },
  "response": "お疲れ様です!"
}
```
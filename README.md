# GPTChat

GPTChat is for those who are eager to build a chatting bot with the latest GPT-2 model especially for Japanese.
GPTChat provides two types of training and generating scripts; **BaseModel** and **ChatModel**.

BaseModel is a GPT-2 languaga model.
This repository aims to train it from scratch for non-English, especially for Japanese.
If you use pre-trained English model, the step to train BaseModel can be skipped.

ChatModel is a fine-tuned model of BaseModel for conversation.
ChatModel is similar to [2] expect for personality and history.
GPTChat ignores personality and only considers the latest utterance in the history.
See [1] and [2] for more details.

This repository uses GPT-2 models provided by powerful and exciting OSS [🤗 Transformers](https://github.com/huggingface/transformers) by HuggingFace.

Reference
- [1] *TransferTransfo: A Transfer Learning Approach for Neural Network Based Conversational Agents* by Thomas Wolf et al. (https://arxiv.org/abs/1901.08149)
- [2] *How to build a State-of-the-Art Conversational AI with Transfer Learning* by Thomas Wolf. (https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313)

## Installation

### Docker

```sh
$ git clone https://github.com/noriyukipy/gptchat
$ cd gptchat
$ docker image build -t gptchat .
```

You can run all the commands in this README from a Docker image to replace `python ` with `docker container run -v $(pwd):/workspace --gpus all --rm -it gptchat`.

### From Source

If you install GPTChat from source, install PyTorch 1.3 First.
Then install GPTChat by pip.

```sh
$ pip install git+https://github.com/noriyukipy/gptchat
```

For `bert-base-japanese` tokenizer, you should also install MeCab and its dictionary.

See [Dockerfile](./Dockerfile) for more details about the required environment.

## Training BaseModel

### Preparation of Dataset

Before training, prepare language corpus. In this tutorial, we use [WikiText-2](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) .

```sh
$ git clone https://github.com/noriyukipy/gptchat
$ cd gptchat/example
$ wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip
$ unzip wikitext-2-raw-v1.zip
```

In English, it may be good choice to skip training BaseModel and just uses GPT-2 pre-trained model `gpt2` provided by HuggingFace.
GPTChat can train ChatModel directly from `gpt2` model, so you can go to GPTChat section.

For Japanese, training BaseModel is indispensable. You shoud first prepare Japanese language corpus such as Wikipedia.

### Training

`gptchat.gpt.train` trains GPT2LMHeadModel from scratch.

```sh
$ python -m gptchat.gpt.train --output_dir=output --data=wikitext-2-raw/wiki.train.raw --seed=0 --checkpoint_steps=50000
```

Pass the dataset path to `--data` option.

While GPT2LMHeadModel is trained from scratch, tokenizer loads a pre-trained model.
`--tokenizer_model` can change a pre-trained tokenizer.
Default is `gpt2` to load `gpt2` pre-trained GPT2Tokenizer.
You can specify `bert-base-japanese` to load `bert-base-japanese` BertJapaneseTokenizer.

| --tokenizer_model | default? | Tokenizer Class | model name |
| --- | --- | --- | --- |
| gpt2 | yes | GPT2Tokenizer | gpt2 |
| bert-base-japanese | no | BertJapaneseTokenizer | bert-base-japanese |

Each `checkpoint_steps`, trained model will be saved under directory specified by `--output_dir` option.

```sh
$ tree output
output
├── step_0
│   ├── added_tokens.json
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.txt
└── step_50000-epoch_1-batch_50000
    ├── added_tokens.json
    ├── config.json
    ├── pytorch_model.bin
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    └── vocab.txt
```

Execute with `--help` to check other options.

```sh
$ python -m gptchat.gpt.train --help
```

### BaseModel Evaluation

`gptbase.basemodel.evaluate` evaluates a trained model with metrics of perplexity.
Pass a directory path to a trained model directory or pre-trained model name `gpt2` to `--model` .

```sh
# Evaluate trained model
$ python -m gptchat.gpt.evaluate --model=output/step_50000-epoch_1-batch_50000 --data=wikitext-2-raw/wiki.valid.raw
# Evaluate pre-trained model gpt2
$ python -m gptchat.gpt.evaluate --model=gpt2 --data=wikitext-2-raw/wiki.valid.raw
```

### Generation

`gptchat.gpt.generate` generates words from stdin input.

```
$ python -m gptchat.gpt.generate --model=gpt2
>>> Happy coding
Happy coding opportunity

After graduating from Technical Engineering with Electrical Engineering at SUNY at an early age I completed my education at Ohio State University.

During
```

## ChatModel

ChatModel utilizes GPT2DoubleHeadModel to generate a response from an input utterance.

Tokens to input the model are separated into input utterance and response by the separator `<sep>` .

|           | 1       | 2       | 3       | 4       | 5       | 6       |
| ---       | ---     | ---     | ---     | ---     | ---     | ---     |
| Token     | \<bos\> | Good    | morning | \<sep\> | Hi      | \<eos\> |

### Dataset Preparation

Prepare a chatting corpus which has two columns separated by a TAB character.
First column is a input utterance, and the second is a response to the first column.

```sh
$ head -n1 chat/train.txt
Hi.     Hi, how are you?
```

To train ChatModel, you should first run `gptchat.chat.add_distructors` to create a dataset having additional columns about distructors to each chat pairs.

```sh
$ cat chat/train.txt | python -m gptchat.chat.add_distractors --num_distractors=2 >chat/train_dist.txt
```

Distructors are invalid responses.
This script randomly select other responses as distractors and add them to new columns.

```sh
$ head -n1 chat/train_dist.txt
Hi.     Hi, how are you?        Machine learning.       Yes, I'm good at it.
```

Before starting training, I should mention that chat/train.txt is a toy corpus.
To train a meaningful chatting bot, you should prepare appropreate dialog dataset.

### Training Chatting Model

`gptchat.chat.train` trains ChatModel.
`--model` can accept the directory path you trained in the previous BaseModel step, or `gpt2` as a pre-traind model.

```sh
# from trained model
$ python -m gptchat.chat.train --output_dir=chat_output --model=output/step_5-epoch_1-batch_5 --data=chat/train_dist.txt --batch_size=2 --num_distructors=2 --checkpoint_steps=5 --num_epochs=10
# from pre-trained model
$ python -m gptchat.chat.train --output_dir=chat_output --model=gpt2 --data=chat/train_dist.txt --num_distructors=2 --checkpoint_steps=5 --num_epochs=10
```

As well as BaseModel, trained models in each `--checkpoint_steps` are saved under `--output_dir` directory.

### Response Generation

`gptchat.chat.generate` generates a response from input utterance via stdin.

```sh
$ python -m gptchat.chat.generate --model=chat_output/step_50-epoch_10-batch_5
>>> Hi
<bos> Hi <sep> Hi! Today!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

`gptchat.chat.serve` serves HTTP server to generate a response.

```sh
$ python -m gptchat.chat.serve --address=localhost --port=8080 --model=chat_output/step_50-epoch_10-batch_5
```

```sh
$ curl localhost:8080/generate -d '{"text": "Hi"}' -H"content-type:application/json" | jq
{
  "text": "Hi",
  "model_output": "<bos> Hi <sep> Hi!Hi!!!!!!!!!!!!!!!!!!!!!!!!!!!",
  "reply": "Hi!Hi!!!!!!!!!!!!!!!!!!!!!!!!!!!"
}
```

## Test

Execute test with pytest.

```sh
$ pytest -v test/
```

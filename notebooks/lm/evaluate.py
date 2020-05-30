import yaml
import attrdict
import os
import transformers
from train import set_seed
from train import load_dataset
from train import build_data
import tensorflow.keras as keras


def main(config):
    params = attrdict.AttrDict(yaml.load(open(config)))
    print(params)
    set_seed(params.seed)

    model_dir = os.path.join(params.output_dir, "model")

    tokenizer = transformers.BertJapaneseTokenizer.from_pretrained(model_dir)
    model = transformers.TFAutoModelWithLMHead.from_pretrained(model_dir)

    test_texts = load_dataset(params.data_dir + "/test.txt")

    x_test, y_test = build_data(tokenizer, test_texts, params.block_size)

    # Create optimizer
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # optimizer = keras.optimizers.Adam()
    model.compile(
        loss=[loss, *[None] * model.config.n_layer],
        metrics=[
            keras.metrics.SparseCategoricalCrossentropy(from_logits=True),
            keras.metrics.SparseCategoricalAccuracy(),
        ],
    )

    # Evaluate best model with test set
    res = model.evaluate(x_test, y_test)
    print(res)


if __name__ == "__main__":
    import fire

    fire.Fire(main)

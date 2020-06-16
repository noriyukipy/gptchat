from .train import encode_plus
import transformers
import numpy as np
import tensorflow as tf


def generate(model, tokenizer, top_k, top_p, max_length, text):
    model_input = encode_plus(
        context=text,
        tokenizer=tokenizer,
        response=None,
        add_eos_token=False,
    )

    for _ in range(max_length):
        model_input = {key: np.array([val]) for key, val in model_input.items()}
        outputs = model(model_input)

        # Predict next token
        next_token_logits = outputs[0][:, -1, :]
        next_token_logits = transformers.modeling_tf_utils.tf_top_k_top_p_filtering(
            next_token_logits,
            top_k=top_k,
            top_p=top_p,
        )
        next_token = tf.squeeze(
            tf.random.categorical(next_token_logits, dtype=tf.int32, num_samples=1),
            axis=1
        )

        # Update inputs to model
        model_input["input_ids"] = np.concatenate(
            [model_input["input_ids"].flatten(), next_token.numpy()]
        )
        model_input["token_type_ids"] = np.concatenate(
            [model_input["token_type_ids"].flatten(), np.array([1])]
        )
        model_input["attention_mask"] = np.concatenate(
            [model_input["attention_mask"].flatten(), np.array([1])]
        )

        if next_token[0] == tokenizer.cls_token_id:
            break

    return tokenizer.decode(model_input["input_ids"])

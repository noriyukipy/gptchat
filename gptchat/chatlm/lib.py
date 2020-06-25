from .train import encode_plus
import transformers
import numpy as np
import tensorflow as tf


def generate(model, tokenizer, top_k, top_p, max_length, context, response, bad_words_ids):
    model_input = encode_plus(
        context=context,
        tokenizer=tokenizer,
        response=response,
        add_eos_token=False,
    )
    max_length_with_context = max_length - len(model_input["input_ids"])

    for _ in range(max_length_with_context):
        model_input = {key: np.array([val]) for key, val in model_input.items()}
        outputs = model(model_input)

        # Predict next token
        next_token_logits = outputs[0][:, -1, :]
        next_token_logits = transformers.modeling_tf_utils.tf_top_k_top_p_filtering(
            next_token_logits,
            top_k=top_k,
            top_p=top_p,
        )

        # Stop bad words
        next_token_logits = stop_bad_words(
            tokenizer=tokenizer,
            prev_input_ids=tf.convert_to_tensor(model_input["input_ids"]),
            bad_words_ids=bad_words_ids,
            next_token_logits=next_token_logits,
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


def stop_bad_words(tokenizer, prev_input_ids, bad_words_ids, next_token_logits):
    """This method works the same as bad_words argument given by `generate` methos
    in transformers library utilizing some functions from
    transformers.modeling_tf_utils.

    https://github.com/huggingface/transformers/blob/75e1eed8d190afa5be30fba05cd872d79b492a24/src/transformers/modeling_tf_utils.py#L551
    """
    next_bad_ids_list = transformers.modeling_tf_utils.calc_banned_bad_words_ids(
        prev_input_ids,
        bad_words_ids
    )
    mask = [
        [next_id in next_bad_ids for next_id in range(len(tokenizer))]
        for next_bad_ids in next_bad_ids_list
    ]
    next_token_logits = transformers.modeling_tf_utils.set_tensor_by_indices_to_value(
        next_token_logits,
        tf.convert_to_tensor(mask, dtype=tf.bool),
        -float("inf")
    )
    return next_token_logits

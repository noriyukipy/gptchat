from .train import encode_plus
import numpy as np
import tensorflow as tf


def generate(
    model, tokenizer, top_k, top_p, max_length, context, bad_words_ids
):
    model_input = encode_plus(context=context, tokenizer=tokenizer)
    input_ids = tf.convert_to_tensor([model_input["input_ids"]])
    gen_ids = model.generate(
        input_ids,
        num_return_sequences=5,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        bad_words_ids=bad_words_ids,
        eos_token_id=tokenizer.sep_token_id,
        max_length=max_length,
    )
    gen_texts = []
    for gen_id in gen_ids:
        gen_id = gen_id.numpy().tolist()
        cln_id = gen_id[len(input_ids[0]):]
        gen_text = tokenizer.decode(cln_id)
        gen_texts.append(gen_text)

    # Select the median length text as a final output
    mid_idx = int(len(gen_texts) / 2)
    selected_text = list(sorted(gen_texts, key=lambda x: len(x)))[mid_idx]

    return selected_text, gen_texts


def generate_prepare_inputs_for_generation(sep_token_id):
    def prepare_inputs_for_generation(self, inputs, **kwargs):
        batch_size, seq_len = inputs.shape
        sep_idx = tf.where(inputs[0] == sep_token_id).numpy().max()

        context_len = sep_idx
        response_len = seq_len - sep_idx

        # After EOS, mask needs to be 0; however, tokens before EOS are only used and
        # outpus after EOS are not used. Therefore, for simplicity, use 1 as a attention mask after OES.
        attention_mask = np.concatenate(
            [np.ones((batch_size, context_len)), np.ones((batch_size, response_len))],
            axis=1,
        )

        # This output is passed to https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_tf_gpt2.py#L545
        return {
            "inputs": inputs,
            "attention_mask": tf.convert_to_tensor(attention_mask, dtype=tf.int32),
        }

    return prepare_inputs_for_generation

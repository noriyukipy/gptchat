from gptchat.lib.generator import TopPGenerator
from gptchat.lib import special_tokens
import torch
import re


def generate(tokenizer, model, text, max_len, top_p):
    # Prepare inputs
    seq = [
        [special_tokens.BOS] + tokenizer.tokenize(text),
        [special_tokens.SEP]
    ]
    tokens = sum(seq, [])
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    segments = (
        [special_tokens.SP1] * len(seq[0]) +
        [special_tokens.SP2] * len(seq[1])
    )
    segment_ids = tokenizer.convert_tokens_to_ids(segments)

    generator = TopPGenerator(model, top_p=top_p)
    input_ids = torch.tensor([token_ids])
    token_type_ids = torch.tensor([segment_ids])
    for _ in range(max_len):
        next_id = generator.step(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )
        input_ids = torch.cat([input_ids, next_id], dim=1)
        token_type_ids = torch.cat(
            [token_type_ids,
             torch.tensor([tokenizer.convert_tokens_to_ids([special_tokens.SP2])])
             ],
            dim=1
        )

        # If the next token is EOS, finish generating
        eos = special_tokens.EOS
        if list(next_id) == tokenizer.convert_tokens_to_ids([eos]):
            break

    return tokenizer.decode(input_ids.tolist()[0])


def extract_reply(model_output, unk_token):
    """Extract reply part from model output."""
    def escape(xs):
        return xs.replace("[", "\[").replace("]", "\]")
    pattern_to_replace = [
        (escape(special_tokens.EOS), ""),
        (escape(unk_token), ""),
        (r"##\S+", ""),
        (r"\s", ""),
    ]
    reply = model_output.split(special_tokens.SEP)[1]
    for pat, replaced in pattern_to_replace:
        reply = re.sub(pat, replaced, reply)

    return reply

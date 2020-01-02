import os
import logging
from transformers import PreTrainedTokenizer
from collections import Counter


class WhitespaceTokenizer(PreTrainedTokenizer):
    vocab_files_names = {'vocab_file': 'vocab.txt'}

    def __init__(self, vocab_file=None, max_len=None, **kwargs):
        super(WhitespaceTokenizer, self).__init__(max_len=max_len, **kwargs)

        if vocab_file:
            tokens = []
            with open(vocab_file) as f:
                for line in f:
                    token = line.strip("\n")
                    tokens.append(token)
            self.set_tokens(tokens)

    def set_tokens(self, tokens):
        self._token_to_id = {token: id_ for id_, token in enumerate(tokens)}
        self._id_to_token = {id_: token for id_, token in enumerate(tokens)}

    def build_from_corpus(self, corpus, max_vocab_size):
        """Build vocabulary from corpus."""
        tokens = self.extract_tokens(
            iterator=(line.strip("\n") for line in open(corpus)),
            max_vocab_size=max_vocab_size
        )
        self.set_tokens(tokens)

    def extract_tokens(self, iterator, max_vocab_size):
        """Extract tokens from text provided by iterator."""
        token_counter = Counter()
        for text in iterator:
            for token in self.tokenize(text):
                token_counter[token] += 1
        freq_ordered_tokens = [
            token for token, num in
            list(sorted(token_counter.items(), key=lambda x: (-x[1], x[0])))
        ]
        clipped_tokens = freq_ordered_tokens[:max_vocab_size]

        # Add special tokens
        special_tokens = list(sorted(self.all_special_tokens))
        if special_tokens:
            tokens = special_tokens + clipped_tokens

        return tokens

    @property
    def vocab_size(self):
        return len(self._token_to_id)

    def _tokenize(self, text):
        return text.split(" ")

    def _convert_token_to_id(self, token):
        return self._token_to_id.get(token, self._token_to_id[self.unk_token])

    def _convert_id_to_token(self, index):
        return self._id_to_token.get(index, self.unk_token)

    def save_vocabulary(self, vocab_path):
        """Save the tokenizer vocabulary to a directory or file."""
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(
                vocab_path,
                self.vocab_files_names['vocab_file']
            )
        else:
            raise ValueError("{} should be directory".format(vocab_path))

        with open(vocab_file, "w", encoding="utf-8") as writer:
            ordered_vocab = sorted(
                self._token_to_id.items(),
                key=lambda x: x[1]
            )
            index = 0
            for token, token_index in ordered_vocab:
                if index != token_index:
                    logging.warning(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(vocab_file)
                    )
                    index = token_index
                writer.write("{}\n".format(token))
                index += 1
        return (vocab_file, )



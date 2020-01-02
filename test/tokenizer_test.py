from gptchat.lib.tokenizer import WhitespaceTokenizer


def test_extract_tokens():
    iterator = [
        "テスト 文書 です 。",
        "どう で しょう か 。",
        "動き が わかる と いい です 。",
    ]

    tokenizer = WhitespaceTokenizer(unk_token="<unk>")
    tokens = tokenizer.extract_tokens(iterator, max_vocab_size=3)

    # tokens should be start from special tokens ordered by dictionary order
    # then, continue with other vocaburaries from the highest frequency
    assert len(tokens) == 4
    assert tokens == ["<unk>", "。", "です", "いい"]


def test_decode():
    iterator = [
        "テスト 文書 です 。",
        "どう で しょう か 。",
        "動き が わかる と いい です 。",
    ]

    tokenizer = WhitespaceTokenizer(unk_token="<unk>")
    tokens = tokenizer.extract_tokens(iterator, max_vocab_size=3)
    tokenizer.set_tokens(tokens)

    ids = tokenizer.encode("いい で しょう 元気")

    assert ids == [3, 0, 0, 0]


def test_encode():
    tokenizer = WhitespaceTokenizer(unk_token="<unk>")
    tokenizer.set_tokens(["<unk>", "はい"])

    tokens = tokenizer.convert_ids_to_tokens([1, 0, 0])

    assert tokens == ["はい", "<unk>", "<unk>"]

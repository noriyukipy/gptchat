from gptchat.lib import chat_utils


def test_extract_reply():
    example_table = [
        ("<bos> 新しい レストラン 開拓 し たい <sep> ##日 買っ て みよ う か [UNK] <eos>", "買ってみようか"),
        ("<bos> 新しい レストラン 開拓 し たい <sep> <sp1> [PAD] [PAD] [PAD] <eos>", ""),
    ]

    for reply, want in example_table:
        ans = chat_utils.extract_reply(reply)
        assert ans == want

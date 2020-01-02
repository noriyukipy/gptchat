from gptchat.lib import chat_utils


def test_extract_reply():
    reply = "<bos> 新しい レストラン 開拓 し たい <sep> ##日 買っ て みよ う か [UNK] <eos>"
    unk_token = "[UNK]"
    ans = chat_utils.extract_reply(reply, unk_token)
    assert ans == "買ってみようか"

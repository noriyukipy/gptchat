from distutils.core import setup


setup(
    name="gptchat",
    packages=["gptchat", "gptchat.gpt", "gptchat.chat", "gptchat.lib"],
    install_requires=[
        "transformers==2.3.0",
        "fire==0.2.1",
        "tqdm==4.38.0",
        "mecab-python3==0.996.2",
        "responder==2.0.5",
    ],
    version="0.1.0",
    author="Noriyuki Abe",
)

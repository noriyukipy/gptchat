import setuptools


setuptools.setup(
    name="gptchat",
    packages=setuptools.find_packages(),
    install_requires=[
        "transformers==2.3.0",
        "fire==0.2.1",
        "tqdm==4.38.0",
        "mecab-python3==0.996.2",
        "responder==2.0.5",
        "torch>=1.3.0",
    ],
    version="0.1.0",
    author="Noriyuki Abe",
)

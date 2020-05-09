import setuptools


setuptools.setup(
    name="gptchat",
    packages=setuptools.find_packages(),
    install_requires=[
        "transformers==2.9.0",
        "mecab-python3==0.996.2",
        "fire==0.2.1",
        "responder==2.0.5",
        "torch>=1.3.0",
        "pytest==5.4.1",
        "tqdm==4.43.0",
        "attrdict==2.0.1",
        "tensorboard==2.1.1",
        "jupyter==1.0.0",
        "papermill==1.2.1",
    ],
    version="0.2.0",
    author="Noriyuki Abe",
)

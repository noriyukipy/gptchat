import setuptools


setuptools.setup(
    name="gptchat",
    packages=setuptools.find_packages(),
    install_requires=[
        "transformers==2.10.0",
        "mecab-python3==0.996.2",
        "fire==0.2.1",
        "fastapi==0.57.0",
        "uvicorn==0.11.5",
        "pytest==5.4.1",
        "tqdm==4.43.0",
        "attrdict==2.0.1",
        "pyyaml==5.3.1",
    ],
    version="0.2.0",
    author="Noriyuki Abe",
)

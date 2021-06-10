import setuptools


setuptools.setup(
    name="gptchat",
    packages=setuptools.find_packages(),
    install_requires=[
        "transformers==2.10.0",
        "sentencepiece==0.1.91",
        "fire==0.2.1",
        "fastapi==0.65.2",
        "uvicorn==0.11.7",
        "pytest==5.4.1",
        "tqdm==4.43.0",
        "envyaml==0.2060",
    ],
    version="0.4.0",
    author="Noriyuki Abe",
)

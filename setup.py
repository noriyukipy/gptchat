import setuptools


setuptools.setup(
    name="gptchat",
    packages=setuptools.find_packages(),
    install_requires=[
        "transformers==2.10.0",
        "tokenizers==0.8.1",
        "fire==0.2.1",
        "fastapi==0.57.0",
        "uvicorn==0.11.5",
        "pytest==5.4.1",
        "tqdm==4.43.0",
        "envyaml==0.2060",
    ],
    version="0.3.1",
    author="Noriyuki Abe",
)

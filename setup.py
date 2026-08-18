from setuptools import setup, find_packages

setup(
    name="whitrum",
    version="0.1.0",
    description="Whitrum AI - ~350M parameter language model based on Qwen architecture",
    author="Oguzhan (Dr0xy-Drawn)",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "accelerate>=0.30.0",
        "safetensors>=0.4.0",
    ],
)

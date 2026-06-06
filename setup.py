import os
from setuptools import setup, find_packages

# Read the contents of the README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="agentic_qa",
    version="0.2.1",
    description="Autonomous Agentic QA System for testing RAG pipelines and LLM systems.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Om",
    url="https://github.com/om-ai-lab/agentic-qa",
    packages=find_packages(include=['agentic_qa', 'agentic_qa.*']),
    install_requires=[
        "langgraph>=0.2.0",
        "langchain>=0.3.0",
        "langchain-openai>=0.2.0",
        "langchain-core>=0.3.0",
        "langsmith>=0.1.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "requests>=2.30.0",
        "ragas>=0.1.0",
        "datasets>=2.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)

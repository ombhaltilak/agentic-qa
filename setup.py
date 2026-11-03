import os
from setuptools import setup, find_packages

# Read the contents of the README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="agentic_qa",
    version="0.3.0",
    description="Autonomous Agentic QA System for testing RAG pipelines and LLM systems.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Om Bhaltilak",
    license="MIT",
    url="https://github.com/ombhaltilak/agentic-qa",
    project_urls={
        "Source": "https://github.com/ombhaltilak/agentic-qa",
        "Changelog": "https://github.com/ombhaltilak/agentic-qa/blob/main/CHANGELOG.md",
        "Issues": "https://github.com/ombhaltilak/agentic-qa/issues",
    },
    keywords=["llm", "rag", "testing", "evaluation", "red-teaming", "langgraph", "llm-as-judge"],
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
        "pyyaml>=6.0",
    ],
    extras_require={
        "jev": ["typesafe-sdk>=0.7.5"],
        "ragas": ["ragas>=0.4.0", "datasets>=2.0.0"],
        "dashboard": ["streamlit>=1.38.0", "pandas>=2.0.0", "markdown>=3.5"],
        "dev": ["pytest>=8.0"],
    },
    entry_points={"console_scripts": ["agentic-qa=agentic_qa.cli:main"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Testing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)

"""
Research-paper RAG ("Attention Is All You Need") — importable system under test.

Same pipeline as research_paper_rag.ipynb (PyPDF -> 1000/200 chunks -> FAISS,
k=3 -> gpt-4o-mini, three-sentence answers), packaged so the CLI can test it:

    # as a LangChain Runnable (contexts read from the chain's "context" docs)
    agentic-qa run --target examples.research_paper_rag.rag:rag_chain ...
    # or as a plain function returning {"output", "contexts"}
    agentic-qa run --target examples.research_paper_rag.rag:answer ...

The FAISS index is cached in .cache/ so repeated QA runs don't re-embed the paper.
Settings: RAG_PDF_PATH, RAG_INDEX_DIR, RAG_MODEL (default gpt-4o-mini), RAG_TOP_K.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

PDF_URL = "https://arxiv.org/pdf/1706.03762.pdf"
SYSTEM_PROMPT = (
    "You are an assistant for question-answering tasks based on a research paper. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "Use three sentences maximum and keep the answer concise."
    "\n\n"
    "{context}"
)
DESCRIPTION = (
    "A Retrieval-Augmented Generation assistant that answers questions about the research paper "
    "'Attention Is All You Need' (Vaswani et al., 2017). It retrieves the 3 most similar 1000-character "
    "chunks from the paper and answers in at most three sentences, saying it does not know when the "
    "paper does not contain the answer."
)

_lock = threading.Lock()
_chain = None


def _pdf_path() -> Path:
    path = Path(os.getenv("RAG_PDF_PATH", "attention_is_all_you_need.pdf"))
    if not path.exists():
        import requests

        print(f"Downloading paper to {path} ...")
        resp = requests.get(PDF_URL, timeout=60)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    return path


def _build_chain():
    from dotenv import load_dotenv
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    from langchain_community.vectorstores import FAISS
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    load_dotenv()
    embeddings = OpenAIEmbeddings()
    index_dir = Path(os.getenv("RAG_INDEX_DIR", ".cache/faiss_attention"))
    if (index_dir / "index.faiss").exists():
        # Our own index, written below — safe to deserialize.
        vectorstore = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
    else:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        docs = PyPDFLoader(str(_pdf_path())).load()
        splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
        vectorstore = FAISS.from_documents(splits, embeddings)
        index_dir.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(index_dir))

    retriever = vectorstore.as_retriever(search_kwargs={"k": int(os.getenv("RAG_TOP_K", "3"))})
    llm = ChatOpenAI(model=os.getenv("RAG_MODEL", "gpt-4o-mini"))
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{input}")])
    return create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))


def get_chain():
    """Build once (thread-safe — the executor calls the SUT concurrently)."""
    global _chain
    with _lock:
        if _chain is None:
            _chain = _build_chain()
        return _chain


def answer(query: str) -> dict:
    """Plain-function interface: answer + retrieved contexts."""
    response = get_chain().invoke({"input": query})
    contexts = [doc.page_content for doc in response.get("context", [])]
    return {"output": response.get("answer", ""), "contexts": contexts}


def __getattr__(name: str):
    # `--target examples.research_paper_rag.rag:rag_chain` resolves the chain lazily.
    if name == "rag_chain":
        return get_chain()
    raise AttributeError(name)

import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src import config


def extract_article_range(text: str) -> str:
    articles = re.findall(r"第[一二三四五六七八九十百千零\d]+条", text)
    if not articles:
        return ""
    nums = []
    for a in articles:
        m = re.search(r"\d+", a)
        if m:
            nums.append(int(m.group()))
    return f"{articles[0]}~第{nums[-1]}条" if len(nums) > 1 else articles[0]


def load_law_documents(laws_dir: str = None) -> list[Document]:
    if laws_dir is None:
        laws_dir = config.LAWS_DIR

    docs = []
    for fname in os.listdir(laws_dir):
        if not fname.endswith(".md"):
            continue
        fpath = os.path.join(laws_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        source_url = lines[0].strip() if lines else ""
        law_name = lines[1].strip() if len(lines) > 1 else fname
        content = "".join(lines[2:]).strip()
        content = re.sub(r"\n\s*\n", "\n", content)
        content = re.sub(r"[\u3000 ]+", "", content)

        doc = Document(
            page_content=content,
            metadata={
                "law_name": law_name,
                "source_file": fname,
                "source_url": source_url,
            },
        )
        docs.append(doc)
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["article_range"] = extract_article_range(chunk.page_content)

    return chunks

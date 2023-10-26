from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS


def get_chunk_text(text):
    
    text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 500,
    chunk_overlap = 20,
    length_function = len
    )

    chunks = text_splitter.split_text(text)

    return chunks
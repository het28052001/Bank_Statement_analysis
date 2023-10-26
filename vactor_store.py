from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def get_vector_store(text_chunks):
    
    # For OpenAI Embeddings
    
    # embeddings = OpenAIEmbeddings()
    
    # For Huggingface Embeddings

    embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")

    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    
    return vectorstore
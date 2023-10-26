from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub

def get_conversation_chain(vector_store):
    
    # OpenAI Model

    # llm = ChatOpenAI()

    # HuggingFace Model

    llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b", huggingfacehub_api_token="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", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vector_store.as_retriever(),
        memory = memory
    )

    return conversation_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms.groq import ChatGroq
from config import GROQ_API_KEY, GROQ_MODEL

def get_conversational_chain(retriever):
    """
    Returns a LangChain Q&A chain using Groq's Mixtral LLM and a retriever.
    """
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0.3
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=True
    )

    return chain

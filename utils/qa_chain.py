from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from config import GROQ_API_KEY, GROQ_MODEL

def get_conversational_chain(retriever):
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0.3
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,  # ✅ this MUST be a retriever, not vectorstore
        return_source_documents=True,
        verbose=True
    )

    return chain


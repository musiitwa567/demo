from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub


import os



class ChatBot():
  DB_FAISS_PATH = r'db_faiss'
  embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
  db = FAISS.load_local(DB_FAISS_PATH, embeddings,allow_dangerous_deserialization=True)
  repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
  llm = HuggingFaceHub(
      repo_id=repo_id, model_kwargs={"temperature": 0.1, "top_p": 0.8, "top_k": 50}, huggingfacehub_api_token=os.getenv('""')
  )
  from langchain import PromptTemplate

  template = """
  You are a teacher. The students will ask you a questions about their life. Use following piece of context to answer the question.
  If you don't know the answer, just say you don't know.
  You answer with short and concise answer and donot leave any sentence hanging.

  Context: {context}
  Question: {question}
  Answer:

  """

  prompt = PromptTemplate(template=template, input_variables=["context", "question"])

  from langchain.schema.runnable import RunnablePassthrough
  from langchain.schema.output_parser import StrOutputParser

  rag_chain = (
    {"context": db.as_retriever(),  "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
  )

import streamlit as st

bot = ChatBot()

st.set_page_config(page_title="Student Companion Assistant Chat Bot")
with st.sidebar:
    st.title('PRIMARY GPT')

# Function for generating LLM response
def generate_response(input):
    result = bot.rag_chain.invoke(input)
    return result

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Welcome, let's answer your question."}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

# Generate a new response if last message is not from assistant
#if st.session_state.messages[-1]["role"] != "assistant":
    #with st.chat_message("assistant"):
       #with st.spinner("Thinking about Your Question.."):
            #response = generate_response(input)
            #st.write(response)
    #message = {"role": "assistant", "content": response}
    #st.session_state.messages.append(message)
# Generate a new response if the last message is not from the assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.spinner("Thinking about Your Question.."):
        response = generate_response(input)
        # Extracting only the answer from the response
        answer_index = response.find("Answer:")
        if answer_index != -1:
            answer = response[answer_index + len("Answer:"):].strip()
            message = {"role": "assistant", "content": answer}
            st.session_state.messages.append(message)
            with st.chat_message("assistant"):
                st.write(answer)
        else:
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)
            with st.chat_message("assistant"):
                st.write(response)    
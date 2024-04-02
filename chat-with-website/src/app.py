import streamlit as st
import importlib
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from transformers import pipeline, AutoTokenizer, AutoModel
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.preprocessing.text import Tokenizer


def get_responce(user_input):
    return "I don't know"

def get_vectorstore_form_url(url):
  # ... (existing code for loading and splitting text)

  # Use TensorFlow for document embedding
  tokenizer = Tokenizer(num_words=None)
  model = TextVectorization('universal_sentence_encoder')  # Or other models like 'average_word_vec'
  maxlen = 1000

  # Encode each document chunk into a vector representation
  document_embeddings = []
  for chunk in document_chunks:
    sequences = tokenizer.texts_to_sequences([chunk])
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen)  # Define maxlen based on model requirements
    embeddings = model.predict(padded_sequences)
    document_embeddings.append(embeddings.squeeze(0))  # Extract document embedding

  # Create a vectorstore from the chunks and their embeddings
  vector_store = Chroma.from_embeddings(document_chunks, document_embeddings)

  return vector_store


def get_context_retriever_chain(vector_store):

  retriever = pipeline('question-answering', model='sentence-transformers/all-mpnet-base-v2')  

  prompt = ChatPromptTemplate.form_messages([
      MessagesPlaceholder(variable_nema="chat_history"),
      ('user', "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
  ])

  def query_fn(chat_history, input):
    # Combine chat history and user input for query generation
    combined_text = "\n".join([message.content for message in chat_history] + [input])
    # Use the pipeline to generate a search query based on the combined text
    answer = retriever(combined_text, question=input)
    return answer['answer']  # Extract the answer (search query)

  retriever_chain = create_history_aware_retriever(query_fn, retriever, prompt)
  return retriever_chain


# app config
st.set_page_config(page_title = "chat-with-websites", page_icon = "🤖")
st.title("Chat With WebSites")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content = "Hello, How can I help you"),
    ]


# sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("WebSite URL")


if website_url is None or website_url == "":
    st.info("Please enter website URl")
else:
    document_chunks = get_vectorstore_form_url(website_url)
    with st.sidebar:
        st.write(document_chunks)

    # user input
    user_query = st.chat_input("Type your message")
    if user_query is not None and user_query != "":
        retrieved_documents = retriever_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_query
    })
    # Use the retrieved_documents (search query) to find relevant website content
    # ... (implementation for finding relevant content based on the search query)
    st.write(retrieved_documents)


    #concersation   
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
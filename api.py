from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_function

import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv


load_dotenv()
app = FastAPI()
model = ChatOpenAI()
embeddings = OpenAIEmbeddings()
output_parser = StrOutputParser()
DB_PATH = "./chroma_db"

class Message(BaseModel):
    type: str
    text: str

class RetrieveMemory(BaseModel):
    """Call this to retrieve memories from the database"""
    query: str = Field(description="search query")

retrieve_func = convert_to_openai_function(RetrieveMemory)

def retrieve_docs(query: str, retriever):
    return retriever.get_relevant_documents(query)

def contextualize_query(chat_history, question):
    messages = []
    for message in chat_history:
        if message.type == 'human':
            messages.append(HumanMessage(message.text))
        elif message.type == 'ai':
            messages.append(AIMessage(message.text))

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    contextualize_q_chain = contextualize_q_prompt | model | output_parser
    contextualize_query = contextualize_q_chain.invoke({"chat_history": messages, "question": question})
    return contextualize_query


async def stream_response(iterable):
    for item in iterable:
        yield item


@app.post("/summarize")
def summarize(chat: str):
    prompt = ChatPromptTemplate.from_template(
        "You should summarize this conversation, by listing down memorable activities throughout the day: {chat}"
    )

    chain = prompt | model | output_parser
    summary = chain.invoke({"chat": chat})

    date_string = datetime.now().strftime("%Y-%m-%d")
    activity = f"Daily Activites on {date_string}: \n{summary}"

    Chroma.from_texts([activity], embeddings, persist_directory=DB_PATH, collection_name='memory')
    return {'summary': activity}


@app.post("/query")
def query(question: str, chat_history: List[Message]):
    retriever = Chroma(embedding_function=embeddings, persist_directory=DB_PATH, collection_name='memory').as_retriever()
    if chat_history:
        question = contextualize_query(chat_history, question)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a dementia assistant, and your job is to refresh elderly memories by recalling events from the past and encourage them to storytell their day."),
        ("user", "{question}")
    ])
    model_with_function = model.bind(functions=[retrieve_func])
    chain = prompt | model_with_function
    response = chain.invoke({"question": question})
    
    if response.additional_kwargs:
        args = json.loads(response.additional_kwargs['function_call']['arguments'])
        docs = retrieve_docs(args['query'], retriever)
        template = """Answer the question based only on the following context: {context}
        Question: {question}
        If the context does not contain the answer, please respond with "I don't know".
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model | output_parser
        streamer = chain.stream({"context": docs, "question": question})
        return StreamingResponse(stream_response(streamer), media_type="text/event-stream")
    else:
        return response.content

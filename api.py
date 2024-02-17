from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_function

import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv


load_dotenv()
app = FastAPI()
model = ChatOpenAI()
embeddings = OpenAIEmbeddings()
output_parser = StrOutputParser()
vector_store = Chroma(embedding_function=embeddings, persist_directory="./chroma_db", collection_name="memory")
date = datetime.now().date()

class Message(BaseModel):
    type: str
    text: str

class RetrieveMemory(BaseModel):
    """Call this to retrieve memories from the database"""
    query: str = Field(description="search query")

retrieve_func = convert_to_openai_function(RetrieveMemory)

def get_date_stamp(date):
    return int(datetime(date.year, date.month, date.day).timestamp())

def retrieve_docs(query: str, retriever):
    documents = retriever.get_relevant_documents(query)
    docs_str = '\n\n'.join([doc.page_content for doc in documents])
    return docs_str

def convert_chat(chat_history):
    messages = []
    for message in chat_history:
        if message.type == 'human':
            messages.append(HumanMessage(message.text))
        elif message.type == 'ai':
            messages.append(AIMessage(message.text))
    return messages

async def stream_response(iterable):
    for item in iterable:
        yield item


@app.post("/summarize")
def summarize(chat_history: List[Message]):
    messages = convert_chat(chat_history)
    summarize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You should summarize this conversation, by listing down memorable activities throughout the day."),
            MessagesPlaceholder(variable_name="chat_history"),
        ]
    )

    chain = summarize_prompt | model | output_parser
    summary = chain.invoke({"chat_history": messages})
    activity = f"Daily Activites on {date}: \n{summary}"
    vector_store.add_texts([activity], [{'date_stamp': get_date_stamp(date)}])
    return {'summary': activity}


@app.post("/query")
def query(question: str, chat_history: List[Message]):
    messages = convert_chat(chat_history)
    general_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a dementia assistant, and your job is to refresh elderly memories by recalling events from the past and encourage them to storytell their day. Below is the chat history, use it as context."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "Answer this question: {question}")
    ])

    model_with_function = model.bind(functions=[retrieve_func])
    chain = general_prompt | model_with_function
    response = chain.invoke({"question": question, "chat_history": messages})
    
    if response.additional_kwargs:
        args = json.loads(response.additional_kwargs['function_call']['arguments'])
        print('Query:', args['query'])
        retriever = vector_store.as_retriever(search_kwargs={'k': 1})
        docs = retrieve_docs(args['query'], retriever)
        print('Documents:\n', docs)

        template = """Answer the question based only on the following context: {context}
        Question: {question}
        If the context does not contain the answer, please respond with "I don't know".
        """
        rag_prompt = ChatPromptTemplate.from_template(template)

        chain = rag_prompt | model | output_parser
        streamer = chain.stream({"context": docs, "question": question})
        return StreamingResponse(stream_response(streamer), media_type="text/event-stream")
    else:
        return response.content


@app.get('/flashcard')
def flashcard(n: int):
    stamp = get_date_stamp(date - timedelta(days=7))
    res = vector_store._collection.get(where={'date_stamp': {'$gt': stamp}})
    doc_str = '\n\n'.join(res['documents'])

    flashcard_prompt = ChatPromptTemplate.from_template('Create {n} flashcard(s) based on the following context: {documents}. Return a list of dictionaries, consist of "question" and "answer" keys. Parse in JSON format.')
    chain = flashcard_prompt | model | output_parser
    flashcards = chain.invoke({'n': n, 'documents': doc_str})
    return json.loads(flashcards)

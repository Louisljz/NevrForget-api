from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.utils.function_calling import convert_to_openai_function

from trulens_eval import Tru, TruChain
from trulens_feedback import init_rag_feedbacks, init_sum_feedbacks, init_card_feedbacks

import json
from fastapi import FastAPI
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv


load_dotenv()
app = FastAPI()
tru = Tru()
model = ChatOpenAI()
embeddings = OpenAIEmbeddings()
output_parser = StrOutputParser()
vector_store = Chroma(embedding_function=embeddings, persist_directory="./chroma_db", collection_name="memory")
date = datetime.now().date()


class Message(BaseModel):
    type: str
    text: str

class RetrieveMemory(BaseModel):
    """Call this to retrieve memories from the database. For example, user activities, events, etc."""
    query: str = Field(description="query for vector search")

retrieve_func = convert_to_openai_function(RetrieveMemory)

def get_date_stamp(date):
    return int(datetime(date.year, date.month, date.day).timestamp())

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def convert_chat(chat_history):
    messages = []
    for message in chat_history:
        if message.type == 'human':
            messages.append(HumanMessage(message.text))
        elif message.type == 'ai':
            messages.append(AIMessage(message.text))
    return messages


@app.post("/summarize")
def summarize(chat_history: List[Message]):
    messages = convert_chat(chat_history)
    summarize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You should summarize this conversation, by listing down memorable activities throughout the day. Write in bullet points."),
            MessagesPlaceholder(variable_name="chat_history"),
        ]
    )

    sum_chain = summarize_prompt | model | output_parser
    sum_feedbacks = init_sum_feedbacks()
    sum_recorder = TruChain(sum_chain,
                            app_id='Summary',
                            feedbacks=sum_feedbacks)
    
    with sum_recorder:
        summary = sum_chain.invoke({"chat_history": messages})
    activity = f"Daily Activites on {date}: \n{summary}"
    vector_store.add_texts([activity], [{'date_stamp': get_date_stamp(date)}])
    return activity


@app.post("/query")
def query(question: str, chat_history: List[Message]):
    messages = convert_chat(chat_history)
    general_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a dementia assistant, and your job is to refresh elderly memories by recalling events from the past and encourage them to storytell their day. Below is the chat history, use it as context."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "Answer this question: {question}")
    ])

    model_with_function = model.bind(functions=[retrieve_func])
    general_chain = general_prompt | model_with_function
    response = general_chain.invoke({"question": question, "chat_history": messages})
    
    if response.additional_kwargs:
        args = json.loads(response.additional_kwargs['function_call']['arguments'])
        retriever = vector_store.as_retriever(search_kwargs={'k': 1})

        template = """Answer the question based only on the following context: {context}
        Question: {question}
        If the context does not contain the answer, please respond with "I don't know".
        """
        rag_prompt = ChatPromptTemplate.from_template(template)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | rag_prompt
            | model
            | output_parser
        )
        rag_feedbacks = init_rag_feedbacks(rag_chain)
        rag_recorder = TruChain(rag_chain,
                                app_id='Conversational Retrieval Chain',
                                feedbacks=rag_feedbacks)
        
        with rag_recorder:
            answer = rag_chain.invoke(args['query'])
        return answer
    else:
        return response.content


@app.get('/flashcard')
def flashcard(n: int):
    stamp = get_date_stamp(date - timedelta(days=7))
    res = vector_store._collection.get(where={'date_stamp': {'$gt': stamp}})
    doc_str = '\n\n'.join(res['documents'])

    flashcard_prompt = ChatPromptTemplate.from_template('Create {n} flashcard(s) based on the following context: {documents}. Return a list of dictionaries, consist of "question" and "answer" keys. Parse in JSON format.')
    card_chain = flashcard_prompt | model | output_parser
    card_feedbacks = init_card_feedbacks()
    card_recorder = TruChain(card_chain,
                            app_id='Flashcards',
                            feedbacks=card_feedbacks)
    
    with card_recorder:
        flashcards = card_chain.invoke({'n': n, 'documents': doc_str})
    return json.loads(flashcards)


@app.post("/base-query")
def base_query(question: str):
    retriever = vector_store.as_retriever(search_kwargs={'k': 1})

    template = """Answer the question based only on the following context: {context}
    Question: {question}
    If the context does not contain the answer, please respond with "I don't know".
    """
    rag_prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | model
        | output_parser
    )
    rag_feedbacks = init_rag_feedbacks(rag_chain)
    rag_recorder = TruChain(rag_chain,
                            app_id='Basic Retrieval Chain',
                            feedbacks=rag_feedbacks)
    
    with rag_recorder:
        answer = rag_chain.invoke(question)
    return answer

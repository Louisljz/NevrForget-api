import requests
from urllib.parse import quote
import json

def send_query(question, chat_history):
    url = f'http://localhost:8000/query?question={quote(question)}'
    payload = json.dumps(chat_history)
    response = requests.post(url, data=payload)
    return response.text

if __name__ == "__main__":
    chat_history = []
    print("Dementia Assistant Chat. Type 'quit' to exit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        
        answer = send_query(user_input, chat_history)
        print("AI: ", answer)
        chat_history.append({"type": "human", "text": user_input})
        chat_history.append({"type": "ai", "text": answer})

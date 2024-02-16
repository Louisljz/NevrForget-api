import requests
from urllib.parse import quote

query = quote(input('Enter your question: '))
url = f'http://localhost:8000/query?question={query}'

response = requests.post(url, stream=True)

if response.headers.get('content-type') == 'text/event-stream; charset=utf-8':
    try:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                decoded_chunk = chunk.decode('utf-8')
                print(decoded_chunk, end='')
    except Exception as e:
        print(f"Error: {e}")
else:
    print(response.text)

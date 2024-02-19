# NevrForget API

## Endpoints

1. **Summarize**: summarize daily conversations to add to vectorstore
2. **Query**:

- Decision Making Layer: Storytelling or Retrieve-Memory
- RAG pipeline: Context and Query

3. **Flashcard**: Get n Flashcards from chat history over past week

## Run Instructions

1. `pip install -r requirements.txt`
2. `uvicorn api:app --reload`
3. Open in browser _localhost:8000/docs_
4. Type `trulens-eval` on your terminal to open evaluation dashboard.

## Server Setup

1. `gcloud init`: setup your account and project-id
2. `docker build -t gcr.io/nevr-forget/api:starter .`: Build docker image, test-run container from docker desktop
3. `docker push gcr.io/nevr-forget/api:starter`: Push the Docker image to Google Container Registry
4. Navigate to [GCloud Console](https://console.cloud.google.com/) -> Cloud Run -> Create Service -> Select image from container registry -> Configure service settings --> Deploy!

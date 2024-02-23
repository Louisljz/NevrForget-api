# NevrForget API

## Cloud Architecture
![system design](https://github.com/Louisljz/NevrForget-api/tree/main/docs/cloud-architecture.jpg)

## Trulens Evaluation

1. **RAG**:

- Groundedness: Ensures responses are based on factual or provided information, with explanations for how the conclusion was reached.
- Question-Context Relevance: Measures the relevance of each context chunk retrieved to the query.
- Question-Answer Relevance: Make sure that the response answers the question.

2. **Summary**:

- Conciseness: Ensure that the summary is expressed in concise manner.
- Coherence: Assess how logically connected and well-structured the summary is.
- Comprehensiveness: Assess how well the summary covers the key points from the original content.

3. **Flashcards**:

- Relevance: Ensure that the flashcards are relevant to the facts in the provided context.
- Correctness: Evaluating the factual accuracy of the questions and answers.
- Helpfulness: Evaluate whether the flashcards are likely to aid in learning or memorization.

## Run Instructions

1. `pip install -r requirements.txt`
2. `uvicorn api:app --reload`
3. Open in browser _localhost:8000/docs_
4. Run `open_dashboard.py` to visualize trulens evaluation dashboard.

## Server Setup

1. `gcloud init`: setup your account and project-id
2. `docker build -t gcr.io/nevr-forget/api:starter .`: Build docker image, test-run container from docker desktop
3. `docker push gcr.io/nevr-forget/api:starter`: Push the Docker image to Google Container Registry
4. Navigate to [GCloud Console](https://console.cloud.google.com/) -> Cloud Run -> Create Service -> Select image from container registry -> Configure service settings --> Deploy!

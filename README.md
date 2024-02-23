# NevrForget API

## Cloud Architecture
![system design](https://github.com/Louisljz/NevrForget-api/blob/main/docs/cloud-architecture.jpg)
Link to [mobile app repo](https://github.com/Louisljz/NevrForget-app/) 

## OpenAI models
1. GPT-3.5-Turbo for LLM
2. Text Embedding for vector search
3. Whisper n TTS for voice assistant
4. DALL-E for avatar
5. Vision (coming soon)

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

## Knowledge Source
"Daily Activites on 2024-02-22: \nSummary:\n- Started the day by gardening and admiring the blooming roses\n- Spent time talking and offering support to a neighbor going through a tough time\n- Enjoyed a long chat over tea with the neighbor\n- Relaxed at home, read a mystery novel, and listened to music\n- Planning to try a new recipe for dinner - lasagna\n- Cooking is another hobby\n- Expressed gratitude for the opportunity to share moments, even virtually"

"Daily Activites on 2024-02-23: \nSummary:\n- Morning walk in the garden to admire the blooming roses\n- Taking pictures of the beautiful roses\n- Uploading photos to share with family via email\n- Knitting a bright blue sweater for grandson's birthday\n- Enjoying afternoon tea"

"Daily Activites on 2024-02-24: \nSummary:\n- Went for a walk in the park and met an old friend\n- Caught up with the old friend, talked about grandchildren and life changes\n- Tried a new recipe for dinner - vegetarian lasagna\n- Experimenting with cooking and learning from the experience"

## Potential Questions
1. Hello, I enjoy my trip to Singapore.
2. What did I do on 22 Feb?
3. How about the next day?
4. What new recipe did I try for dinner?
5. When did I knit a blue sweater?

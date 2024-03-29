# NevrForget API

## Cloud Architecture
![system design](https://github.com/Louisljz/NevrForget-api/blob/main/docs/cloud-architecture.jpg)
Link to [mobile app repo](https://github.com/jacdevv/nvr_forget) 

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
These are retrieved from your conversation with AI.

"Daily Activites on 2024-02-22: \n- Enjoyed a cup of tea in the morning\n- Tended to the blooming roses in the garden\n- Shared memories of the garden with red roses\n- Planned to bake cookies for grandchildren\n- Discussed a family recipe with cinnamon\n- Offered gardening tips for later\n- Expressed gratitude for the conversation"

"Daily Activites on 2024-02-23: \n- Gardening and tending to blooming roses\n- Reading historical fiction\n- Seeking book recommendations"

"Daily Activites on 2024-02-24: \nSummary of the day's activities:\n- Watching the sunrise\n- Enjoying favorite tea\n- Gardening plans: tending to roses and planting new herbs like basil and mint\n- Seeking tips for caring for wilted roses and planting herbs\n- Tips for roses: ensure enough water, check for pests, use natural fertilizer, prune dead heads\n- Tips for herbs: basil and mint thrive in well-drained soil and sunlight; plant mint in a pot to contain spread\n- Remember to water herbs according to their needs and consider talking to your plants for growth."

## Potential Questions
1. Hello, I had a wonderful trip to the botanic gardens today.
2. What did I do on 22 Feb?
3. How about the next day?
4. What gardening plans do I have?
5. What are some tips I learnt to grow roses?

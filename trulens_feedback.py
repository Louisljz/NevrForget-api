from trulens_eval import Feedback
from trulens_eval.app import App
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider import OpenAI
import numpy as np

openai = OpenAI()

def init_rag_feedbacks(rag_chain):
    context = App.select_context(rag_chain)
    grounded = Groundedness(groundedness_provider=OpenAI())
    f_groundedness = (
        Feedback(grounded.groundedness_measure_with_cot_reasons)
        .on(context.collect())
        .on_output()
        .aggregate(grounded.grounded_statements_aggregator)
    )

    f_qa_relevance = Feedback(openai.relevance).on_input_output()
    f_context_relevance = (
        Feedback(openai.qs_relevance)
        .on_input()
        .on(context)
        .aggregate(np.mean)
    )

    return [f_groundedness, f_qa_relevance, f_context_relevance]

def init_sum_feedbacks():
    return [Feedback(openai.comprehensiveness_with_cot_reasons).on_input_output(),
            Feedback(openai.coherence).on_output(),
            Feedback(openai.conciseness).on_output()]

def init_card_feedbacks():
    return [Feedback(openai.relevance_with_cot_reasons).on_input_output(),
            Feedback(openai.correctness).on_output(),
            Feedback(openai.helpfulness).on_output()]

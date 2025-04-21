import os
from dotenv import load_dotenv, find_dotenv
import numpy as np
from trulens.providers.openai import OpenAI
from trulens_eval import TruChain
from trulens.core import Feedback

def get_openai_api_key():
    _ = load_dotenv(find_dotenv())
    return os.getenv("OPENAI_API_KEY")


openai_provider = OpenAI()

# Feedback definitions for RAG evaluation
qa_relevance = Feedback(openai_provider.relevance_with_cot_reasons, name="Answer Relevance").on_input_output()
qs_relevance = Feedback(openai_provider.context_relevance_with_cot_reasons, name="Context Relevance").on_input_output()
groundedness = Feedback(openai_provider.groundedness_measure_with_cot_reasons, name="Groundedness").on_input_output()

feedbacks = [qa_relevance, qs_relevance, groundedness]

def get_trulens_recorder(chain, app_id):
    """Return a TruChain recorder for the given LangChain chain and app_id, with prebuilt feedbacks."""
    tru_recorder = TruChain(
        chain,
        app_id=app_id,
        feedbacks=feedbacks
    )
    return tru_recorder

# Optionally, expose feedbacks for direct use
__all__ = [
    "get_openai_api_key",
    "qa_relevance",
    "qs_relevance",
    "feedbacks",
    "get_trulens_recorder"
]

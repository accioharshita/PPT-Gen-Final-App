from crewai import LLM
from .config import get_llm_config

def get_llm(model_name=None):
    config = get_llm_config(model_name)
    return LLM(
        model=config["model"],
        api_key=config["api_key"]
    )

import os

LLM_CONFIGS = {
    "openai": {
        "available_models": ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"],
        "default_model": "gpt-4o-mini",
        "api_key": os.getenv('OPENAI_API_KEY')
    },
    "groq": {
        "model": "groq/llama3-groq-70b-8192-tool-use-preview", 
        "api_key": os.getenv('GROQ_API_KEY')
    },
    "anthropic": {
        "model": "anthropic/claude-3-5-sonnet-20240620",
        "api_key": os.getenv('ANTHROPIC_API_KEY')
    }
}

def get_llm_config(model_name=None):
    config = LLM_CONFIGS["openai"].copy()
    if model_name:
        config["model"] = model_name
    else:
        config["model"] = config["default_model"]
    return config

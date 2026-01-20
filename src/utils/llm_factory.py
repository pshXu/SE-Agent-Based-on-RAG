import os
from langchain_openai import ChatOpenAI

def get_llm():
    """
    Factory function to get an LLM instance based on environment variables.
    Directly uses OpenAI-compatible variables for maximum flexibility.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set. Please check your .env file.")
    
    # Custom API base URL (Optional, defaults to OpenAI official)
    api_base = os.getenv("OPENAI_API_BASE")
    
    # Model configuration
    model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", 0.7))
    
    # Return a generic ChatOpenAI client that works with any compatible backend
    return ChatOpenAI(
        api_key=api_key,
        base_url=api_base, # Works for OpenAI, proxies, DeepSeek, etc.
        model=model_name,
        temperature=temperature
    )


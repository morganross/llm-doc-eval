import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

def get_llm(model_name: str, temperature: float = 0.0) -> BaseChatModel:
    """
    Returns a LangChain LLM instance based on the model name.

    Args:
        model_name (str): The name of the model (e.g., "o4-mini-2025-04-16", "gemini-2.5-flash-preview-05-20").
        temperature (float): The temperature setting for the LLM.
                             Note: Some models may not support temperature=0.0.

    Returns:
        BaseChatModel: An instance of a LangChain chat model.

    Raises:
        ValueError: If an unsupported model name is provided.
    """
    if model_name.startswith("o4-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        # Set temperature to 1.0 for o4-mini as it doesn't support 0.0
        return ChatOpenAI(model=model_name, temperature=1.0, api_key=api_key)
    elif model_name.startswith("gemini-"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, google_api_key=api_key)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

if __name__ == '__main__':
    # Example usage (requires API keys in .env)
    # For testing, you might want to mock these calls or use dummy API keys
    try:
        print("Testing OpenAI o4-mini-2025-04-16:")
        openai_llm = get_llm("o4-mini-2025-04-16")
        # response = openai_llm.invoke("Hello, what is your name?")
        # print(response.content)

        print("\nTesting Google gemini-2.5-flash-preview-05-20:")
        google_llm = get_llm("gemini-2.5-flash-preview-05-20")
        # response = google_llm.invoke("Hello, what is your name?")
        # print(response.content)

    except ValueError as e:
        print(f"Error: {e}. Please ensure API keys are set in your .env file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
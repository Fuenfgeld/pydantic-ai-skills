import os
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# Set your OpenRouter API key in your environment or load it here
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_URL = 'https://openrouter.ai/api/v1'

# Define the provider for OpenRouter
provider = OpenAIProvider(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_URL
)

# Choose your OpenRouter model (e.g., 'gpt-4o-mini', or another supported model)
model = OpenAIChatModel(
    model_name='gpt-4o-mini',
    provider=provider
)

# (Optional) Define a Pydantic model for structured output
class Answer(BaseModel):
    response: str

# Create the agent
agent = Agent(
    model=model,
    output_type=Answer,  # or just 'str' if you want plain text
    system_prompt="You are a helpful assistant."
)

# Synchronous usage
result = agent.run_sync("What's the capital of France?")
print(result.output)  # If using result_type=Answer
# print(result.data)        # If using result_type=str
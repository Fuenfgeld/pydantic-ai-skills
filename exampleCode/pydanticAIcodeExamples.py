
"""Structured output

Pydantic AI’s core advantage is turning unstructured or semi-structured AI outputs into reliable, validated, and well-typed data-making downstream processing, automation, 
and integration much safer and easie based on pydantic data classes.Pydantic models enforce strict type and value checks at runtime, ensuring that any data generated or 
received by your AI application matches the expected structure and types."""

from pydantic import BaseModel, Field

# The docstring is an import it explains to the LLM what the data class is for
class HistoricCelebrity(BaseModel):
    """Historic significant person - includes name,year of birth and short biography"""
    name: str
    birthYear: int
    #the Field function is optional and gives additional instructions to the LLM
    shortBiography: str  = Field(...,description="The short biography should be structured first giving a short text about the person followed by a chronological list of the main events in the life of the celebrity")


# Define the agent
agent = Agent(model=openAI_model, output_type=HistoricCelebrity)

# Run the agent
result = agent.run_sync("who is Florence Nightingale?")


"""Nested Data Structure"""

import os
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel, Field

from typing import List, Optional
from datetime import date
from enum import Enum
from pprint import pprint

# Define the model
model = OpenAIModel('gpt-4o-mini')


# Enum to represent gender options — teaches use of fixed choices
class Gender(str, Enum):
    """Enum to represent gender options"""
    male = "male"
    female = "female"
    other = "other"


class Allergy(BaseModel):
    """Describes an allergy, including the substance and optional reaction details"""
    substance: str  = Field(...,description="What the patient is allergic to ")
    reaction: Optional[str]     = Field(...,description="Description of the reaction")


class Medication(BaseModel):
    """Describes a medication the patient is currently taking"""
    name: str = Field(..., description="Name of the medication ")
    dose_mg: float = Field(..., description="Dosage in milligrams ")
    frequency_per_day: int = Field(..., description="How many times per day the medication is taken")


class Condition(BaseModel):
    """Represents a past or current medical condition"""
    name: str = Field(..., description="Name of the condition (e.g., 'Asthma')") # it can be problematic to prime LLM models with examples
    diagnosed_date: Optional[date] = Field(None, description="Date the condition was diagnosed")
    chronic: bool = Field(..., description="Whether the condition is long-term")


class PatientHistory(BaseModel):
    """Captures a patient's overall medical history including medications, allergies, and diagnoses"""
    name: str = Field(..., description="Full name of the patient")
    birth_date: date = Field(..., description="Patient's date of birth")
    gender: Gender = Field(..., description="Patient's gender")
    height_cm: Optional[float] = Field(None, description="Height in centimeters")
    weight_kg: Optional[float] = Field(None, description="Weight in kilograms")
    smoker: bool = Field(..., description="Whether the patient currently smokes")

    allergies: List[Allergy] = Field(default_factory=list, description="List of known allergies")
    medications: List[Medication] = Field(default_factory=list, description="List of current medications")
    conditions: List[Condition] = Field(default_factory=list, description="List of medical conditions")

# Define the agent
patient_generator_agent = Agent(model=model,
              output_type=PatientHistory,
              system_prompt="You are a writer for synthetic patient history, the history will be used as examples for students",)

# Run the agent
result = patient_generator_agent.run_sync("Generate patient history for Jane Bond", model_settings={'temperature': 1.0})

print( f"""
The result is a pydantic data class of type: {type(result.output)}
here are the details:
""")
pprint(result.output.model_dump_json(indent=2))
     
     

"""System Prompt

System prompts set the overall context, tone, and objectives for the AI, essentially telling it "who it is" and "how it should act." For example, assigning a role like "helpful assistant" or "technical expert" through a system prompt gives the AI a framework for generating relevant and coherent responses. Dynamic system prompts significantly enhance AI responses by making AI interactions more personalized and context-aware.
"""
from pydantic_ai import Agent, RunContext
from datetime import date

agent = Agent(
    model=openAI_model,
    deps_type=str,
    system_prompt="Use the user name while replying to them.",
)

# system prompts can be altered using decorators
#here information about the run context like session ids can be injected
@agent.system_prompt
def add_user_name(ctx: RunContext[str]) -> str:
    return f"The user name is {ctx.deps}."

# here current information can be added
@agent.system_prompt
def add_the_date() -> str:
    return f'The date is {date.today()}.'

result = agent.run_sync('What is the date today?', deps='Max')
print(result.output)
     

"""Tools

Tools in Pydantic AI are functions that the language model (LLM) can call to retrieve extra information or perform specific actions to help generate a response. They are especially useful when the model needs to access data or capabilities beyond its own training or context, such as:

    Fetching up-to-date or external information (e.g., weather, database lookups)

    Performing calculations or transformations

    Accessing APIs or custom business logic

When an agent is configured with tools, the LLM can decide to "call" these tools during a conversation. The results from these tool calls are then incorporated into the AI's response, making the agent more powerful, interactive, and capable of handling complex tasks that go beyond simple text generation
"""
import random

agent = Agent(
    model=openAI_model,
    deps_type=str,
    system_prompt="you are a lab assistant giving patient their lab results and explain the result",
)

@agent.tool_plain
def getLabResult() -> str:
    """Get the Calcium level """
    return "1.45 mmol/L"

@agent.tool
def getPatientName(ctx: RunContext[str]) -> str:
    """Get the patient names"""
    return ctx.deps

result = agent.run_sync('What is my Calcium level', deps='Max')
print(result.output)

"""

Memory

LLMS are stateless and all previous messages need to be submitted to an LLM calls for the LLM to keep memory. memory refers to the agent’s ability to retain and access previous messages or conversation history during interactions. This memory is typically managed as a list of message objects (such as user requests and model responses) that are passed to the agent at runtime in pydanticAI this is message_history variable. The memory can be short-term (only available during an active session) or long-term (persisted to disk or a database for retrieval in future sessions).
"""
from pydantic_ai import Agent
from pydantic_ai.messages import (ModelMessage)
from pydantic_ai.models.openai import OpenAIModel


agent = Agent(model=openAI_model,
    system_prompt="you are a funny pirate",
)


def main_loop():
    message_history: list[ModelMessage] = []

    while True:
        user_input = input("> ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("finish")
            break

        # Run the agent
        result = agent.run_sync(user_input, deps=user_input, message_history=message_history)
        print(result.output)
        msg = result.new_messages()
        message_history.extend(msg)

main_loop()
     

"""Output Validators

Sometimes ther is a need to go byond data type validation provided by Pydantic here offers validation functions via the agent.output_validator decorator.
"""
import os
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic import BaseModel

# Define the model
model = OpenAIChatModel('gpt-4o-mini')



# Define the agent
agent = Agent(model=model,
              retries=1,
              system_prompt="see if you have been given all three dimesion to describe the size of a box (width height depth). Answer with a single word either 'correct' or 'wrong'",)

# Define the result validator
@agent.output_validator
def output_validator_simple(data: str) -> str:

    print(f"output_validatorinput data:{data}" )
    if 'wrong' in data.lower():
        raise ModelRetry('wrong response')
    return data

# Run the agent
result = agent.run_sync("The box is 10x20x30 cm")
print(result.output)

result = agent.run_sync("The box is 10")
print(result.output)
     


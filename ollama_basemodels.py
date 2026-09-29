from pydantic import BaseModel, Field
from typing import Literal
"""
Pydantic Basemodels for local LLMs 
Define pydantic BaseModel similar to Dataclass
"""

Label = Literal["VAGUE_LOC", "CONCEPTUAL_LOC"]


class Span(BaseModel):
    start: int
    end: int
    text: str
    label: Label


class Fairytale(BaseModel):
    khm: int
    title: str
    text: str
    spans: list[Span] = Field(default_factory=list)


class FairytaleChunk(BaseModel):
    khm: int
    title: str
    text: str
    start: int
    end: int
    chunk_id: int
    spans: list[Span] = Field(default_factory=list)


# LLM class without start+end
class LLMSpan(BaseModel):
    text: str
    label: Label

# LLM class don't need to respond whole fairytale again
class LLMFairytaleAnnotation(BaseModel):
    spans: list[LLMSpan]
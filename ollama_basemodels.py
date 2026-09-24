from pydantic import BaseModel, Field
from typing import Literal
"""
Pydantic Basemodels for local LLMs 
Define pydantic BaseModel similar to Dataclass
"""

Label = Literal["VAGUE_LOC", "CONCEPTUAL_LOC"]

class Fairytale(BaseModel):
    khm: int
    title: str
    text: str


class Span(BaseModel):
    start: int
    end: int
    text: str
    label: Label


class FairytaleChunk(BaseModel):
    khm: int
    title: str
    text: str
    start: int
    end: int
    chunk_id: int
    spans: list[Span] = Field(default_factory=list)


class FairytaleAnnotation(BaseModel):
    fairytale: Fairytale
    spans: list[Span]


# LLM class without start+end
class LLMSpan(BaseModel):
    text: str
    label: Label


class LLMFairytaleAnnotation(BaseModel):
    spans: list[LLMSpan]
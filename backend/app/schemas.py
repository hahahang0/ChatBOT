from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]

    content: str = Field(
        min_length=1,
        max_length=12000,
    )


class ChatResponse(BaseModel):
    reply: str
    model: str

    sources: list[dict] = Field(
        default_factory=list,
    )
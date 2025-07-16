# 이 파일은 API 요청/응답의 데이터 구조(Schema)만을 정의합니다.
import time
import uuid
from pydantic import BaseModel, Field
from typing import List

# --- /v1/chat/completions 용 모델 ---

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    user: str | None = None
    stream: bool = False

class ChatCompletionResponseChoice(BaseModel):
    index: int = 0
    message: ChatMessage

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]

# --- 스트리밍 응답 '조각(Chunk)'을 위한 모델 ---

class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None

class ChatCompletionStreamResponseChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: str | None = None

class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionStreamResponseChoice]

# --- /v1/models 응답을 위한 모델 ---

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "user"

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]
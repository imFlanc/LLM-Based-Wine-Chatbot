
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Dict

from langchain.chains import ConversationChain
from chatbot import init_chatbot, model_id, chain_stream_generator, fake_stream_generator

from schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatCompletionResponseChoice,
    ModelList,
    ModelCard,
)

# 이 딕셔너리가 여러 사용자의 대화 세션을 관리합니다.
# key: 세션 ID, value: 해당 세션의 ConversationChain 인스턴스
SESSIONS: Dict[str, ConversationChain] = {}
# --- 개발용 플래그: 모델 로딩 없이 테스트하려면 True로 설정 ---
USE_FAKE_CHATBOT = True
MODEL_NAME = "Test-Model-Hardcoded"

# FastAPI 세팅
app = FastAPI(
    title="Custom ConversationChain Server",
    description="An API server that wraps a legacy ConversationChain for Open Web UI.",
)

# cors 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """
    Open Web UI에 제공할 모델 목록을 반환합니다.
    """
    model_card = ModelCard(id=model_id)
    return ModelList(data=[model_card])

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_handler(request: ChatCompletionRequest):
    """
    Open Web UI의 요청을 받아 처리하는 메인 핸들러
    """
    # 요청 메시지 리스트에서 마지막 사용자 입력을 추출합니다. (대화 기록은 chain의 memory가 관리하므로 마지막 질문만 필요)
    user_input = request.messages[-1].content
    
    # 테스트 플래그 사용 시
    if USE_FAKE_CHATBOT:
        print("--- FAKE CHATBOT MODE ---")
        # 가짜 제너레이터를 사용하여 즉시 응답 스트리밍 시작
        return StreamingResponse(
            fake_stream_generator(MODEL_NAME, user_input),
            media_type="text/event-stream"
        )
        
    # 세션 ID를 결정합니다. Open Web UI가 `user` 필드를 보내주면 그것을 사용합니다.
    session_id = request.user or "default_session"

    # 세션 ID에 해당하는 챗봇 인스턴스가 있는지 확인
    if session_id not in SESSIONS:
        # 없다면 새로 생성하여 딕셔너리에 저장
        SESSIONS[session_id] = init_chatbot()
        print(f"새로운 세션 생성: {session_id}")
    
    # 해당 세션의 챗봇 인스턴스를 가져옵니다.
    chain = SESSIONS[session_id]
        
    if request.stream:
        # 스트리밍 요청이면, chain_stream_generator를 사용하여 응답을 실시간으로 보냅니다.
        return StreamingResponse(
            chain_stream_generator(chain, user_input, model_id), # <--- 실제 체인을 전달
            media_type="text/event-stream"
        )
    else:
        # 비스트리밍 요청이면, 기존처럼 .predict()를 사용합니다.
        response_text = chain.run(user_input)
        response_message = ChatMessage(role="assistant", content=response_text)
        choice = ChatCompletionResponseChoice(message=response_message)
        return ChatCompletionResponse(model=model_id, choices=[choice])
